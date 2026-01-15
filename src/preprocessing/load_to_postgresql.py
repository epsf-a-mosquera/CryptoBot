# src/preprocessing/load_to_postgresql.py
"""
load_to_postgresql.py (version "prod")
-------------------------------------
Charge les Parquet "clean / features / ml" dans PostgreSQL avec idempotence.

✅ Charge :
- data/processed/*_clean.parquet   -> table klines_clean
- data/features/*_features.parquet -> table features_1h
- data/processed/*_ml.parquet      -> table ml_dataset_1h

✅ Garanties :
- Table de tracking loaded_files (évite rechargement fichier)
- Idempotence niveau ligne via clé unique (symbol, kline_interval, timestamp)
- Insertion performante via psycopg2.extras.execute_values
- ON CONFLICT DO NOTHING pour éviter doublons

Pré-requis :
    pip install psycopg2-binary sqlalchemy python-dotenv pandas pyarrow

Usage :
    python src/preprocessing/load_to_postgresql.py
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

import psycopg2
import psycopg2.extras

from src.utils.config import (
    ensure_directories,
    PROCESSED_DIR,
    FEATURES_DIR,
)

# -------------------------
# Logging
# -------------------------
ensure_directories()
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/load_to_postgresql.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.getLogger().addHandler(console)

logger = logging.getLogger("load_to_postgresql_prod")

# -------------------------
# Env / DB
# -------------------------
load_dotenv()

DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "cryptobot_db")

TRACK_TABLE = "loaded_files"


def get_engine() -> Engine:
    url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url, pool_pre_ping=True)


engine = get_engine()


def get_psycopg2_conn():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )


# -------------------------
# Tracking table
# -------------------------
def create_tracking_table() -> None:
    q = text(
        f"""
        CREATE TABLE IF NOT EXISTS {TRACK_TABLE} (
            file_name TEXT PRIMARY KEY,
            table_name TEXT NOT NULL,
            loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    with engine.begin() as conn:
        conn.execute(q)
    logger.info(f"✅ Tracking table '{TRACK_TABLE}' ready.")


def already_loaded(file_name: str) -> bool:
    q = text(f"SELECT 1 FROM {TRACK_TABLE} WHERE file_name = :fn LIMIT 1;")
    with engine.connect() as conn:
        row = conn.execute(q, {"fn": file_name}).fetchone()
        return row is not None


def mark_as_loaded(file_name: str, table_name: str) -> None:
    q = text(
        f"""
        INSERT INTO {TRACK_TABLE} (file_name, table_name)
        VALUES (:fn, :tn)
        ON CONFLICT (file_name) DO NOTHING;
        """
    )
    with engine.begin() as conn:
        conn.execute(q, {"fn": file_name, "tn": table_name})


# -------------------------
# Table schemas (DDL)
# -------------------------
def create_tables_if_needed() -> None:
    """
    DDL minimal "prod" :
    - klines_clean : colonne de base OHLCV + trades
    - features_1h  : features (beaucoup de colonnes -> on les crée dynamiquement lors du 1er chargement)
    - ml_dataset_1h: features + label/label_int (+ target_* optionnel)
    """
    # Table klines_clean: schéma fixe
    q1 = text(
        """
        CREATE TABLE IF NOT EXISTS klines_clean (
            symbol TEXT NOT NULL,
            kline_interval TEXT NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            open_time BIGINT,
            open DOUBLE PRECISION,
            high DOUBLE PRECISION,
            low DOUBLE PRECISION,
            close DOUBLE PRECISION,
            volume DOUBLE PRECISION,
            number_of_trades BIGINT,
            PRIMARY KEY (symbol, kline_interval, timestamp)
        );
        """
    )

    # features_1h et ml_dataset_1h : on crée une base minimale, puis on ajoutera des colonnes au besoin
    q2 = text(
        """
        CREATE TABLE IF NOT EXISTS features_1h (
            symbol TEXT NOT NULL,
            kline_interval TEXT NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (symbol, kline_interval, timestamp)
        );
        """
    )

    q3 = text(
        """
        CREATE TABLE IF NOT EXISTS ml_dataset_1h (
            symbol TEXT NOT NULL,
            kline_interval TEXT NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            label TEXT,
            label_int SMALLINT,
            PRIMARY KEY (symbol, kline_interval, timestamp)
        );
        """
    )

    with engine.begin() as conn:
        conn.execute(q1)
        conn.execute(q2)
        conn.execute(q3)

    logger.info("✅ Base tables ready: klines_clean, features_1h, ml_dataset_1h")


def ensure_columns(table: str, df: pd.DataFrame, base_cols: List[str]) -> None:
    """
    Ajoute les colonnes manquantes dans une table (ALTER TABLE ADD COLUMN).
    - base_cols: colonnes déjà prévues / clés
    - le reste des colonnes df => ajoutées si absentes
    Types :
      - float -> DOUBLE PRECISION
      - int   -> BIGINT
      - bool  -> BOOLEAN
      - object-> TEXT
      - datetime -> TIMESTAMPTZ
    """
    # Récupère les colonnes existantes
    q = text(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = :t;
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(q, {"t": table}).fetchall()
    existing = {r[0] for r in rows}

    to_add = [c for c in df.columns if c not in existing]
    if not to_add:
        return

    # Déduction types SQL
    def sql_type_for(col: str) -> str:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            return "TIMESTAMPTZ"
        if pd.api.types.is_bool_dtype(s):
            return "BOOLEAN"
        if pd.api.types.is_integer_dtype(s):
            return "BIGINT"
        if pd.api.types.is_float_dtype(s):
            return "DOUBLE PRECISION"
        # fallback
        return "TEXT"

    alter_stmts = []
    for c in to_add:
        # sécurité: éviter les noms bizarres
        if not c.replace("_", "").isalnum():
            raise ValueError(f"Nom de colonne invalide: {c}")
        alter_stmts.append(f'ALTER TABLE {table} ADD COLUMN IF NOT EXISTS "{c}" {sql_type_for(c)};')

    with engine.begin() as conn:
        for stmt in alter_stmts:
            conn.execute(text(stmt))

    logger.info(f"🧱 {table}: added {len(to_add)} column(s): {to_add[:8]}{'...' if len(to_add)>8 else ''}")


# -------------------------
# Routing (file -> table)
# -------------------------
def route_file_to_table(file_path: Path) -> Optional[str]:
    name = file_path.name.lower()
    if name.endswith("_clean.parquet"):
        return "klines_clean"
    if name.endswith("_features.parquet"):
        return "features_1h"
    if name.endswith("_ml.parquet"):
        return "ml_dataset_1h"
    return None


# -------------------------
# Insert with upsert
# -------------------------
def insert_upsert(table: str, df: pd.DataFrame, key_cols: List[str]) -> None:
    """
    Insert en masse (execute_values) avec ON CONFLICT DO NOTHING (idempotent).
    """
    if df.empty:
        return

    cols = list(df.columns)
    values = df.to_records(index=False).tolist()

    col_sql = ", ".join([f'"{c}"' for c in cols])
    conflict_sql = ", ".join([f'"{c}"' for c in key_cols])

    sql = f"""
        INSERT INTO {table} ({col_sql})
        VALUES %s
        ON CONFLICT ({conflict_sql}) DO NOTHING;
    """

    with get_psycopg2_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, sql, values, page_size=5000)
        conn.commit()


def normalize_for_db(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalisation DB:
    - timestamp: datetime tz-aware
    - symbol, kline_interval présents
    """
    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # garantir types simples pour psycopg2
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], utc=True)
        elif df[c].dtype.name == "Int64":
            df[c] = df[c].astype("float").astype("Int64")  # stabilise parfois les NA
    return df


# -------------------------
# Main loading
# -------------------------
def list_parquet_sources() -> List[Path]:
    """
    Liste les Parquet à charger depuis processed/ et features/
    """
    files = []
    files.extend(sorted(PROCESSED_DIR.glob("*.parquet")))
    files.extend(sorted(FEATURES_DIR.glob("*.parquet")))
    return files


def main() -> None:
    create_tracking_table()
    create_tables_if_needed()

    files = list_parquet_sources()
    if not files:
        logger.info("No parquet files found in processed/ or features/")
        return

    logger.info(f"Found {len(files)} parquet file(s) to consider.")

    for fp in files:
        table = route_file_to_table(fp)
        if table is None:
            logger.info(f"⏭️  Skip (unknown type): {fp.name}")
            continue

        if already_loaded(fp.name):
            logger.info(f"⏭️  Skip already loaded: {fp.name}")
            continue

        logger.info(f"➡️  Loading {fp.name} -> {table}")

        try:
            df = pd.read_parquet(fp)
            df = normalize_for_db(df)

            # exigences minimales
            required = {"symbol", "kline_interval", "timestamp"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"{fp.name}: missing required columns: {missing}")

            # Créer/adapter colonnes dynamiques pour features/ml (tables larges)
            if table in {"features_1h", "ml_dataset_1h"}:
                ensure_columns(table, df, base_cols=["symbol", "kline_interval", "timestamp"])

            # Pour klines_clean, on ne veut pas de colonnes non prévues
            if table == "klines_clean":
                allowed = {
                    "symbol", "kline_interval", "timestamp",
                    "open_time", "open", "high", "low", "close", "volume", "number_of_trades"
                }
                extra = set(df.columns) - allowed
                if extra:
                    # on drop les extras si jamais
                    df = df[[c for c in df.columns if c in allowed]]

            # Upsert (idempotent)
            insert_upsert(
                table=table,
                df=df,
                key_cols=["symbol", "kline_interval", "timestamp"],
            )

            mark_as_loaded(fp.name, table)
            logger.info(f"✅ Loaded {fp.name} into {table} (rows={len(df)})")

        except Exception as e:
            logger.exception(f"❌ Failed loading {fp.name}: {e}")

    logger.info("✅ Done.")


if __name__ == "__main__":
    main()
