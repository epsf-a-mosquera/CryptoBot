# src/ml/train.py
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COL = "label"
TIME_COL = "timestamp"

DEFAULT_TEST_RATIO = 0.2
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")


@dataclass
class TrainMeta:
    symbol: str
    interval: str
    input_path: str
    rows_total: int
    rows_train: int
    rows_test: int
    test_ratio: float
    train_end_timestamp: str
    test_start_timestamp: str
    target_col: str
    time_col: str
    feature_count: int
    features: List[str]
    label_mapping: Dict
    trained_at_utc: str
    f1_macro_test: float


def _infer_symbol_interval_from_filename(path: Path) -> Tuple[str, str]:
    # ex: btcusdt_1h_ml.parquet
    parts = path.stem.lower().split("_")
    if len(parts) >= 3:
        return parts[0].upper(), parts[1]
    return "UNKNOWN", "UNKNOWN"


def _normalize_labels(y: pd.Series) -> Tuple[np.ndarray, Dict, Dict]:
    """
    Supporte:
    - str labels: BUY/SELL/HOLD
    - int labels: -1/0/1 ou 0/1/2
    Retourne y_enc + mapping + inverse_mapping
    """
    if pd.api.types.is_numeric_dtype(y):
        uniq = sorted(pd.unique(y.dropna()))
        mapping = {v: int(i) for i, v in enumerate(uniq)}
        inv = {int(i): v for v, i in mapping.items()}
        y_enc = y.map(mapping).astype(int).to_numpy()
        return y_enc, mapping, inv

    y_str = y.astype(str).str.upper()
    canonical = {"SELL": 0, "HOLD": 1, "BUY": 2}  # ordre fixe
    extras = sorted(set(y_str.unique()) - set(canonical.keys()))

    mapping = dict(canonical)
    start = max(mapping.values()) + 1
    for i, v in enumerate(extras):
        mapping[v] = start + i

    inv = {v: k for k, v in mapping.items()}
    y_enc = y_str.map(mapping).astype(int).to_numpy()
    return y_enc, mapping, inv


def _select_features(df: pd.DataFrame) -> List[str]:
    """
    Sélectionne des features NUMÉRIQUES, en excluant:
    - colonnes d'identifiants / meta
    - colonnes de label/target
    - colonnes futures (anti data leakage)
    IMPORTANT: doit rester compatible avec le fichier features.parquet (inférence).
    """
    excluded_exact = {TARGET_COL, TIME_COL, "open_time", "symbol", "kline_interval"}

    feats: List[str] = []
    for c in df.columns:
        if c in excluded_exact:
            continue

        lc = c.lower()

        # Anti-target / anti-leakage
        if lc.startswith("target_"):
            continue
        if lc.startswith("label_"):
            continue
        if "future" in lc:
            continue

        # Garde uniquement numériques
        if pd.api.types.is_numeric_dtype(df[c]):
            feats.append(c)

    return feats


def _time_split(df: pd.DataFrame, test_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    n = len(df)
    split_idx = int((1.0 - test_ratio) * n)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True, help="Parquet ML (ex: data/processed/btcusdt_1h_ml.parquet)")
    p.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    p.add_argument("--model-out", type=str, default="", help="Chemin de sortie .joblib (optionnel)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    symbol, interval = _infer_symbol_interval_from_filename(input_path)

    df = pd.read_parquet(input_path)

    # checks
    if TIME_COL not in df.columns:
        raise ValueError(f"Missing '{TIME_COL}' column in {input_path.name}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing '{TARGET_COL}' column in {input_path.name}")

    # clean basic
    df = df.dropna(subset=[TIME_COL, TARGET_COL]).copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True, errors="coerce")
    df = df.dropna(subset=[TIME_COL]).copy()
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    features = _select_features(df)
    if not features:
        raise ValueError("No numeric features found after exclusions. Check your dataset columns.")

    # split temporel
    train_df, test_df = _time_split(df, args.test_ratio)

    # X / y
    X_train = train_df[features].astype(float).to_numpy()
    X_test = test_df[features].astype(float).to_numpy()

    y_train, mapping, inv = _normalize_labels(train_df[TARGET_COL])

    # Pour y_test, on réutilise mapping (cohérent train/test)
    y_test_raw = test_df[TARGET_COL]
    if pd.api.types.is_numeric_dtype(y_test_raw):
        y_test = y_test_raw.map(mapping).astype(int).to_numpy()
    else:
        y_test = y_test_raw.astype(str).str.upper().map(mapping).astype(int).to_numpy()

    # pipeline
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    f1m = float(f1_score(y_test, y_pred, average="macro"))

    trained_at = datetime.now(timezone.utc).isoformat()

    # save model bundle
    model_out = Path(args.model_out) if args.model_out else (MODELS_DIR / f"{symbol.lower()}_{interval}_model.joblib")
    bundle = {
        "pipeline": pipeline,
        "features": features,
        "label_mapping": mapping,
        "label_inverse_mapping": inv,
        "target_col": TARGET_COL,
        "time_col": TIME_COL,
        "symbol": symbol,
        "interval": interval,
        "trained_at_utc": trained_at,
    }
    joblib.dump(bundle, model_out)

    meta = TrainMeta(
        symbol=symbol,
        interval=interval,
        input_path=str(input_path),
        rows_total=len(df),
        rows_train=len(train_df),
        rows_test=len(test_df),
        test_ratio=float(args.test_ratio),
        train_end_timestamp=str(train_df[TIME_COL].max()),
        test_start_timestamp=str(test_df[TIME_COL].min()),
        target_col=TARGET_COL,
        time_col=TIME_COL,
        feature_count=len(features),
        features=features,
        label_mapping=mapping,
        trained_at_utc=trained_at,
        f1_macro_test=f1m,
    )

    meta_path = REPORTS_DIR / f"{symbol.lower()}_{interval}_train_meta.json"
    meta_path.write_text(json.dumps(asdict(meta), indent=2))

    print(f"[OK] Model saved: {model_out}")
    print(f"[OK] Train meta:  {meta_path}")
    print(f"[OK] Test F1-macro: {f1m:.4f}")
    print(f"[INFO] Feature count: {len(features)}")
    print(f"[INFO] First features: {features[:10]}")


if __name__ == "__main__":
    main()
