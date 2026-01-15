# src/ml/evaluate.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

REPORTS_DIR = Path("reports")


def _time_split(df: pd.DataFrame, time_col: str, test_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)
    split_idx = int((1.0 - test_ratio) * n)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="Chemin du modèle .joblib (models/...)")
    p.add_argument("--input", type=str, required=True, help="Parquet ML (data/processed/..._ml.parquet)")
    p.add_argument("--test-ratio", type=float, default=0.2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(args.model)
    pipeline = bundle["pipeline"]
    features = bundle["features"]
    mapping = bundle["label_mapping"]
    inv = bundle["label_inverse_mapping"]
    target_col = bundle["target_col"]
    time_col = bundle["time_col"]
    symbol = bundle.get("symbol", "unknown")
    interval = bundle.get("interval", "unknown")

    df = pd.read_parquet(args.input)
    if time_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"Input parquet must contain '{time_col}' and '{target_col}'")

    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col, target_col]).copy()

    # split temporel (même règle que train)
    _, test_df = _time_split(df, time_col, args.test_ratio)

    # checks features
    missing_feats = [f for f in features if f not in test_df.columns]
    if missing_feats:
        raise KeyError(
            f"Model expects {len(features)} features but input is missing {len(missing_feats)}. "
            f"Example missing: {missing_feats[:10]}"
        )

    X_test = test_df[features].astype(float).to_numpy()

    # encode y_test via mapping du modèle
    y_raw = test_df[target_col]
    if pd.api.types.is_numeric_dtype(y_raw):
        y_test = y_raw.map(mapping).astype(int).to_numpy()
    else:
        y_test = y_raw.astype(str).str.upper().map(mapping).astype(int).to_numpy()

    y_pred = pipeline.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    f1m = float(f1_score(y_test, y_pred, average="macro"))

    labels_sorted = sorted(inv.keys())
    target_names = [str(inv[i]) for i in labels_sorted]

    report = classification_report(
        y_test, y_pred,
        labels=labels_sorted,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    out = {
        "symbol": symbol,
        "interval": interval,
        "model_path": args.model,
        "input_path": args.input,
        "test_ratio": float(args.test_ratio),
        "rows_test": int(len(test_df)),
        "accuracy": acc,
        "f1_macro": f1m,
        "labels_order": target_names,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "test_start_timestamp": str(test_df[time_col].min()),
        "test_end_timestamp": str(test_df[time_col].max()),
    }

    out_path = REPORTS_DIR / f"{str(symbol).lower()}_{interval}_evaluation.json"
    out_path.write_text(json.dumps(out, indent=2))

    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    cm_csv = REPORTS_DIR / f"{str(symbol).lower()}_{interval}_confusion_matrix.csv"
    cm_df.to_csv(cm_csv)

    print(f"[OK] Evaluation saved: {out_path}")
    print(f"[OK] Confusion matrix CSV: {cm_csv}")
    print(f"[OK] accuracy={acc:.4f} | f1_macro={f1m:.4f}")


if __name__ == "__main__":
    main()
