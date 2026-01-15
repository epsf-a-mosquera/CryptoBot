# src/ml/predict.py
from __future__ import annotations

import argparse
import joblib
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="Chemin du modèle .joblib")
    p.add_argument("--input", type=str, required=True, help="Parquet FEATURES (data/features/..._features.parquet)")
    p.add_argument("--timestamp", type=str, default="", help="Optionnel: timestamp ISO exact à prédire")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    bundle = joblib.load(args.model)
    pipeline = bundle["pipeline"]
    features = bundle["features"]
    inv = bundle["label_inverse_mapping"]
    time_col = bundle.get("time_col", "timestamp")

    df = pd.read_parquet(args.input)

    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")

    # choisir la ligne
    if args.timestamp:
        ts = pd.to_datetime(args.timestamp, utc=True)
        row = df.loc[df[time_col] == ts]
        if row.empty:
            raise ValueError(f"No row found for timestamp={args.timestamp}")
        row = row.sort_values(time_col).tail(1)
    else:
        row = df.sort_values(time_col).tail(1) if time_col in df.columns else df.tail(1)

    # check features présentes
    missing = [f for f in features if f not in row.columns]
    if missing:
        raise KeyError(
            f"Missing {len(missing)} feature(s) in input parquet. "
            f"Example missing: {missing[:10]}. "
            f"Your model was trained with features not present in this features file."
        )

    X = row[features].astype(float).to_numpy()
    pred = int(pipeline.predict(X)[0])

    proba = None
    if hasattr(pipeline, "predict_proba"):
        proba_arr = pipeline.predict_proba(X)[0]
        # on mappe seulement les classes connues
        proba = {str(inv[i]): float(proba_arr[i]) for i in range(len(proba_arr)) if i in inv}

    decision = inv.get(pred, str(pred))
    ts_out = row[time_col].iloc[0].isoformat() if time_col in row.columns else None

    print("=== PREDICTION ===")
    print(f"timestamp: {ts_out}")
    print(f"decision:  {decision}")
    if proba:
        print("probabilities:")
        for k, v in sorted(proba.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
