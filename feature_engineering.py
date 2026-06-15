import pandas as pd
import numpy as np


def load_data(path="paysim dataset.csv"):
    df = pd.read_csv(path)
    return df


def engineer_features(df):
    df = df.copy()

    # --- one-hot encode transaction type ---
    df = pd.get_dummies(df, columns=["type"], prefix="type")

    # --- balance consistency flags ---
    df["balance_mismatch_orig"] = (
        df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    ).abs() > 1e-2

    df["balance_mismatch_dest"] = (
        df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    ).abs() > 1e-2

    df["is_balance_drained"] = (df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0)

    # --- sender (nameOrig) behavioral features ---
    sender_stats = df.groupby("nameOrig")["amount"].agg(["mean", "count"])
    sender_stats.columns = ["sender_avg_amount", "sender_txn_count"]
    df = df.merge(sender_stats, on="nameOrig", how="left")

    df["amount_to_sender_avg_ratio"] = df["amount"] / df["sender_avg_amount"].replace(0, np.nan)
    df["amount_to_sender_avg_ratio"] = df["amount_to_sender_avg_ratio"].fillna(1.0)

    # --- destination (nameDest) graph / mule-account features ---
    dest_stats = df.groupby("nameDest").agg(
        dest_in_degree=("amount", "count"),
        dest_total_inflow=("amount", "sum"),
    )
    df = df.merge(dest_stats, on="nameDest", how="left")

    # destination immediately cashed-out: dest received money then its balance went to 0
    # (proxy using whether dest also appears as an origin with balance drained)
    drained_origins = set(df.loc[df["is_balance_drained"], "nameOrig"])
    df["dest_is_drained_account"] = df["nameDest"].isin(drained_origins)

    # --- bool to int ---
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


def get_feature_columns(df):
    exclude = {"nameOrig", "nameDest", "isFraud", "isFlaggedFraud"}
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df):,} rows")

    df = engineer_features(df)
    print(f"Engineered features, shape: {df.shape}")
    print("Fraud rate:", df["isFraud"].mean())

    feature_cols = get_feature_columns(df)
    print(f"Feature columns ({len(feature_cols)}):", feature_cols)

    df.to_parquet("paysim_features.parquet", index=False)
    print("Saved paysim_features.parquet")
