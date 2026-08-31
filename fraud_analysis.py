import pandas as pd

df = pd.read_csv(
    "data/train_transaction.csv",
    nrows=100_000
)

print("\n========== TRANSACTION AMOUNT ==========")

print(
    df.groupby("isFraud")["TransactionAmt"]
      .agg(["count", "mean", "median", "std", "min", "max"])
)

print("\n========== PRODUCT × FRAUD ==========")

print(
    pd.crosstab(
        df["ProductCD"],
        df["isFraud"],
        normalize="index"
    ) * 100
)

print("\n========== CARD TYPE × FRAUD ==========")

print(
    pd.crosstab(
        df["card4"],
        df["isFraud"],
        normalize="index"
    ) * 100
)

print("\n========== CARD TYPE 6 × FRAUD ==========")

print(
    pd.crosstab(
        df["card6"],
        df["isFraud"],
        normalize="index"
    ) * 100
)

print("\n========== EMAIL MISSINGNESS ==========")

df["P_email_missing"] = df["P_emaildomain"].isna()
df["R_email_missing"] = df["R_emaildomain"].isna()

print(
    df.groupby("isFraud")[
        ["P_email_missing", "R_email_missing"]
    ].mean() * 100
)

print("\n========== DISTANCE MISSINGNESS ==========")

df["dist1_missing"] = df["dist1"].isna()
df["dist2_missing"] = df["dist2"].isna()

print(
    df.groupby("isFraud")[
        ["dist1_missing", "dist2_missing"]
    ].mean() * 100
)

print("\n========== OVERALL MISSINGNESS ==========")

feature_cols = [
    c for c in df.columns
    if c not in ["isFraud"]
]

df["missing_count"] = df[feature_cols].isna().sum(axis=1)

print(
    df.groupby("isFraud")["missing_count"]
      .agg(["mean", "median", "min", "max"])
)