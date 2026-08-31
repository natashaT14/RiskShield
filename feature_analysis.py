import pandas as pd

PATH = "data/train_transaction.csv"

print("Loading a 100,000-row sample...")

df = pd.read_csv(
    PATH,
    nrows=100_000,
    low_memory=True
)

print("\n========== DATASET ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\n========== TARGET ==========")
print(df["isFraud"].value_counts())
print("\nTarget percentage:")
print(df["isFraud"].value_counts(normalize=True) * 100)

print("\n========== UNIQUE VALUES ==========")

for col in [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain"
]:
    print(f"\n{col}:")
    print("Unique:", df[col].nunique(dropna=True))
    print(df[col].value_counts(dropna=False).head(10))

print("\n========== NUMERIC SUMMARY ==========")

print(
    df[
        [
            "TransactionAmt",
            "TransactionDT",
            "dist1",
            "dist2"
        ]
    ].describe().T
)

print("\n========== FRAUD RATE BY PRODUCT ==========")

print(
    df.groupby("ProductCD")["isFraud"]
      .agg(["count", "mean"])
      .sort_values("mean", ascending=False)
)

print("\n========== FRAUD RATE BY CARD TYPE ==========")

print(
    df.groupby("card4")["isFraud"]
      .agg(["count", "mean"])
      .sort_values("mean", ascending=False)
)

print("\n========== MEMORY USAGE ==========")

memory_mb = df.memory_usage(deep=True).sum() / 1024**2

print(f"Memory used: {memory_mb:.2f} MB")