import pandas as pd
import numpy as np
# 1. LOAD DATA

print("=" * 60)
print("RISKSHIELD DATA QUALITY AUDIT")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(
    "data/train_transaction.csv",
    nrows=100_000
)

print("Dataset loaded:", df.shape)

print(
    "Memory usage:",
    round(df.memory_usage(deep=True).sum() / 1024**2, 2),
    "MB"
)

# 2. BASIC INFORMATION

print("\n" + "=" * 60)
print("1. BASIC DATASET INFORMATION")
print("=" * 60)

print("\nRows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nData types:")

print(
    df.dtypes.value_counts()
)

# 3. TARGET DISTRIBUTION

print("\n" + "=" * 60)
print("2. TARGET DISTRIBUTION")
print("=" * 60)

if "isFraud" in df.columns:

    fraud_counts = df["isFraud"].value_counts()

    print("\nClass counts:")
    print(fraud_counts)

    fraud_percentage = (
        df["isFraud"]
        .mean()
        * 100
    )

    print(
        "\nFraud percentage:",
        round(fraud_percentage, 4),
        "%"
    )

    print(
        "Legitimate percentage:",
        round(100 - fraud_percentage, 4),
        "%"
    )

# 4. MISSING VALUES

print("\n" + "=" * 60)
print("3. MISSING VALUE ANALYSIS")
print("=" * 60)

missing_count = df.isna().sum()

missing_percentage = (
    missing_count
    / len(df)
    * 100
)

missing_df = pd.DataFrame({
    "column": df.columns,
    "missing_count": missing_count.values,
    "missing_percentage": missing_percentage.values
})

missing_df = (
    missing_df
    .sort_values(
        "missing_percentage",
        ascending=False
    )
)

print(
    "\nTop 30 columns by missing percentage:"
)

print(
    missing_df.head(30).to_string(
        index=False
    )
)

# 5. DUPLICATE ROWS

print("\n" + "=" * 60)
print("4. DUPLICATE ANALYSIS")
print("=" * 60)

duplicate_rows = df.duplicated().sum()

print(
    "\nCompletely duplicated rows:",
    duplicate_rows
)

print(
    "Duplicate percentage:",
    round(
        duplicate_rows / len(df) * 100,
        4
    ),
    "%"
)

# 6. UNIQUE VALUES

print("\n" + "=" * 60)
print("5. UNIQUE VALUE ANALYSIS")
print("=" * 60)

unique_counts = df.nunique(
    dropna=False
)

unique_df = pd.DataFrame({
    "column": df.columns,
    "unique_values": unique_counts.values
})

unique_df = (
    unique_df
    .sort_values(
        "unique_values"
    )
)

print("\nColumns with fewest unique values:")

print(
    unique_df.head(30).to_string(
        index=False
    )
)


print("\nColumns with most unique values:")

print(
    unique_df.tail(20).sort_values(
        "unique_values",
        ascending=False
    ).to_string(
        index=False
    )
)

# 7. CONSTANT FEATURES

print("\n" + "=" * 60)
print("6. CONSTANT / NEAR-CONSTANT FEATURES")
print("=" * 60)

constant_features = []

near_constant_features = []

for column in df.columns:

    counts = (
        df[column]
        .value_counts(
            dropna=False,
            normalize=True
        )
    )

    if len(counts) == 1:

        constant_features.append(
            column
        )

    elif counts.iloc[0] >= 0.99:

        near_constant_features.append(
            (
                column,
                counts.iloc[0] * 100
            )
        )


print(
    "\nConstant features:"
)

print(
    constant_features
)

print(
    "\nNear-constant features (>99% same value):"
)

for column, percentage in near_constant_features:

    print(
        f"{column}: "
        f"{percentage:.2f}%"
    )

# 8. NUMERICAL SUMMARY

print("\n" + "=" * 60)
print("7. NUMERICAL FEATURE SUMMARY")
print("=" * 60)

numeric_columns = (
    df
    .select_dtypes(
        include=np.number
    )
    .columns
)

print(
    "\nNumber of numerical columns:",
    len(numeric_columns)
)

numeric_summary = (
    df[numeric_columns]
    .describe()
    .T
)

print(
    "\nSelected numerical statistics:"
)

print(
    numeric_summary[
        [
            "min",
            "mean",
            "50%",
            "max"
        ]
    ]
    .head(30)
    .to_string()
)

# 9. TRANSACTION AMOUNT ANALYSIS

print("\n" + "=" * 60)
print("8. TRANSACTION AMOUNT ANALYSIS")
print("=" * 60)

if "TransactionAmt" in df.columns:

    amount = df["TransactionAmt"]

    print(
        "\nMinimum:",
        amount.min()
    )

    print(
        "Maximum:",
        amount.max()
    )

    print(
        "Mean:",
        amount.mean()
    )

    print(
        "Median:",
        amount.median()
    )

    print(
        "Zero amounts:",
        (amount == 0).sum()
    )

    print(
        "Negative amounts:",
        (amount < 0).sum()
    )

    print(
        "\nAmount percentiles:"
    )

    print(
        amount.quantile(
            [
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
                0.999
            ]
        )
    )

# 10. CATEGORICAL CARDINALITY

print("\n" + "=" * 60)
print("9. CATEGORICAL FEATURE CARDINALITY")
print("=" * 60)

categorical_columns = (
    df
    .select_dtypes(
        include=["object"]
    )
    .columns
)

print(
    "\nNumber of categorical columns:",
    len(categorical_columns)
)

for column in categorical_columns:

    print(
        f"{column:25} "
        f"unique={df[column].nunique(dropna=False):6}"
    )

# 11. POTENTIAL ID-LIKE FEATURES

print("\n" + "=" * 60)
print("10. POTENTIAL ID-LIKE FEATURES")
print("=" * 60)

print(
    "\nFeatures with extremely high cardinality:"
)

for column in df.columns:

    unique_ratio = (
        df[column]
        .nunique(dropna=False)
        / len(df)
    )

    if unique_ratio > 0.90:

        print(
            f"{column:25} "
            f"unique ratio={unique_ratio:.3f}"
        )

# 12. TARGET CORRELATION

print("\n" + "=" * 60)
print("11. NUMERICAL FEATURE CORRELATION WITH FRAUD")
print("=" * 60)

if "isFraud" in df.columns:

    correlations = (
        df[numeric_columns]
        .corrwith(df["isFraud"])
        .abs()
        .sort_values(
            ascending=False
        )
    )

    print(
        "\nTop numerical correlations:"
    )

    print(
        correlations
        .head(20)
        .to_string()
    )

# 13. FRAUD RATE BY IMPORTANT CATEGORICAL FEATURES

print("\n" + "=" * 60)
print("12. FRAUD RATE BY CATEGORICAL FEATURES")
print("=" * 60)

categorical_to_check = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain"
]

for column in categorical_to_check:

    if column not in df.columns:
        continue

    print(
        f"\n--- {column} ---"
    )

    fraud_rate = (
        df
        .groupby(column)["isFraud"]
        .agg(
            [
                "count",
                "mean"
            ]
        )
        .sort_values(
            "mean",
            ascending=False
        )
    )

    fraud_rate["fraud_percentage"] = (
        fraud_rate["mean"] * 100
    )

    print(
        fraud_rate
        .head(15)
        .to_string()
    )

# 14. DATA QUALITY WARNINGS

print("\n" + "=" * 60)
print("13. DATA QUALITY WARNINGS")
print("=" * 60)

warnings = 0


# Missingness warning

high_missing = missing_df[
    missing_df["missing_percentage"] > 90
]

if len(high_missing) > 0:

    warnings += 1

    print(
        f"\n⚠ {len(high_missing)} "
        "features have >90% missing values."
    )


# Duplicate warning

if duplicate_rows > 0:

    warnings += 1

    print(
        f"\n⚠ {duplicate_rows} "
        "completely duplicated rows found."
    )


# Negative amount warning

if "TransactionAmt" in df.columns:

    negative_amounts = (
        df["TransactionAmt"] < 0
    ).sum()

    if negative_amounts > 0:

        warnings += 1

        print(
            f"\n⚠ {negative_amounts} "
            "negative transaction amounts found."
        )


# Zero amount warning

if "TransactionAmt" in df.columns:

    zero_amounts = (
        df["TransactionAmt"] == 0
    ).sum()

    if zero_amounts > 0:

        warnings += 1

        print(
            f"\n⚠ {zero_amounts} "
            "zero transaction amounts found."
        )


# Constant feature warning

if len(constant_features) > 0:

    warnings += 1

    print(
        f"\n⚠ {len(constant_features)} "
        "constant features found."
    )


# Final status

print("\n" + "=" * 60)

if warnings == 0:

    print(
        "DATA QUALITY STATUS: CLEAN"
    )

else:

    print(
        f"DATA QUALITY STATUS: "
        f"{warnings} areas require investigation"
    )

print("=" * 60)