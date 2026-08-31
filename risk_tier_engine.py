import pandas as pd
import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.calibration import calibration_curve

# 1. LOAD DATA

print("Loading dataset...")

df = pd.read_csv(
    "data/train_transaction.csv",
    nrows=100_000
)

print("Dataset loaded:", df.shape)

# 2. SORT CHRONOLOGICALLY
print("\nSorting transactions chronologically...")

df = df.sort_values(
    "TransactionDT"
).reset_index(drop=True)

print("Transactions sorted.")

# 3. BASIC FEATURES

print("\nCreating features...")


# Log transaction amount
df["log_amount"] = np.log1p(
    df["TransactionAmt"]
)


# Missingness indicators
df["P_email_missing"] = (
    df["P_emaildomain"].isna().astype(int)
)

df["R_email_missing"] = (
    df["R_emaildomain"].isna().astype(int)
)

df["dist1_missing"] = (
    df["dist1"].isna().astype(int)
)

df["dist2_missing"] = (
    df["dist2"].isna().astype(int)
)


# Total missing values
df["missing_count"] = (
    df.isna().sum(axis=1)
)


# Time features
df["transaction_day"] = (
    df["TransactionDT"] // (24 * 60 * 60)
)

df["transaction_time"] = (
    df["TransactionDT"] % (24 * 60 * 60)
)


# Day of week
df["day_of_week"] = (
    df["transaction_day"] % 7
)

# 4. REMOVE HIGH-MISSING FEATURES

high_missing_features = [
    "dist2",
    "D7",
    "D12",
    "D13"
]

for col in high_missing_features:

    if col in df.columns:
        df.drop(
            columns=col,
            inplace=True
        )

# 5. TEMPORAL TRAIN / TEST SPLIT

print("\nCreating temporal split...")

split_index = int(
    len(df) * 0.80
)

train_df = df.iloc[
    :split_index
].copy()

test_df = df.iloc[
    split_index:
].copy()

print(
    "Training transactions:",
    len(train_df)
)

print(
    "Testing transactions:",
    len(test_df)
)

print(
    "\nTraining time:",
    train_df["TransactionDT"].min(),
    "→",
    train_df["TransactionDT"].max()
)

print(
    "Testing time:",
    test_df["TransactionDT"].min(),
    "→",
    test_df["TransactionDT"].max()
)

# 6. TARGET

y_train = train_df["isFraud"].copy()

y_test = test_df["isFraud"].copy()

# 7. LEAKAGE-SAFE BEHAVIORAL FEATURES

print(
    "\nCreating leakage-safe behavioral features..."
)


# ------------------------------------------------------------
# CARD FREQUENCY
# ------------------------------------------------------------

card_frequency = (
    train_df["card1"]
    .value_counts()
)

train_df["card_frequency"] = (
    train_df["card1"]
    .map(card_frequency)
    .fillna(0)
)

test_df["card_frequency"] = (
    test_df["card1"]
    .map(card_frequency)
    .fillna(0)
)


# ------------------------------------------------------------
# EMAIL FREQUENCY
# ------------------------------------------------------------

email_frequency = (
    train_df["P_emaildomain"]
    .value_counts()
)

train_df["email_frequency"] = (
    train_df["P_emaildomain"]
    .map(email_frequency)
    .fillna(0)
)

test_df["email_frequency"] = (
    test_df["P_emaildomain"]
    .map(email_frequency)
    .fillna(0)
)


# ------------------------------------------------------------
# ADDRESS FREQUENCY
# ------------------------------------------------------------

address_frequency = (
    train_df["addr1"]
    .value_counts()
)

train_df["address_frequency"] = (
    train_df["addr1"]
    .map(address_frequency)
    .fillna(0)
)

test_df["address_frequency"] = (
    test_df["addr1"]
    .map(address_frequency)
    .fillna(0)
)


# ------------------------------------------------------------
# CARD AVERAGE TRANSACTION AMOUNT
# ------------------------------------------------------------

card_amount_mean = (
    train_df
    .groupby("card1")["TransactionAmt"]
    .mean()
)

train_df["amount_vs_card_mean"] = (
    train_df["TransactionAmt"]
    /
    train_df["card1"]
    .map(card_amount_mean)
    .replace(0, np.nan)
)

test_df["amount_vs_card_mean"] = (
    test_df["TransactionAmt"]
    /
    test_df["card1"]
    .map(card_amount_mean)
    .replace(0, np.nan)
)


# ------------------------------------------------------------
# PRODUCT AVERAGE TRANSACTION AMOUNT
# ------------------------------------------------------------

product_amount_mean = (
    train_df
    .groupby("ProductCD")["TransactionAmt"]
    .mean()
)

train_df["amount_vs_product_mean"] = (
    train_df["TransactionAmt"]
    /
    train_df["ProductCD"]
    .map(product_amount_mean)
    .replace(0, np.nan)
)

test_df["amount_vs_product_mean"] = (
    test_df["TransactionAmt"]
    /
    test_df["ProductCD"]
    .map(product_amount_mean)
    .replace(0, np.nan)
)


print(
    "Leakage-safe behavioral features created!"
)

# 8. FEATURE SELECTION

features = [

    # Transaction information
    "TransactionAmt",
    "ProductCD",

    # Card information
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",

    # Address
    "addr1",
    "addr2",

    # Email
    "P_emaildomain",
    "R_emaildomain",

    # Basic engineered features
    "log_amount",
    "P_email_missing",
    "R_email_missing",
    "dist1_missing",
    "dist2_missing",
    "missing_count",

    # Time
    "transaction_day",
    "transaction_time",
    "day_of_week",

    # Behavioral
    "card_frequency",
    "email_frequency",
    "address_frequency",
    "amount_vs_card_mean",
    "amount_vs_product_mean"
]


# Keep only features that actually exist
features = [
    col
    for col in features
    if col in train_df.columns
]


X_train = train_df[features].copy()

X_test = test_df[features].copy()

# 9. FEATURE TYPES

categorical_features = [
    col
    for col in [
        "ProductCD",
        "card4",
        "card6",
        "P_emaildomain",
        "R_emaildomain"
    ]
    if col in features
]


numerical_features = [
    col
    for col in features
    if col not in categorical_features
]

# 10. PREPROCESSING

print("\nEncoding categorical features...")


numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "encoder",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[

        (
            "num",
            numeric_pipeline,
            numerical_features
        ),

        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ]
)

# 11. MODEL

model = HistGradientBoostingClassifier(

    max_iter=200,

    learning_rate=0.08,

    max_leaf_nodes=31,

    min_samples_leaf=50,

    l2_regularization=1.0,

    random_state=42
)

# 12. PIPELINE

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            model
        )
    ]
)

# 13. TRAIN

print("\nTraining RiskShield...")

pipeline.fit(
    X_train,
    y_train
)

print("Training complete.")

# 14. GENERATE RISK SCORES

print(
    "\nGenerating risk scores..."
)


# IMPORTANT:
# This is the probability that the transaction is fraudulent.

risk_scores = pipeline.predict_proba(
    X_test
)[:, 1]

# 15. MODEL PERFORMANCE

roc_auc = roc_auc_score(
    y_test,
    risk_scores
)

pr_auc = average_precision_score(
    y_test,
    risk_scores
)


print("\n======================================")
print("MODEL PERFORMANCE")
print("======================================")

print(
    f"ROC-AUC: {roc_auc:.4f}"
)

print(
    f"PR-AUC: {pr_auc:.4f}"
)

# 16. RISK TIERS

risk_tier = np.select(

    [
        risk_scores < 0.10,

        (
            (risk_scores >= 0.10)
            &
            (risk_scores < 0.50)
        ),

        risk_scores >= 0.50
    ],

    [
        "LOW",
        "MEDIUM",
        "HIGH"
    ],

    default="LOW"
)

# 17. ACTION

action = np.select(

    [
        risk_scores < 0.10,

        (
            (risk_scores >= 0.10)
            &
            (risk_scores < 0.50)
        ),

        risk_scores >= 0.50
    ],

    [
        "APPROVE",
        "REVIEW",
        "BLOCK"
    ],

    default="APPROVE"
)

# 18. RESULTS DATAFRAME

results = pd.DataFrame({

    "risk_score": risk_scores,

    "risk_tier": risk_tier,

    "action": action,

    "actual_fraud": y_test.values
})

# 19. RISK TIER DISTRIBUTION

print("\n======================================")
print("RISK TIER DISTRIBUTION")
print("======================================")

print(
    results["risk_tier"]
    .value_counts()
)


print("\nPercentage of transactions:")

print(
    (
        results["risk_tier"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)

# 20. FRAUD RATE BY RISK TIER

print("\n======================================")
print("FRAUD RATE BY RISK TIER")
print("======================================")


tier_analysis = (
    results
    .groupby("risk_tier")
    .agg(
        transactions=(
            "actual_fraud",
            "count"
        ),

        fraud_cases=(
            "actual_fraud",
            "sum"
        )
    )
)


tier_analysis["fraud_rate"] = (

    tier_analysis["fraud_cases"]

    /

    tier_analysis["transactions"]

    * 100
)


print(
    tier_analysis
)

# 21. ACTION DISTRIBUTION

print("\n======================================")
print("ACTION DISTRIBUTION")
print("======================================")

print(
    results["action"]
    .value_counts()
)

# 22. DECISION ENGINE PERFORMANCE

decision_pred = (
    risk_scores >= 0.10
).astype(int)


true_positive = (
    (decision_pred == 1)
    &
    (y_test.values == 1)
).sum()


false_positive = (
    (decision_pred == 1)
    &
    (y_test.values == 0)
).sum()


true_negative = (
    (decision_pred == 0)
    &
    (y_test.values == 0)
).sum()


false_negative = (
    (decision_pred == 0)
    &
    (y_test.values == 1)
).sum()


precision = precision_score(
    y_test,
    decision_pred,
    zero_division=0
)


recall = recall_score(
    y_test,
    decision_pred,
    zero_division=0
)


f1 = f1_score(
    y_test,
    decision_pred,
    zero_division=0
)


print("\n======================================")
print("DECISION ENGINE PERFORMANCE")
print("======================================")

print(
    "True Positives:",
    true_positive
)

print(
    "False Positives:",
    false_positive
)

print(
    "True Negatives:",
    true_negative
)

print(
    "False Negatives:",
    false_negative
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall: {recall:.4f}"
)

print(
    f"F1: {f1:.4f}"
)

# 23. HIGH-RISK ANALYSIS

high_risk = results[
    results["risk_tier"] == "HIGH"
]


print("\n======================================")
print("HIGH-RISK ANALYSIS")
print("======================================")


print(
    "High-risk transactions:",
    len(high_risk)
)


print(
    "Fraud in high-risk:",
    high_risk["actual_fraud"].sum()
)


if len(high_risk) > 0:

    high_risk_rate = (
        high_risk["actual_fraud"].mean()
        * 100
    )

else:

    high_risk_rate = 0


print(
    f"High-risk fraud rate: "
    f"{high_risk_rate:.2f} %"
)

# 24. RISK SCORE DISTRIBUTION

print("\n======================================")
print("RISK SCORE DISTRIBUTION")
print("======================================")


print(
    results["risk_score"]
    .describe()
)
# 25. SAMPLE RISK DECISIONS

print("\n======================================")
print("SAMPLE RISK DECISIONS")
print("======================================")


print(
    results
    .sort_values(
        "risk_score",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)

# 26. CALIBRATION ANALYSIS

print("\n======================================")
print("CALIBRATION ANALYSIS")
print("======================================")


prob_true, prob_pred = calibration_curve(

    y_test,

    risk_scores,

    n_bins=10,

    strategy="quantile"
)


calibration_df = pd.DataFrame({

    "Predicted_Risk": prob_pred,

    "Actual_Fraud_Rate": prob_true
})


calibration_df["Absolute_Error"] = (

    calibration_df["Predicted_Risk"]

    -

    calibration_df["Actual_Fraud_Rate"]

).abs()


print("\nCalibration table:")

print(
    calibration_df.to_string(
        index=False
    )
)


mean_calibration_error = (

    calibration_df["Absolute_Error"]
    .mean()
)


print("\n======================================")
print("CALIBRATION ERROR")
print("======================================")


print(
    f"Mean Calibration Error: "
    f"{mean_calibration_error:.4f}"
)

# 27. RISKSHIELD DECISION POLICY

print("\n======================================")
print("RISKSHIELD DECISION POLICY")
print("======================================")

print(
    "\nLOW RISK"
)

print(
    "Score < 0.10"
)

print(
    "→ APPROVE"
)


print(
    "\nMEDIUM RISK"
)

print(
    "0.10 <= Score < 0.50"
)

print(
    "→ REVIEW"
)


print(
    "\nHIGH RISK"
)

print(
    "Score >= 0.50"
)

print(
    "→ BLOCK"
)

# 28. SAVE RESULTS

results.to_csv(
    "risk_scores.csv",
    index=False
)


print(
    "\nRisk scores saved to "
    "'risk_scores.csv'"
)


print(
    "\nRisk tier engine complete."
)