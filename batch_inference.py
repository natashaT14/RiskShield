import warnings
warnings.filterwarnings("ignore")

import gc
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)



# CONFIGURATION


DATA_PATH = "data/train_transaction.csv"

MODEL_PATH = (
    "models/riskshield_histgradientboosting.pkl"
)

MAPPINGS_PATH = (
    "models/behavioral_mappings.pkl"
)

THRESHOLD_PATH = (
    "models/riskshield_threshold.csv"
)

OUTPUT_PATH = (
    "models/batch_inference_results.csv"
)



# FEATURES


FEATURES = [

    "TransactionAmt",
    "ProductCD",

    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",

    "addr1",
    "addr2",

    "P_emaildomain",
    "R_emaildomain",

    "log_amount",

    "transaction_hour",
    "transaction_day",
    "day_of_week",

    "missing_count",

    "P_email_missing",
    "R_email_missing",

    "dist1_missing",
    "dist2_missing",

    "card_frequency",
    "address_frequency",
    "email_frequency",

    "amount_vs_product_mean",
    "amount_vs_card_mean"
]



# RAW COLUMNS


RAW_COLUMNS = [

    "TransactionDT",
    "TransactionAmt",
    "ProductCD",

    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",

    "addr1",
    "addr2",

    "P_emaildomain",
    "R_emaildomain",

    "dist1",
    "dist2",

    "isFraud"
]



# HEADER


print("=" * 70)
print("RISKSHEILD BATCH INFERENCE")
print("=" * 70)



# 1. LOAD MODEL


print("\nLoading trained RiskShield model...")

model = joblib.load(
    MODEL_PATH
)

print("Model loaded.")



# 2. LOAD BEHAVIORAL MAPPINGS


print("\nLoading behavioral mappings...")

mappings = joblib.load(
    MAPPINGS_PATH
)

card_counts = mappings["card_counts"]
address_counts = mappings["address_counts"]
email_counts = mappings["email_counts"]
product_mean = mappings["product_mean"]
card_mean = mappings["card_mean"]

print("Behavioral mappings loaded.")



# 3. LOAD THRESHOLD


threshold_df = pd.read_csv(
    THRESHOLD_PATH
)

threshold = float(
    threshold_df.iloc[0]["Threshold"]
)

print(
    f"\nOperating threshold: {threshold:.2f}"
)

# 4. LOAD DATA

print("\nLoading test data...")

df = pd.read_csv(
    DATA_PATH,
    usecols=RAW_COLUMNS
)

print(
    f"Dataset loaded: {df.shape}"
)

# 5. TEMPORAL SORT

print("\nSorting transactions chronologically...")

df = (
    df
    .sort_values("TransactionDT")
    .reset_index(drop=True)
)

# 6. CREATE SAME TEMPORAL TEST SPLIT

split_index = int(
    len(df) * 0.80
)

test_df = (
    df.iloc[split_index:]
    .copy()
)

print(
    f"\nHeld-out test transactions: "
    f"{len(test_df):,}"
)

# 7. BASIC FEATURES

print("\nCreating basic features...")


test_df["log_amount"] = np.log1p(
    test_df["TransactionAmt"]
)


test_df["transaction_hour"] = (
    (test_df["TransactionDT"] // 3600) % 24
)


test_df["transaction_day"] = (
    test_df["TransactionDT"]
    // (3600 * 24)
)


test_df["day_of_week"] = (
    test_df["transaction_day"] % 7
)


test_df["missing_count"] = (
    test_df.isnull().sum(axis=1)
)


test_df["P_email_missing"] = (
    test_df["P_emaildomain"]
    .isna()
    .astype(np.int8)
)


test_df["R_email_missing"] = (
    test_df["R_emaildomain"]
    .isna()
    .astype(np.int8)
)


test_df["dist1_missing"] = (
    test_df["dist1"]
    .isna()
    .astype(np.int8)
)


test_df["dist2_missing"] = (
    test_df["dist2"]
    .isna()
    .astype(np.int8)
)

# 8. APPLY TRAINING BEHAVIORAL MAPPINGS

print(
    "\nApplying training behavioral mappings..."
)


test_df["card_frequency"] = (
    test_df["card1"]
    .map(card_counts)
    .fillna(0)
)


test_df["address_frequency"] = (
    test_df["addr1"]
    .map(address_counts)
    .fillna(0)
)


test_df["email_frequency"] = (
    test_df["P_emaildomain"]
    .map(email_counts)
    .fillna(0)
)


test_df["amount_vs_product_mean"] = (

    test_df["TransactionAmt"]
    /
    test_df["ProductCD"].map(product_mean)

)


test_df["amount_vs_card_mean"] = (

    test_df["TransactionAmt"]
    /
    test_df["card1"].map(card_mean)

)


for column in [

    "amount_vs_product_mean",
    "amount_vs_card_mean"

]:

    test_df[column] = (

        test_df[column]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(1)

    )


print(
    "Behavioral mappings applied."
)

# 9. PREPARE FEATURES

X_test = (
    test_df[FEATURES]
    .copy()
)

y_test = (
    test_df["isFraud"]
    .copy()
)

# 10. GENERATE FRAUD PROBABILITIES

print("\n")
print("=" * 70)
print("GENERATING FRAUD PREDICTIONS")
print("=" * 70)

print("\nRunning batch inference...")

fraud_probability = (

    model
    .predict_proba(X_test)[:, 1]

)

print("Batch inference complete.")

# 11. RISK SCORE

risk_score = (
    fraud_probability * 100
)

# 12. RISK LEVEL

risk_level = np.select(

    [

        fraud_probability < 0.10,

        fraud_probability < 0.50

    ],

    [

        "LOW",

        "MEDIUM"

    ],

    default="HIGH"

)

# 13. DECISION

decision = np.select(

    [

        fraud_probability < threshold,

        fraud_probability < 0.50

    ],

    [

        "APPROVE",

        "REVIEW"

    ],

    default="BLOCK"

)

# 14. REASON

reason = np.select(

    [

        fraud_probability < threshold,

        fraud_probability < 0.50

    ],

    [

        "LOW_FRAUD_PROBABILITY",

        "ELEVATED_FRAUD_PROBABILITY"

    ],

    default="HIGH_FRAUD_PROBABILITY"

)

# 15. PREDICTED CLASS

predicted_class = (

    fraud_probability >= threshold
).astype(int)

# 16. BUILD RESULTS

results_df = pd.DataFrame({

    "TransactionDT":
        test_df["TransactionDT"].values,

    "TransactionAmt":
        test_df["TransactionAmt"].values,

    "ActualFraud":
        y_test.values,

    "FraudProbability":
        fraud_probability,

    "RiskScore":
        risk_score,

    "RiskLevel":
        risk_level,

    "Decision":
        decision,

    "Reason":
        reason,

    "PredictedFraud":
        predicted_class

})

# 17. MODEL PERFORMANCE

roc_auc = roc_auc_score(
    y_test,
    fraud_probability
)


pr_auc = average_precision_score(
    y_test,
    fraud_probability
)


precision = precision_score(
    y_test,
    predicted_class,
    zero_division=0
)


recall = recall_score(
    y_test,
    predicted_class,
    zero_division=0
)


f1 = f1_score(
    y_test,
    predicted_class,
    zero_division=0
)


tn, fp, fn, tp = (
    confusion_matrix(
        y_test,
        predicted_class
    ).ravel()
)

# 18. OPERATIONAL METRICS

total = len(
    results_df
)


approved = (
    results_df["Decision"] == "APPROVE"
).sum()


reviewed = (
    results_df["Decision"] == "REVIEW"
).sum()


blocked = (
    results_df["Decision"] == "BLOCK"
).sum()


actual_fraud = (
    y_test == 1
).sum()


actual_legit = (
    y_test == 0
).sum()


fraud_detected = (
    (y_test == 1)
    &
    (predicted_class == 1)
).sum()


fraud_missed = (
    (y_test == 1)
    &
    (predicted_class == 0)
).sum()


false_positive_rate = (

    fp
    /
    actual_legit

)


approval_rate = (
    approved / total
)


review_rate = (
    reviewed / total
)


block_rate = (
    blocked / total
)

# 19. DISPLAY MODEL METRICS

print("\n")
print("=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)


print(
    f"ROC-AUC : {roc_auc:.4f}"
)


print(
    f"PR-AUC  : {pr_auc:.4f}"
)


print(
    f"Precision: {precision:.4f}"
)


print(
    f"Recall   : {recall:.4f}"
)


print(
    f"F1       : {f1:.4f}"
)

# 20. CONFUSION MATRIX

print("\n")
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)


print(
    f"True Positives : {tp:,}"
)


print(
    f"False Positives: {fp:,}"
)


print(
    f"False Negatives: {fn:,}"
)


print(
    f"True Negatives : {tn:,}"
)

# 21. OPERATIONAL PERFORMANCE

print("\n")
print("=" * 70)
print("RISKSHEILD OPERATIONAL PERFORMANCE")
print("=" * 70)


print(
    f"\nTotal transactions : {total:,}"
)


print(
    f"Actual fraud       : {actual_fraud:,}"
)


print(
    f"Actual legitimate  : {actual_legit:,}"
)


print(
    f"\nAPPROVE : {approved:,} "
    f"({approval_rate:.2%})"
)


print(
    f"REVIEW  : {reviewed:,} "
    f"({review_rate:.2%})"
)


print(
    f"BLOCK   : {blocked:,} "
    f"({block_rate:.2%})"
)


print(
    f"\nFraud detected     : "
    f"{fraud_detected:,}"
)


print(
    f"Fraud missed       : "
    f"{fraud_missed:,}"
)


print(
    f"False positive rate: "
    f"{false_positive_rate:.2%}"
)

# 22. DECISION DISTRIBUTION

print("\n")
print("=" * 70)
print("DECISION DISTRIBUTION")
print("=" * 70)


decision_distribution = (
    results_df["Decision"]
    .value_counts()
)


print(
    decision_distribution
)

# 23. SAVE RESULTS

print("\n")
print("=" * 70)
print("SAVING BATCH RESULTS")
print("=" * 70)


results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    f"\nResults saved to:\n"
    f"{OUTPUT_PATH}"
)

# 24. CLEANUP

del df
del test_df
del X_test
del model
del mappings

gc.collect()


print("\n")
print("=" * 70)
print("RISKSHEILD BATCH INFERENCE COMPLETE")
print("=" * 70)