import pandas as pd
import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# 1. LOAD DATA

print("Loading dataset...")

df = pd.read_csv(
    "data/train_transaction.csv",
    nrows=100_000
)

print("Dataset loaded:", df.shape)

# 2. SORT CHRONOLOGICALLY

print("\nSorting transactions chronologically...")

df = df.sort_values("TransactionDT").reset_index(drop=True)

print("Transactions sorted.")

# 3. BASIC FEATURE ENGINEERING

print("\nCreating features...")

df["log_amount"] = np.log1p(df["TransactionAmt"])

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

df["missing_count"] = df.isna().sum(axis=1)

df["transaction_day"] = (
    df["TransactionDT"] // (24 * 60 * 60)
)

df["transaction_time"] = (
    df["TransactionDT"] % (24 * 60 * 60)
)

print("Feature engineering complete.")

# 4. SELECT FEATURES

features = [
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
    "dist1",
    "dist2",
    "P_emaildomain",
    "R_emaildomain",
    "log_amount",
    "P_email_missing",
    "R_email_missing",
    "dist1_missing",
    "dist2_missing",
    "missing_count",
    "transaction_day",
    "transaction_time"
]

# 5. REMOVE HIGH-MISSING FEATURES

print("\nRemoving high-missing features...")

high_missing = [
    "dist2",
]

features = [
    feature
    for feature in features
    if feature not in high_missing
]

print("Final features:", len(features))


X = df[features]
y = df["isFraud"]

# 6. HANDLE CATEGORICAL FEATURES

categorical_features = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain"
]

numerical_features = [
    feature
    for feature in features
    if feature not in categorical_features
]


# Convert categorical variables into category codes

for col in categorical_features:

    X[col] = (
        X[col]
        .astype("category")
        .cat.codes
    )


# Replace unknown/missing category codes

X = X.replace(-1, np.nan)

# 7. TEMPORAL SPLIT

print("\nCreating temporal split...")

X_model = X.iloc[:70_000]
y_model = y.iloc[:70_000]

X_calibration = X.iloc[70_000:80_000]
y_calibration = y.iloc[70_000:80_000]

X_test = X.iloc[80_000:]
y_test = y.iloc[80_000:]


print("\nMODEL TRAINING:", len(X_model))
print("CALIBRATION SET:", len(X_calibration))
print("FINAL TEST:", len(X_test))


print("\nFraud counts:")

print(
    "Model training:",
    y_model.sum()
)

print(
    "Calibration:",
    y_calibration.sum()
)

print(
    "Test:",
    y_test.sum()
)

# 8. TRAIN MODEL

print("\nTraining HistGradientBoosting...")

model = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=42
)

model.fit(
    X_model,
    y_model
)

print("Model training complete.")

# 9. RAW MODEL PROBABILITIES

print("\nGenerating raw probabilities...")

calibration_probabilities = (
    model.predict_proba(X_calibration)[:, 1]
)

test_probabilities_raw = (
    model.predict_proba(X_test)[:, 1]
)

# 10. CALIBRATION USING PLATT SCALING

print("\nCalibrating probabilities...")

from sklearn.linear_model import LogisticRegression

calibration_model = LogisticRegression()

calibration_model.fit(
    calibration_probabilities.reshape(-1, 1),
    y_calibration
)


test_probabilities_calibrated = (
    calibration_model.predict_proba(
        test_probabilities_raw.reshape(-1, 1)
    )[:, 1]
)

print("Calibration complete.")

# 11. RAW MODEL PERFORMANCE

print("\n======================================")
print("RAW MODEL")
print("======================================")

raw_roc = roc_auc_score(
    y_test,
    test_probabilities_raw
)

raw_pr = average_precision_score(
    y_test,
    test_probabilities_raw
)

raw_brier = brier_score_loss(
    y_test,
    test_probabilities_raw
)

print("ROC-AUC:", round(raw_roc, 6))
print("PR-AUC:", round(raw_pr, 6))
print("Brier Score:", round(raw_brier, 6))

# 12. CALIBRATED MODEL PERFORMANCE

print("\n======================================")
print("CALIBRATED MODEL")
print("======================================")

calibrated_roc = roc_auc_score(
    y_test,
    test_probabilities_calibrated
)

calibrated_pr = average_precision_score(
    y_test,
    test_probabilities_calibrated
)

calibrated_brier = brier_score_loss(
    y_test,
    test_probabilities_calibrated
)

print(
    "ROC-AUC:",
    round(calibrated_roc, 6)
)

print(
    "PR-AUC:",
    round(calibrated_pr, 6)
)

print(
    "Brier Score:",
    round(calibrated_brier, 6)
)

# 13. CALIBRATION CURVE

print("\nCreating calibration curve...")

prob_true_raw, prob_pred_raw = calibration_curve(
    y_test,
    test_probabilities_raw,
    n_bins=10,
    strategy="quantile"
)

prob_true_cal, prob_pred_cal = calibration_curve(
    y_test,
    test_probabilities_calibrated,
    n_bins=10,
    strategy="quantile"
)


plt.figure(figsize=(8, 6))

plt.plot(
    prob_pred_raw,
    prob_true_raw,
    marker="o",
    label="Raw Model"
)

plt.plot(
    prob_pred_cal,
    prob_true_cal,
    marker="o",
    label="Calibrated Model"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect Calibration"
)

plt.xlabel("Predicted Probability")
plt.ylabel("Actual Fraud Rate")

plt.title("RiskShield Probability Calibration")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "calibration_curve.png",
    dpi=150
)

plt.show()

# 14. EXAMPLE RISK SCORES

print("\n======================================")
print("EXAMPLE RISK SCORES")
print("======================================")

examples = [
    0.01,
    0.05,
    0.10,
    0.20,
    0.40,
    0.60,
    0.80,
    0.95
]

for probability in examples:

    calibrated = calibration_model.predict_proba(
        np.array([[probability]])
    )[0, 1]

    print(
        f"Raw: {probability:.2f}"
        f" → Calibrated: {calibrated:.4f}"
    )


print("\n======================================")
print("CALIBRATION EXPERIMENT COMPLETE")
print("======================================")