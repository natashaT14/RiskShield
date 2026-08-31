import pandas as pd
import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)

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

df["log_amount"] = np.log1p(
    df["TransactionAmt"]
)

df["P_email_missing"] = (
    df["P_emaildomain"]
    .isna()
    .astype(int)
)

df["R_email_missing"] = (
    df["R_emaildomain"]
    .isna()
    .astype(int)
)

df["dist1_missing"] = (
    df["dist1"]
    .isna()
    .astype(int)
)

df["dist2_missing"] = (
    df["dist2"]
    .isna()
    .astype(int)
)

df["missing_count"] = (
    df.isna().sum(axis=1)
)

df["transaction_day"] = (
    df["TransactionDT"]
    // (24 * 60 * 60)
)

df["transaction_time"] = (
    df["TransactionDT"]
    % (24 * 60 * 60)
)

print("Features created.")

# 4. FEATURES

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


X = df[features].copy()
y = df["isFraud"]

# 5. CATEGORICAL ENCODING

categorical_features = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain"
]

for col in categorical_features:

    X[col] = (
        X[col]
        .astype("category")
        .cat.codes
        .replace(-1, np.nan)
    )

# 6. TEMPORAL SPLIT

print("\nCreating temporal split...")

X_train = X.iloc[:80_000]
y_train = y.iloc[:80_000]

X_test = X.iloc[80_000:]
y_test = y.iloc[80_000:]


print(
    "Training transactions:",
    len(X_train)
)

print(
    "Testing transactions:",
    len(X_test)
)

# 7. TRAIN MODEL

print("\nTraining RiskShield...")

model = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("Training complete.")

# 8. PREDICTIONS

print("\nGenerating fraud probabilities...")

probabilities = model.predict_proba(
    X_test
)[:, 1]


print(
    "\nOverall ROC-AUC:",
    round(
        roc_auc_score(
            y_test,
            probabilities
        ),
        4
    )
)

print(
    "Overall PR-AUC:",
    round(
        average_precision_score(
            y_test,
            probabilities
        ),
        4
    )
)

# 9. COST ASSUMPTIONS

print("COST-SENSITIVE ANALYSIS")

# Cost of allowing fraud
COST_FALSE_NEGATIVE = 100

# Cost of blocking a legitimate transaction
COST_FALSE_POSITIVE = 10


print(
    "\nAssumed cost of FALSE NEGATIVE:",
    COST_FALSE_NEGATIVE
)

print(
    "Assumed cost of FALSE POSITIVE:",
    COST_FALSE_POSITIVE
)

# 10. THRESHOLD ANALYSIS

thresholds = np.arange(
    0.05,
    0.96,
    0.05
)


results = []


for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions
    ).ravel()

    total_cost = (
        fp * COST_FALSE_POSITIVE
        +
        fn * COST_FALSE_NEGATIVE
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    results.append({

        "threshold": threshold,

        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "total_cost": total_cost
    })


results_df = pd.DataFrame(
    results
)

# 11. DISPLAY RESULTS

print("\nThreshold comparison:")

print(
    results_df[
        [
            "threshold",
            "precision",
            "recall",
            "f1",
            "FP",
            "FN",
            "total_cost"
        ]
    ].to_string(
        index=False
    )
)

# 12. BEST THRESHOLD

best_row = results_df.loc[
    results_df["total_cost"].idxmin()
]

print("BEST THRESHOLD")

print(
    "Threshold:",
    round(
        best_row["threshold"],
        2
    )
)

print(
    "Precision:",
    round(
        best_row["precision"],
        4
    )
)

print(
    "Recall:",
    round(
        best_row["recall"],
        4
    )
)

print(
    "F1:",
    round(
        best_row["f1"],
        4
    )
)

print(
    "False Positives:",
    int(best_row["FP"])
)

print(
    "False Negatives:",
    int(best_row["FN"])
)

print(
    "Estimated Total Cost:",
    int(best_row["total_cost"])
)

# 13. BUSINESS DECISION

print("RISKSHIELD DECISION POLICY")

best_threshold = best_row["threshold"]

print(
    f"""
Risk Score < {best_threshold:.2f}
→ APPROVE

Risk Score >= {best_threshold:.2f}
→ FLAG FOR FRAUD REVIEW
"""
)

print("\nCost-sensitive experiment complete.")