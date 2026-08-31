import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)

# 1. LOAD DATA

print("Loading dataset...")

required_columns = [
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
    "TransactionDT",
    "isFraud"
]

df = pd.read_csv(
    "data/train_transaction.csv",
    usecols=required_columns,
    nrows=100_000
)

print("Dataset loaded:", df.shape)

# 2. FEATURE ENGINEERING

print("\nCreating behavioral features...")


# ------------------------------------------------------------
# A. Amount features
# ------------------------------------------------------------

df["log_amount"] = np.log1p(
    df["TransactionAmt"]
)

# Average transaction amount by ProductCD
product_amount_mean = (
    df.groupby("ProductCD")["TransactionAmt"]
    .transform("mean")
)

df["amount_vs_product_mean"] = (
    df["TransactionAmt"] /
    (product_amount_mean + 1e-6)
)


# Average transaction amount by card1
card_amount_mean = (
    df.groupby("card1")["TransactionAmt"]
    .transform("mean")
)

df["amount_vs_card_mean"] = (
    df["TransactionAmt"] /
    (card_amount_mean + 1e-6)
)


# ------------------------------------------------------------
# B. Frequency features
# ------------------------------------------------------------

# How often does this card appear?
card_frequency = (
    df.groupby("card1")["card1"]
    .transform("count")
)

df["card_frequency"] = card_frequency


# How often does this email domain appear?
email_frequency = (
    df.groupby("P_emaildomain")["P_emaildomain"]
    .transform("count")
)

df["email_frequency"] = email_frequency


# How often does this address appear?
address_frequency = (
    df.groupby("addr1")["addr1"]
    .transform("count")
)

df["address_frequency"] = address_frequency


# ------------------------------------------------------------
# C. Missingness features
# ------------------------------------------------------------

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
    df.isna()
    .sum(axis=1)
)


# ------------------------------------------------------------
# D. Time features
# ------------------------------------------------------------

SECONDS_PER_DAY = 24 * 60 * 60

df["transaction_day"] = (
    df["TransactionDT"] //
    SECONDS_PER_DAY
)

df["transaction_time"] = (
    df["TransactionDT"] %
    SECONDS_PER_DAY
)

df["transaction_hour"] = (
    df["transaction_time"] // 3600
)

df["is_night"] = (
    (df["transaction_hour"] < 6) |
    (df["transaction_hour"] >= 22)
).astype(int)

df["day_of_week"] = (
    df["transaction_day"] % 7
)

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)


print("Behavioral feature engineering complete!")

# 3. SELECT FEATURES

features = [

    # Original
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

    # Amount
    "log_amount",
    "amount_vs_product_mean",
    "amount_vs_card_mean",

    # Frequency
    "card_frequency",
    "email_frequency",
    "address_frequency",

    # Missingness
    "P_email_missing",
    "R_email_missing",
    "dist1_missing",
    "dist2_missing",
    "missing_count",

    # Time
    "transaction_day",
    "transaction_time",
    "transaction_hour",
    "is_night",
    "day_of_week",
    "is_weekend"
]

X = df[features]

y = df["isFraud"]

# 4. FEATURE TYPES

categorical_features = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain"
]

numerical_features = [
    "TransactionAmt",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "dist1",
    "dist2",

    # Amount
    "log_amount",
    "amount_vs_product_mean",
    "amount_vs_card_mean",

    # Frequency
    "card_frequency",
    "email_frequency",
    "address_frequency",

    # Missingness
    "P_email_missing",
    "R_email_missing",
    "dist1_missing",
    "dist2_missing",
    "missing_count",

    # Time
    "transaction_day",
    "transaction_time",
    "transaction_hour",
    "is_night",
    "day_of_week",
    "is_weekend"
]

# 5. PREPROCESSING

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
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
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

# 6. MODEL

model = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.05,
    max_leaf_nodes=31,
    random_state=42
)

# 7. PIPELINE

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

# 8. TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 9. TRAIN

print(
    "\nTraining HistGradientBoosting "
    "with behavioral features..."
)

pipeline.fit(
    X_train,
    y_train
)

print("Training complete!")

# 10. PREDICTIONS

y_probability = (
    pipeline
    .predict_proba(X_test)[:, 1]
)

y_pred = (
    y_probability >= 0.5
).astype(int)

# 11. EVALUATION

print(
    "\n========== CLASSIFICATION REPORT =========="
)

print(
    classification_report(
        y_test,
        y_pred,
        digits=4,
        zero_division=0
    )
)


print(
    "\n========== CONFUSION MATRIX =========="
)

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


print(
    "\n========== ROC-AUC =========="
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print(roc_auc)


print(
    "\n========== PR-AUC =========="
)

pr_auc = average_precision_score(
    y_test,
    y_probability
)

print(pr_auc)

# 12. THRESHOLD ANALYSIS

print(
    "\n========== THRESHOLD ANALYSIS =========="
)

thresholds = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7
]

for threshold in thresholds:

    y_threshold = (
        y_probability >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        y_threshold,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_threshold,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_threshold,
        zero_division=0
    )

    print(
        f"Threshold: {threshold:.1f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f}"
    )