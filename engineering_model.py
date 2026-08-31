import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
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
print("Memory usage:", df.memory_usage(deep=True).sum() / 1024**2, "MB")

# 2. FEATURE ENGINEERING

print("\nCreating engineered features...")

# Log transaction amount
df["log_amount"] = np.log1p(df["TransactionAmt"])

# Missingness indicators
df["P_email_missing"] = df["P_emaildomain"].isna().astype(int)
df["R_email_missing"] = df["R_emaildomain"].isna().astype(int)

df["dist1_missing"] = df["dist1"].isna().astype(int)
df["dist2_missing"] = df["dist2"].isna().astype(int)

# Total number of missing values
df["missing_count"] = df.isna().sum(axis=1)

# Relative time features
df["transaction_day"] = df["TransactionDT"] // (24 * 60 * 60)

df["transaction_time"] = (
    df["TransactionDT"] % (24 * 60 * 60)
)

print("Feature engineering complete!")

# 3. SELECT FEATURES

features = [
    # Original features
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

    # Engineered features
    "log_amount",
    "P_email_missing",
    "R_email_missing",
    "dist1_missing",
    "dist2_missing",
    "missing_count",
    "transaction_day",
    "transaction_time"
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

    # Engineered
    "log_amount",
    "P_email_missing",
    "R_email_missing",
    "dist1_missing",
    "dist2_missing",
    "missing_count",
    "transaction_day",
    "transaction_time"
]

# 5. PREPROCESSING

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
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

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

# 7. PIPELINE

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
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

print("\nTraining engineered Logistic Regression...")

pipeline.fit(X_train, y_train)

print("Training complete!")

# 10. PREDICTIONS

y_pred = pipeline.predict(X_test)

y_probability = pipeline.predict_proba(X_test)[:, 1]

# 11. EVALUATION

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)

print("\n========== CONFUSION MATRIX ==========")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

print("\n========== ROC-AUC ==========")

print(
    roc_auc_score(
        y_test,
        y_probability
    )
)

print("\n========== PR-AUC ==========")

print(
    average_precision_score(
        y_test,
        y_probability
    )
)

# 12. THRESHOLD ANALYSIS

from sklearn.metrics import precision_score, recall_score, f1_score

print("\n========== THRESHOLD ANALYSIS ==========")

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

for threshold in thresholds:

    y_threshold = (y_probability >= threshold).astype(int)

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