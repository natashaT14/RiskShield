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

df = pd.read_csv(
    "data/train_transaction.csv",
    nrows=100_000
)

print("Dataset loaded:", df.shape)

# 2. SELECT FEATURES

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
    "R_emaildomain"
]

X = df[features]
y = df["isFraud"]

print("\nFeatures:", features)
print("Target distribution:")
print(y.value_counts())

# 3. IDENTIFY FEATURE TYPES

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
    "dist2"
]

# 4. PREPROCESSING

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
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

# 5. MODEL

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

# 6. COMPLETE PIPELINE

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# 7. TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 8. TRAIN

print("\nTraining Logistic Regression...")

pipeline.fit(X_train, y_train)

print("Training complete!")

# 9. PREDICTIONS

y_pred = pipeline.predict(X_test)

y_probability = pipeline.predict_proba(X_test)[:, 1]

# 10. EVALUATION

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