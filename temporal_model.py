import pandas as pd
import numpy as np

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

# 2. SORT BY TIME

print("\nSorting transactions chronologically...")

df = df.sort_values(
    "TransactionDT"
).reset_index(drop=True)

print("Transactions sorted by time.")

# 3. BASIC FEATURE ENGINEERING

print("\nCreating basic features...")

# Transaction amount
df["log_amount"] = np.log1p(
    df["TransactionAmt"]
)

# Missingness
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

# 4. TIME FEATURES

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

# 5. TEMPORAL TRAIN / TEST SPLIT

print("\nCreating temporal train/test split...")

split_index = int(
    len(df) * 0.8
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
    "\nTraining time range:",
    train_df["TransactionDT"].min(),
    "→",
    train_df["TransactionDT"].max()
)

print(
    "Testing time range:",
    test_df["TransactionDT"].min(),
    "→",
    test_df["TransactionDT"].max()
)

# 6. TARGET

y_train = train_df["isFraud"]

y_test = test_df["isFraud"]

# 7. LEAKAGE-SAFE BEHAVIORAL FEATURES

print(
    "\nCreating behavioral features "
    "using TRAIN data only..."
)


# ------------------------------------------------------------
# PRODUCT MEAN
# ------------------------------------------------------------

product_mean = (
    train_df
    .groupby("ProductCD")["TransactionAmt"]
    .mean()
)

global_mean = (
    train_df["TransactionAmt"].mean()
)

train_df["amount_vs_product_mean"] = (
    train_df["TransactionAmt"] /
    train_df["ProductCD"]
    .map(product_mean)
    .fillna(global_mean)
)

test_df["amount_vs_product_mean"] = (
    test_df["TransactionAmt"] /
    test_df["ProductCD"]
    .map(product_mean)
    .fillna(global_mean)
)


# ------------------------------------------------------------
# CARD MEAN
# ------------------------------------------------------------

card_mean = (
    train_df
    .groupby("card1")["TransactionAmt"]
    .mean()
)

train_df["amount_vs_card_mean"] = (
    train_df["TransactionAmt"] /
    train_df["card1"]
    .map(card_mean)
    .fillna(global_mean)
)

test_df["amount_vs_card_mean"] = (
    test_df["TransactionAmt"] /
    test_df["card1"]
    .map(card_mean)
    .fillna(global_mean)
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

print(
    "Leakage-safe behavioral "
    "features created!"
)

# 8. FEATURES

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

    "log_amount",

    "P_email_missing",
    "R_email_missing",
    "dist1_missing",
    "dist2_missing",
    "missing_count",

    "transaction_day",
    "transaction_time",
    "transaction_hour",
    "is_night",
    "day_of_week",
    "is_weekend",

    "amount_vs_product_mean",
    "amount_vs_card_mean",
    "card_frequency",
    "email_frequency",
    "address_frequency"
]


X_train = train_df[
    categorical_features +
    numerical_features
]

X_test = test_df[
    categorical_features +
    numerical_features
]

# 9. PREPROCESSING

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

# 10. MODEL

model = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.05,
    max_leaf_nodes=31,
    random_state=42
)

# 11. PIPELINE

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
# 12. TRAIN

print(
    "\nTraining temporal HistGradientBoosting..."
)

pipeline.fit(
    X_train,
    y_train
)

print("Training complete!")

# 13. PREDICTIONS

y_probability = (
    pipeline
    .predict_proba(X_test)[:, 1]
)

y_pred = (
    y_probability >= 0.5
).astype(int)

# 14. CLASSIFICATION REPORT

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

# 15. CONFUSION MATRIX

print(
    "\n========== CONFUSION MATRIX =========="
)

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# 16. ROC-AUC

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print(
    "\n========== ROC-AUC =========="
)

print(roc_auc)

# 17. PR-AUC

pr_auc = average_precision_score(
    y_test,
    y_probability
)

print(
    "\n========== PR-AUC =========="
)

print(pr_auc)

# 18. THRESHOLD ANALYSIS

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
    
# 19. FEATURE IMPORTANCE

from sklearn.inspection import permutation_importance

print(
    "\n========== FEATURE IMPORTANCE =========="
)

print(
    "Preparing test sample for permutation importance..."
)

# ------------------------------------------------------------
# Use a smaller sample to reduce memory usage
# ------------------------------------------------------------

importance_sample_size = min(
    5000,
    len(X_test)
)

X_importance = X_test.sample(
    n=importance_sample_size,
    random_state=42
)

y_importance = y_test.loc[
    X_importance.index
]

print(
    "Importance sample size:",
    len(X_importance)
)


# ------------------------------------------------------------
# Calculate permutation importance
# ------------------------------------------------------------

print(
    "Calculating permutation importance..."
)

importance = permutation_importance(
    pipeline,
    X_importance,
    y_importance,
    scoring="average_precision",
    n_repeats=3,
    random_state=42,
    n_jobs=1
)


# ------------------------------------------------------------
# Get original feature names
# ------------------------------------------------------------

feature_names = (
    X_importance.columns.tolist()
)


# ------------------------------------------------------------
# Create importance DataFrame
# ------------------------------------------------------------

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance.importances_mean
})


# ------------------------------------------------------------
# Sort by importance
# ------------------------------------------------------------

importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False
    )
)


# ------------------------------------------------------------
# Display results
# ------------------------------------------------------------

print(
    "\nTop 20 most important features:"
)

print(
    importance_df
    .head(20)
    .to_string(index=False)
)