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

# 2. BASIC FEATURES

print("\nCreating basic features...")

df["log_amount"] = np.log1p(
    df["TransactionAmt"]
)

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

df["missing_count"] = (
    df.isna().sum(axis=1)
)

# 3. TIME FEATURES

SECONDS_PER_DAY = 24 * 60 * 60

df["transaction_day"] = (
    df["TransactionDT"] // SECONDS_PER_DAY
)

df["transaction_time"] = (
    df["TransactionDT"] % SECONDS_PER_DAY
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

# 4. TARGET / FEATURES

y = df["isFraud"]

X = df.drop(
    columns=["isFraud"]
)

# 5. TRAIN / TEST SPLIT

print("\nSplitting data BEFORE behavioral statistics...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# 6. LEAKAGE-SAFE FEATURE FUNCTION

def add_behavioral_features(
    train_df,
    test_df
):

    train_df = train_df.copy()
    test_df = test_df.copy()

    print("\nCreating leakage-safe behavioral features...")


    # --------------------------------------------------------
    # A. PRODUCT AVERAGE AMOUNT
    # --------------------------------------------------------

    product_mean = (
        train_df
        .groupby("ProductCD")["TransactionAmt"]
        .mean()
    )

    train_df["amount_vs_product_mean"] = (
        train_df["TransactionAmt"] /
        train_df["ProductCD"]
        .map(product_mean)
        .fillna(train_df["TransactionAmt"].median())
    )

    test_df["amount_vs_product_mean"] = (
        test_df["TransactionAmt"] /
        test_df["ProductCD"]
        .map(product_mean)
        .fillna(train_df["TransactionAmt"].median())
    )


    # --------------------------------------------------------
    # B. CARD AVERAGE AMOUNT
    # --------------------------------------------------------

    card_mean = (
        train_df
        .groupby("card1")["TransactionAmt"]
        .mean()
    )

    global_mean = (
        train_df["TransactionAmt"].mean()
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


    # --------------------------------------------------------
    # C. CARD FREQUENCY
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # D. EMAIL FREQUENCY
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # E. ADDRESS FREQUENCY
    # --------------------------------------------------------

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

    return train_df, test_df

# 7. APPLY FEATURE ENGINEERING

X_train, X_test = add_behavioral_features(
    X_train,
    X_test
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

    # Leakage-safe behavioral features
    "amount_vs_product_mean",
    "amount_vs_card_mean",
    "card_frequency",
    "email_frequency",
    "address_frequency"
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
    "\nTraining leakage-safe "
    "HistGradientBoosting..."
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

# 14. EVALUATION

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



# 15. THRESHOLD ANALYSIS


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