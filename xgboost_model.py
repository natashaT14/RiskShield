import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier

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


# Transaction amount
df["log_amount"] = np.log1p(df["TransactionAmt"])


# Time features
df["transaction_day"] = (
    df["TransactionDT"] // (24 * 60 * 60)
)

df["transaction_time"] = (
    df["TransactionDT"] % (24 * 60 * 60)
)

df["day_of_week"] = (
    df["transaction_day"] % 7
)


# Missingness
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

# 4. TEMPORAL TRAIN / TEST SPLIT

print("\nCreating temporal split...")

split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

print("Training transactions:", len(train_df))
print("Testing transactions:", len(test_df))

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

# 5. LEAKAGE-SAFE BEHAVIORAL FEATURES

print("\nCreating leakage-safe behavioral features...")


def add_behavioral_features(train, test):

    # --------------------------------------------------------
    # CARD FREQUENCY
    # --------------------------------------------------------

    card_frequency = (
        train.groupby("card1")
        .size()
    )

    train["card_frequency"] = (
        train["card1"]
        .map(card_frequency)
        .fillna(0)
    )

    test["card_frequency"] = (
        test["card1"]
        .map(card_frequency)
        .fillna(0)
    )


    # --------------------------------------------------------
    # EMAIL FREQUENCY
    # --------------------------------------------------------

    email_frequency = (
        train.groupby("P_emaildomain")
        .size()
    )

    train["email_frequency"] = (
        train["P_emaildomain"]
        .map(email_frequency)
        .fillna(0)
    )

    test["email_frequency"] = (
        test["P_emaildomain"]
        .map(email_frequency)
        .fillna(0)
    )


    # --------------------------------------------------------
    # ADDRESS FREQUENCY
    # --------------------------------------------------------

    address_frequency = (
        train.groupby("addr1")
        .size()
    )

    train["address_frequency"] = (
        train["addr1"]
        .map(address_frequency)
        .fillna(0)
    )

    test["address_frequency"] = (
        test["addr1"]
        .map(address_frequency)
        .fillna(0)
    )


    # --------------------------------------------------------
    # PRODUCT AVERAGE TRANSACTION
    # --------------------------------------------------------

    product_amount_mean = (
        train.groupby("ProductCD")["TransactionAmt"]
        .mean()
    )

    train["amount_vs_product_mean"] = (
        train["TransactionAmt"]
        /
        train["ProductCD"]
        .map(product_amount_mean)
        .replace(0, np.nan)
    )

    test["amount_vs_product_mean"] = (
        test["TransactionAmt"]
        /
        test["ProductCD"]
        .map(product_amount_mean)
        .replace(0, np.nan)
    )


    # --------------------------------------------------------
    # CARD AVERAGE TRANSACTION
    # --------------------------------------------------------

    card_amount_mean = (
        train.groupby("card1")["TransactionAmt"]
        .mean()
    )

    train["amount_vs_card_mean"] = (
        train["TransactionAmt"]
        /
        train["card1"]
        .map(card_amount_mean)
        .replace(0, np.nan)
    )

    test["amount_vs_card_mean"] = (
        test["TransactionAmt"]
        /
        test["card1"]
        .map(card_amount_mean)
        .replace(0, np.nan)
    )


    return train, test


train_df, test_df = add_behavioral_features(
    train_df,
    test_df
)

print("Leakage-safe behavioral features created!")

# 6. FEATURES

features = [

    # Transaction
    "TransactionAmt",
    "ProductCD",

    # Card
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",

    # Address
    "addr1",
    "addr2",
    "dist1",
    "dist2",

    # Email
    "P_emaildomain",
    "R_emaildomain",

    # Engineered
    "log_amount",

    "P_email_missing",
    "R_email_missing",

    "dist1_missing",
    "dist2_missing",

    "missing_count",

    "transaction_day",
    "transaction_time",

    "day_of_week",

    # Behavioral
    "card_frequency",
    "email_frequency",
    "address_frequency",

    "amount_vs_product_mean",
    "amount_vs_card_mean"
]


X_train = train_df[features].copy()
y_train = train_df["isFraud"].copy()

X_test = test_df[features].copy()
y_test = test_df["isFraud"].copy()

# 7. HANDLE CATEGORICAL FEATURES

print("\nEncoding categorical features...")


categorical_features = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain"
]


for column in categorical_features:

    X_train[column] = (
        X_train[column]
        .astype("category")
    )

    X_test[column] = (
        X_test[column]
        .astype("category")
    )


# Make train/test categories identical
for column in categorical_features:

    categories = (
        X_train[column]
        .cat.categories
    )

    X_train[column] = (
        X_train[column]
        .cat.set_categories(categories)
    )

    X_test[column] = (
        X_test[column]
        .cat.set_categories(categories)
    )


# Convert categorical columns to integer codes

for column in categorical_features:

    X_train[column] = (
        X_train[column]
        .cat.codes
        .astype("int32")
    )

    X_test[column] = (
        X_test[column]
        .cat.codes
        .astype("int32")
    )

# 8. HANDLE MISSING VALUES

X_train = X_train.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)

# XGBoost handles NaN natively

# 9. MODEL

print("\nTraining XGBoost RiskShield...")

fraud_count = y_train.sum()
legitimate_count = len(y_train) - fraud_count

scale_pos_weight = (
    legitimate_count / fraud_count
)

print(
    "Scale positive weight:",
    round(scale_pos_weight, 2)
)

model = XGBClassifier(

    n_estimators=500,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    min_child_weight=5,

    gamma=0,

    reg_alpha=0.1,

    reg_lambda=1.0,

    objective="binary:logistic",

    eval_metric="aucpr",

    scale_pos_weight=scale_pos_weight,

    tree_method="hist",

    random_state=42,

    n_jobs=1
)

# 10. TRAIN

model.fit(
    X_train,
    y_train
)

print("Training complete!")

# 11. PREDICTIONS

print("\nGenerating fraud probabilities...")

y_probability = (
    model.predict_proba(X_test)[:, 1]
)


# Default threshold
threshold = 0.5

y_pred = (
    y_probability >= threshold
).astype(int)

# 12. CLASSIFICATION REPORT

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)

# 13. CONFUSION MATRIX

print("\n========== CONFUSION MATRIX ==========")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# 14. ROC-AUC

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print("\n========== ROC-AUC ==========")

print(roc_auc)

# 15. PR-AUC

pr_auc = average_precision_score(
    y_test,
    y_probability
)

print("\n========== PR-AUC ==========")

print(pr_auc)

# 16. THRESHOLD ANALYSIS

print("\n========== THRESHOLD ANALYSIS ==========")


thresholds = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70
]


for threshold in thresholds:

    predictions = (
        y_probability >= threshold
    ).astype(int)

    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score
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

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f}"
    )

# 17. FEATURE IMPORTANCE

print("\n========== FEATURE IMPORTANCE ==========")

importance_df = pd.DataFrame({

    "feature": features,

    "importance": model.feature_importances_

}).sort_values(
    "importance",
    ascending=False
)


print(
    "\nTop 20 features:"
)

print(
    importance_df.head(20).to_string(
        index=False
    )
)

# 18. FINAL SUMMARY

print("\n======================================")
print("XGBOOST RISKSHIELD SUMMARY")
print("======================================")

print(
    f"ROC-AUC: {roc_auc:.4f}"
)

print(
    f"PR-AUC: {pr_auc:.4f}"
)

print(
    "\nModel: XGBoost"
)

print(
    "Validation: Temporal split"
)

print(
    "Behavioral features: Leakage-safe"
)

print(
    "Class imbalance: scale_pos_weight"
)

print(
    "\nXGBoost experiment complete!"
)