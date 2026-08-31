import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
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

# 2. SORT CHRONOLOGICALLY

print("\nSorting transactions chronologically...")

df = (
    df
    .sort_values("TransactionDT")
    .reset_index(drop=True)
)

print("Transactions sorted by time.")

# 3. BASIC FEATURES

print("\nCreating basic features...")

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
    df["TransactionDT"] //
    (24 * 60 * 60)
)

df["transaction_time"] = (
    df["TransactionDT"] %
    (24 * 60 * 60)
)

df["day_of_week"] = (
    df["transaction_day"] % 7
)

# 4. TEMPORAL TRAIN / TEST SPLIT

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

# 5. POPULATION-LEVEL FEATURES

print(
    "\nCreating cold-start behavioral features..."
)

# ------------------------------------------------------------
# Product frequency
# ------------------------------------------------------------

product_frequency = (
    train_df["ProductCD"]
    .value_counts()
)

train_df["product_frequency"] = (
    train_df["ProductCD"]
    .map(product_frequency)
    .fillna(0)
)

test_df["product_frequency"] = (
    test_df["ProductCD"]
    .map(product_frequency)
    .fillna(0)
)


# ------------------------------------------------------------
# Email frequency
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
# Address frequency
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

# 6. GLOBAL AMOUNT STATISTICS

global_amount_mean = (
    train_df["TransactionAmt"]
    .mean()
)

global_amount_std = (
    train_df["TransactionAmt"]
    .std()
)

train_df["amount_vs_global_mean"] = (
    train_df["TransactionAmt"]
    / global_amount_mean
)

test_df["amount_vs_global_mean"] = (
    test_df["TransactionAmt"]
    / global_amount_mean
)

train_df["amount_zscore_global"] = (
    (
        train_df["TransactionAmt"]
        - global_amount_mean
    )
    / global_amount_std
)

test_df["amount_zscore_global"] = (
    (
        test_df["TransactionAmt"]
        - global_amount_mean
    )
    / global_amount_std
)

# 7. PRODUCT-LEVEL AMOUNT STATISTICS

product_amount_mean = (
    train_df
    .groupby("ProductCD")["TransactionAmt"]
    .mean()
)

product_amount_std = (
    train_df
    .groupby("ProductCD")["TransactionAmt"]
    .std()
)


train_df["amount_vs_product_mean"] = (
    train_df["TransactionAmt"]
    /
    train_df["ProductCD"]
    .map(product_amount_mean)
)


test_df["amount_vs_product_mean"] = (
    test_df["TransactionAmt"]
    /
    test_df["ProductCD"]
    .map(product_amount_mean)
)


train_df["amount_zscore_product"] = (
    (
        train_df["TransactionAmt"]
        -
        train_df["ProductCD"]
        .map(product_amount_mean)
    )
    /
    train_df["ProductCD"]
    .map(product_amount_std)
)


test_df["amount_zscore_product"] = (
    (
        test_df["TransactionAmt"]
        -
        test_df["ProductCD"]
        .map(product_amount_mean)
    )
    /
    test_df["ProductCD"]
    .map(product_amount_std)
)

# 8. EMAIL + PRODUCT INTERACTION

train_df["email_product"] = (
    train_df["P_emaildomain"].astype(str)
    + "_"
    + train_df["ProductCD"].astype(str)
)

test_df["email_product"] = (
    test_df["P_emaildomain"].astype(str)
    + "_"
    + test_df["ProductCD"].astype(str)
)


email_product_frequency = (
    train_df["email_product"]
    .value_counts()
)

train_df["email_product_frequency"] = (
    train_df["email_product"]
    .map(email_product_frequency)
    .fillna(0)
)

test_df["email_product_frequency"] = (
    test_df["email_product"]
    .map(email_product_frequency)
    .fillna(0)
)

# 9. ADDRESS + PRODUCT INTERACTION

train_df["address_product"] = (
    train_df["addr1"].astype(str)
    + "_"
    + train_df["ProductCD"].astype(str)
)

test_df["address_product"] = (
    test_df["addr1"].astype(str)
    + "_"
    + test_df["ProductCD"].astype(str)
)


address_product_frequency = (
    train_df["address_product"]
    .value_counts()
)

train_df["address_product_frequency"] = (
    train_df["address_product"]
    .map(address_product_frequency)
    .fillna(0)
)

test_df["address_product_frequency"] = (
    test_df["address_product"]
    .map(address_product_frequency)
    .fillna(0)
)

# 10. REMOVE RAW INTERACTION COLUMNS

train_df.drop(
    columns=[
        "email_product",
        "address_product"
    ],
    inplace=True
)

test_df.drop(
    columns=[
        "email_product",
        "address_product"
    ],
    inplace=True
)


print(
    "Cold-start behavioral features created!"
)

# 11. FEATURES

features = [

    # Transaction
    "TransactionAmt",
    "ProductCD",

    # Context
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",

    # Other transaction attributes
    "dist1",
    "dist2",

    # Basic engineered
    "log_amount",
    "P_email_missing",
    "R_email_missing",
    "dist1_missing",
    "dist2_missing",
    "missing_count",
    "transaction_day",
    "transaction_time",
    "day_of_week",

    # Population-level behavior
    "product_frequency",
    "email_frequency",
    "address_frequency",

    # Amount anomalies
    "amount_vs_global_mean",
    "amount_zscore_global",
    "amount_vs_product_mean",
    "amount_zscore_product",

    # Context interactions
    "email_product_frequency",
    "address_product_frequency"
]

# 12. TARGET

X_train = train_df[features]
y_train = train_df["isFraud"]

X_test = test_df[features]
y_test = test_df["isFraud"]

# 13. FEATURE TYPES

categorical_features = [
    "ProductCD",
    "P_emaildomain",
    "R_emaildomain"
]


numerical_features = [
    feature
    for feature in features
    if feature not in categorical_features
]

# 14. PREPROCESSING

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

# 15. MODEL

model = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.05,
    max_leaf_nodes=31,
    random_state=42
)


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

# 16. TRAIN

print(
    "\nTraining cold-start RiskShield..."
)

pipeline.fit(
    X_train,
    y_train
)

print(
    "Training complete!"
)

# 17. PREDICTIONS

print(
    "\nGenerating predictions..."
)

y_probability = (
    pipeline
    .predict_proba(X_test)[:, 1]
)

# 18. OVERALL PERFORMANCE

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

pr_auc = average_precision_score(
    y_test,
    y_probability
)


print(
    "\n========== OVERALL PERFORMANCE =========="
)

print(
    "ROC-AUC:",
    round(roc_auc, 4)
)

print(
    "PR-AUC:",
    round(pr_auc, 4)
)

# 19. IDENTIFY UNSEEN CARDS

print(
    "\nAnalyzing unseen cards..."
)

training_cards = set(
    train_df["card1"]
    .dropna()
    .unique()
)

# card1 isn't used as a feature,
# but we use it ONLY to evaluate
# generalization.

unseen_mask = (
    ~test_df["card1"]
    .isin(training_cards)
)

seen_mask = (
    test_df["card1"]
    .isin(training_cards)
)

# 20. SEEN CARDS

seen_y = y_test[seen_mask]
seen_probability = y_probability[seen_mask]


print(
    "\n========== SEEN CARDS =========="
)

print(
    "Transactions:",
    len(seen_y)
)

print(
    "Fraud:",
    int(seen_y.sum())
)

if len(seen_y.unique()) > 1:

    print(
        "ROC-AUC:",
        round(
            roc_auc_score(
                seen_y,
                seen_probability
            ),
            4
        )
    )

    print(
        "PR-AUC:",
        round(
            average_precision_score(
                seen_y,
                seen_probability
            ),
            4
        )
    )

# 21. UNSEEN CARDS

unseen_y = y_test[unseen_mask]
unseen_probability = y_probability[unseen_mask]


print(
    "\n========== UNSEEN CARDS =========="
)

print(
    "Transactions:",
    len(unseen_y)
)

print(
    "Fraud:",
    int(unseen_y.sum())
)

if len(unseen_y.unique()) > 1:

    print(
        "ROC-AUC:",
        round(
            roc_auc_score(
                unseen_y,
                unseen_probability
            ),
            4
        )
    )

    print(
        "PR-AUC:",
        round(
            average_precision_score(
                unseen_y,
                unseen_probability
            ),
            4
        )
    )

# 22. SUMMARY

print(
    "\n========== COLD-START SUMMARY =========="
)

print(
    "Unseen-card percentage:",
    round(
        unseen_mask.mean() * 100,
        2
    ),
    "%"
)

print(
    "\nCold-start model does NOT use:"
)

print(
    "  - card1"
)

print(
    "  - card2"
)

print(
    "  - card3"
)

print(
    "  - card5"
)

print(
    "  - card6"
)

print(
    "\nIt relies on transaction + population context."
)