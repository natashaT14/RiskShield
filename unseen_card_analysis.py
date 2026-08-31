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

# 4. TEMPORAL SPLIT

print("\nCreating temporal split...")

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

# 5. IDENTIFY SEEN / UNSEEN CARDS

print(
    "\nAnalyzing seen vs unseen cards..."
)

training_cards = set(
    train_df["card1"]
    .dropna()
    .unique()
)

test_df["card_seen_in_training"] = (
    test_df["card1"]
    .isin(training_cards)
)


seen_test = test_df[
    test_df["card_seen_in_training"]
].copy()

unseen_test = test_df[
    ~test_df["card_seen_in_training"]
].copy()


print(
    "\n========== CARD GENERALIZATION =========="
)

print(
    "Unique cards in training:",
    len(training_cards)
)

print(
    "Test transactions with SEEN cards:",
    len(seen_test)
)

print(
    "Test transactions with UNSEEN cards:",
    len(unseen_test)
)

# 6. LEAKAGE-SAFE BEHAVIORAL FEATURES

print(
    "\nCreating behavioral features..."
)


# Card frequency
card_counts = (
    train_df["card1"]
    .value_counts()
)

train_df["card_frequency"] = (
    train_df["card1"]
    .map(card_counts)
    .fillna(0)
)

test_df["card_frequency"] = (
    test_df["card1"]
    .map(card_counts)
    .fillna(0)
)


# Email frequency
email_counts = (
    train_df["P_emaildomain"]
    .value_counts()
)

train_df["email_frequency"] = (
    train_df["P_emaildomain"]
    .map(email_counts)
    .fillna(0)
)

test_df["email_frequency"] = (
    test_df["P_emaildomain"]
    .map(email_counts)
    .fillna(0)
)


# Address frequency
address_counts = (
    train_df["addr1"]
    .value_counts()
)

train_df["address_frequency"] = (
    train_df["addr1"]
    .map(address_counts)
    .fillna(0)
)

test_df["address_frequency"] = (
    test_df["addr1"]
    .map(address_counts)
    .fillna(0)
)


# Product average amount
product_mean = (
    train_df
    .groupby("ProductCD")["TransactionAmt"]
    .mean()
)

train_df["amount_vs_product_mean"] = (
    train_df["TransactionAmt"] /
    train_df["ProductCD"]
    .map(product_mean)
)

test_df["amount_vs_product_mean"] = (
    test_df["TransactionAmt"] /
    test_df["ProductCD"]
    .map(product_mean)
)

# Card average amount
card_mean = (
    train_df
    .groupby("card1")["TransactionAmt"]
    .mean()
)

train_df["amount_vs_card_mean"] = (
    train_df["TransactionAmt"] /
    train_df["card1"]
    .map(card_mean)
)

test_df["amount_vs_card_mean"] = (
    test_df["TransactionAmt"] /
    test_df["card1"]
    .map(card_mean)
)

# 7. FEATURES

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
    "transaction_time",
    "day_of_week",

    "card_frequency",
    "email_frequency",
    "address_frequency",

    "amount_vs_product_mean",
    "amount_vs_card_mean"
]


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

# 8. PREPROCESSOR

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

# 9. MODEL

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

# 10. TRAIN

print(
    "\nTraining RiskShield..."
)

pipeline.fit(
    train_df[features],
    train_df["isFraud"]
)

print(
    "Training complete!"
)

# 11. PREDICT TEST SET

print(
    "\nGenerating predictions..."
)

test_probabilities = (
    pipeline
    .predict_proba(
        test_df[features]
    )[:, 1]
)

# 12. OVERALL PERFORMANCE

overall_roc = roc_auc_score(
    test_df["isFraud"],
    test_probabilities
)

overall_pr = average_precision_score(
    test_df["isFraud"],
    test_probabilities
)


print(
    "\n========== OVERALL =========="
)

print(
    "ROC-AUC:",
    round(overall_roc, 4)
)

print(
    "PR-AUC:",
    round(overall_pr, 4)
)

# 13. SEEN CARD PERFORMANCE

seen_mask = (
    test_df["card_seen_in_training"]
    .values
)

unseen_mask = (
    ~test_df["card_seen_in_training"]
    .values
)


seen_y = (
    test_df.loc[
        seen_mask,
        "isFraud"
    ]
)

seen_prob = (
    test_probabilities[
        seen_mask
    ]
)


unseen_y = (
    test_df.loc[
        unseen_mask,
        "isFraud"
    ]
)

unseen_prob = (
    test_probabilities[
        unseen_mask
    ]
)


print(
    "\n========== SEEN CARDS =========="
)

if len(seen_y.unique()) > 1:

    print(
        "Transactions:",
        len(seen_y)
    )

    print(
        "Fraud:",
        int(seen_y.sum())
    )

    print(
        "ROC-AUC:",
        round(
            roc_auc_score(
                seen_y,
                seen_prob
            ),
            4
        )
    )

    print(
        "PR-AUC:",
        round(
            average_precision_score(
                seen_y,
                seen_prob
            ),
            4
        )
    )

else:

    print(
        "Not enough class diversity."
    )

# 14. UNSEEN CARD PERFORMANCE

print(
    "\n========== UNSEEN CARDS =========="
)

if len(unseen_y.unique()) > 1:

    print(
        "Transactions:",
        len(unseen_y)
    )

    print(
        "Fraud:",
        int(unseen_y.sum())
    )

    print(
        "ROC-AUC:",
        round(
            roc_auc_score(
                unseen_y,
                unseen_prob
            ),
            4
        )
    )

    print(
        "PR-AUC:",
        round(
            average_precision_score(
                unseen_y,
                unseen_prob
            ),
            4
        )
    )

else:

    print(
        "Not enough class diversity."
    )

# 15. UNSEEN CARD PERCENTAGE

unseen_percentage = (
    len(unseen_test) /
    len(test_df)
) * 100


print(
    "\n========== SUMMARY =========="
)

print(
    "Unseen-card percentage:",
    round(
        unseen_percentage,
        2
    ),
    "%"
)