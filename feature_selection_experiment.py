import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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

df = df.sort_values(
    "TransactionDT"
).reset_index(drop=True)

# 3. BASIC FEATURE ENGINEERING

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
    df.isna()
    .sum(axis=1)
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

print(
    "\nCreating temporal train/test split..."
)

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

# 5. LEAKAGE-SAFE BEHAVIORAL FEATURES

print(
    "\nCreating leakage-safe behavioral features..."
)


def create_behavioral_features(
    train,
    test
):

    train = train.copy()
    test = test.copy()

    # --------------------------------------------------------
    # CARD FREQUENCY
    # --------------------------------------------------------

    card_counts = (
        train["card1"]
        .value_counts()
    )

    train["card_frequency"] = (
        train["card1"]
        .map(card_counts)
        .fillna(0)
    )

    test["card_frequency"] = (
        test["card1"]
        .map(card_counts)
        .fillna(0)
    )


    # --------------------------------------------------------
    # EMAIL FREQUENCY
    # --------------------------------------------------------

    email_counts = (
        train["P_emaildomain"]
        .value_counts()
    )

    train["email_frequency"] = (
        train["P_emaildomain"]
        .map(email_counts)
        .fillna(0)
    )

    test["email_frequency"] = (
        test["P_emaildomain"]
        .map(email_counts)
        .fillna(0)
    )


    # --------------------------------------------------------
    # ADDRESS FREQUENCY
    # --------------------------------------------------------

    address_counts = (
        train["addr1"]
        .value_counts()
    )

    train["address_frequency"] = (
        train["addr1"]
        .map(address_counts)
        .fillna(0)
    )

    test["address_frequency"] = (
        test["addr1"]
        .map(address_counts)
        .fillna(0)
    )


    # --------------------------------------------------------
    # PRODUCT AVERAGE AMOUNT
    # --------------------------------------------------------

    product_mean = (
        train
        .groupby("ProductCD")[
            "TransactionAmt"
        ]
        .mean()
    )

    train["amount_vs_product_mean"] = (
        train["TransactionAmt"] /
        train["ProductCD"]
        .map(product_mean)
        .replace(0, np.nan)
    )

    test["amount_vs_product_mean"] = (
        test["TransactionAmt"] /
        test["ProductCD"]
        .map(product_mean)
        .replace(0, np.nan)
    )


    # --------------------------------------------------------
    # CARD AVERAGE AMOUNT
    # --------------------------------------------------------

    card_mean = (
        train
        .groupby("card1")[
            "TransactionAmt"
        ]
        .mean()
    )

    train["amount_vs_card_mean"] = (
        train["TransactionAmt"] /
        train["card1"]
        .map(card_mean)
        .replace(0, np.nan)
    )

    test["amount_vs_card_mean"] = (
        test["TransactionAmt"] /
        test["card1"]
        .map(card_mean)
        .replace(0, np.nan)
    )

    return train, test


train_df, test_df = create_behavioral_features(
    train_df,
    test_df
)

print(
    "Leakage-safe behavioral features created!"
)

# 6. TARGET

y_train = train_df["isFraud"]

y_test = test_df["isFraud"]

# 7. FEATURE GROUPS

all_features = [
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

weak_removed = [
    feature
    for feature in all_features
    if feature not in [
        "dist1",
        "dist2_missing",
        "transaction_time"
    ]
]


no_card_identifiers = [
    feature
    for feature in all_features
    if feature not in [
        "card1",
        "card2",
        "card3",
        "card5",
        "card6"
    ]
]


behavioral_features = [
    "TransactionAmt",
    "ProductCD",

    "P_emaildomain",
    "R_emaildomain",

    "card_frequency",
    "email_frequency",
    "address_frequency",

    "amount_vs_product_mean",
    "amount_vs_card_mean",

    "missing_count",
    "day_of_week"
]


experiments = {

    "A_All_Features": all_features,

    "B_Weak_Features_Removed": weak_removed,

    "C_No_Card_Identifiers": no_card_identifiers,

    "D_Behavioral_Focused": behavioral_features

}

# 8. MODEL FUNCTION

def run_experiment(
    name,
    features
):

    print(
        "\n======================================"
    )

    print(
        "Running:",
        name
    )

    print(
        "Number of features:",
        len(features)
    )

    X_train = train_df[
        features
    ]

    X_test = test_df[
        features
    ]


    categorical_features = [
        feature
        for feature in features
        if feature in [
            "ProductCD",
            "card4",
            "card6",
            "P_emaildomain",
            "R_emaildomain"
        ]
    ]


    numerical_features = [
        feature
        for feature in features
        if feature not in categorical_features
    ]


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


    print("Training...")

    pipeline.fit(
        X_train,
        y_train
    )

    print(
        "Training complete!"
    )


    probabilities = (
        pipeline
        .predict_proba(X_test)[:, 1]
    )


    roc_auc = (
        roc_auc_score(
            y_test,
            probabilities
        )
    )


    pr_auc = (
        average_precision_score(
            y_test,
            probabilities
        )
    )


    print(
        "ROC-AUC:",
        round(roc_auc, 4)
    )

    print(
        "PR-AUC:",
        round(pr_auc, 4)
    )


    return {
        "Model": name,
        "Features": len(features),
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc
    }

# 9. RUN ALL EXPERIMENTS

results = []


for name, features in experiments.items():

    result = run_experiment(
        name,
        features
    )

    results.append(
        result
    )

# 10. RESULTS

results_df = pd.DataFrame(
    results
)

results_df = (
    results_df
    .sort_values(
        "PR-AUC",
        ascending=False
    )
)


print(
    "\n\n======================================"
)

print(
    "FINAL FEATURE SELECTION RESULTS"
)

print(
    "======================================"
)

print(
    results_df.to_string(
        index=False
    )
)