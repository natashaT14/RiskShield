import warnings
warnings.filterwarnings("ignore")

import gc
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

# CONFIGURATION

DATA_PATH = "data/train_transaction.csv"
RESULTS_PATH = "models/tuning_results.csv"


print("=" * 70)
print("RISKSHEILD HISTGRADIENTBOOSTING HYPERPARAMETER TUNING")
print("=" * 70)

# 1. LOAD REQUIRED COLUMNS

print("\nLoading required columns...")

required_columns = [
    "TransactionDT",
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

    "P_emaildomain",
    "R_emaildomain",

    "dist1",
    "dist2",

    "isFraud"
]

df = pd.read_csv(
    DATA_PATH,
    usecols=required_columns
)

print(f"Dataset loaded: {df.shape}")

print(
    f"Memory usage: "
    f"{df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB"
)

# 2. TEMPORAL SORT

print("\nSorting transactions chronologically...")

df = (
    df
    .sort_values("TransactionDT")
    .reset_index(drop=True)
)

print("Temporal sorting complete.")

# 3. BASIC FEATURES

print("\nCreating basic features...")

df["log_amount"] = np.log1p(
    df["TransactionAmt"]
)

df["transaction_hour"] = (
    (df["TransactionDT"] // 3600) % 24
)

df["transaction_day"] = (
    df["TransactionDT"] // (3600 * 24)
)

df["day_of_week"] = (
    df["transaction_day"] % 7
)

df["missing_count"] = (
    df.isnull().sum(axis=1)
)

df["P_email_missing"] = (
    df["P_emaildomain"]
    .isna()
    .astype(np.int8)
)

df["R_email_missing"] = (
    df["R_emaildomain"]
    .isna()
    .astype(np.int8)
)

df["dist1_missing"] = (
    df["dist1"]
    .isna()
    .astype(np.int8)
)

df["dist2_missing"] = (
    df["dist2"]
    .isna()
    .astype(np.int8)
)

# 4. TEMPORAL TRAIN / TEST SPLIT

print("\nCreating temporal train/test split...")

split_index = int(
    len(df) * 0.80
)

train_df = df.iloc[
    :split_index
].copy()

test_df = df.iloc[
    split_index:
].copy()

print(
    f"Training rows: {len(train_df):,}"
)

print(
    f"Testing rows : {len(test_df):,}"
)

# 5. LEAKAGE-SAFE BEHAVIORAL FEATURES

print("\nCreating leakage-safe behavioral mappings...")


# CARD FREQUENCY

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


# ADDRESS FREQUENCY

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


# EMAIL FREQUENCY

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


# PRODUCT AVERAGE AMOUNT

product_mean = (
    train_df
    .groupby("ProductCD")["TransactionAmt"]
    .mean()
)

train_df["amount_vs_product_mean"] = (
    train_df["TransactionAmt"]
    /
    train_df["ProductCD"].map(product_mean)
)

test_df["amount_vs_product_mean"] = (
    test_df["TransactionAmt"]
    /
    test_df["ProductCD"].map(product_mean)
)


# CARD AVERAGE AMOUNT

card_mean = (
    train_df
    .groupby("card1")["TransactionAmt"]
    .mean()
)

train_df["amount_vs_card_mean"] = (
    train_df["TransactionAmt"]
    /
    train_df["card1"].map(card_mean)
)

test_df["amount_vs_card_mean"] = (
    test_df["TransactionAmt"]
    /
    test_df["card1"].map(card_mean)
)


# CLEAN INFINITE VALUES

for column in [
    "amount_vs_product_mean",
    "amount_vs_card_mean"
]:

    train_df[column] = (
        train_df[column]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(1)
    )

    test_df[column] = (
        test_df[column]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(1)
    )


print("Leakage-safe behavioral features created.")

# 6. FINAL FEATURES

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

    "P_emaildomain",
    "R_emaildomain",

    "log_amount",

    "transaction_hour",
    "transaction_day",
    "day_of_week",

    "missing_count",

    "P_email_missing",
    "R_email_missing",

    "dist1_missing",
    "dist2_missing",

    "card_frequency",
    "address_frequency",
    "email_frequency",

    "amount_vs_product_mean",
    "amount_vs_card_mean"
]


X_train = train_df[features].copy()
X_test = test_df[features].copy()

y_train = train_df["isFraud"].copy()
y_test = test_df["isFraud"].copy()

print(
    f"\nFinal feature count: {len(features)}"
)

# 7. FEATURE TYPES

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

def create_preprocessor():

    numeric_transformer = Pipeline(
        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_transformer = Pipeline(
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

    return ColumnTransformer(
        transformers=[

            (
                "num",
                numeric_transformer,
                numerical_features
            ),

            (
                "cat",
                categorical_transformer,
                categorical_features
            )
        ]
    )

# 9. HYPERPARAMETER CONFIGURATIONS

configs = [

    {
        "learning_rate": 0.05,
        "max_iter": 200,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0
    },

    {
        "learning_rate": 0.08,
        "max_iter": 200,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0
    },

    {
        "learning_rate": 0.10,
        "max_iter": 200,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0
    },

    {
        "learning_rate": 0.05,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0
    },

    {
        "learning_rate": 0.08,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0
    },

    {
        "learning_rate": 0.05,
        "max_iter": 300,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0
    },

    {
        "learning_rate": 0.08,
        "max_iter": 300,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0
    },

    {
        "learning_rate": 0.05,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 50,
        "l2_regularization": 1.0
    },

    {
        "learning_rate": 0.08,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 50,
        "l2_regularization": 1.0
    },

    {
        "learning_rate": 0.05,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 5.0
    }
]

# 10. TUNING

results = []

print("\n")
print("=" * 70)
print("STARTING CONTROLLED HYPERPARAMETER SEARCH")
print("=" * 70)

print(
    f"\nConfigurations to test: {len(configs)}"
)


for index, params in enumerate(
    configs,
    start=1
):

    print("\n")
    print("-" * 70)
    print(
        f"CONFIGURATION {index}/{len(configs)}"
    )
    print("-" * 70)

    print(
        f"learning_rate     : "
        f"{params['learning_rate']}"
    )

    print(
        f"max_iter          : "
        f"{params['max_iter']}"
    )

    print(
        f"max_leaf_nodes    : "
        f"{params['max_leaf_nodes']}"
    )

    print(
        f"min_samples_leaf  : "
        f"{params['min_samples_leaf']}"
    )

    print(
        f"l2_regularization : "
        f"{params['l2_regularization']}"
    )

    print("\nTraining...")

    model = Pipeline(
        steps=[

            (
                "preprocessor",
                create_preprocessor()
            ),

            (
                "model",
                HistGradientBoostingClassifier(

                    learning_rate=params[
                        "learning_rate"
                    ],

                    max_iter=params[
                        "max_iter"
                    ],

                    max_leaf_nodes=params[
                        "max_leaf_nodes"
                    ],

                    min_samples_leaf=params[
                        "min_samples_leaf"
                    ],

                    l2_regularization=params[
                        "l2_regularization"
                    ],

                    random_state=42
                )
            )
        ]
    )

    model.fit(
        X_train,
        y_train
    )

    print("Training complete.")

    print("Generating probabilities...")

    probabilities = (
        model
        .predict_proba(X_test)[:, 1]
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )

    print(
        f"PR-AUC : {pr_auc:.4f}"
    )

    results.append({

        "configuration": index,

        "learning_rate":
            params["learning_rate"],

        "max_iter":
            params["max_iter"],

        "max_leaf_nodes":
            params["max_leaf_nodes"],

        "min_samples_leaf":
            params["min_samples_leaf"],

        "l2_regularization":
            params["l2_regularization"],

        "ROC-AUC":
            roc_auc,

        "PR-AUC":
            pr_auc
    })

    del model
    del probabilities

    gc.collect()

# 11. FINAL RESULTS

results_df = pd.DataFrame(results)

results_df = (
    results_df
    .sort_values(
        "PR-AUC",
        ascending=False
    )
    .reset_index(drop=True)
)


print("\n\n")
print("=" * 70)
print("HYPERPARAMETER TUNING RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# 12. BEST CONFIGURATION

best = results_df.iloc[0]

print("\n")
print("=" * 70)
print("BEST CONFIGURATION")
print("=" * 70)

print(
    f"Configuration    : "
    f"{int(best['configuration'])}"
)

print(
    f"Learning rate    : "
    f"{best['learning_rate']}"
)

print(
    f"Max iterations   : "
    f"{int(best['max_iter'])}"
)

print(
    f"Max leaf nodes   : "
    f"{int(best['max_leaf_nodes'])}"
)

print(
    f"Min samples leaf : "
    f"{int(best['min_samples_leaf'])}"
)

print(
    f"L2 regularization: "
    f"{best['l2_regularization']}"
)

print(
    f"ROC-AUC          : "
    f"{best['ROC-AUC']:.4f}"
)

print(
    f"PR-AUC           : "
    f"{best['PR-AUC']:.4f}"
)

# 13. SAVE RESULTS

results_df.to_csv(
    RESULTS_PATH,
    index=False
)

print("\n")
print("=" * 70)
print("TUNING RESULTS SAVED")
print("=" * 70)

print(
    f"Saved to:\n{RESULTS_PATH}"
)

print("\n")
print("=" * 70)
print("RISKSHEILD HYPERPARAMETER TUNING COMPLETE")
print("=" * 70)