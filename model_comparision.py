import warnings
warnings.filterwarnings("ignore")

import gc
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from xgboost import XGBClassifier

# CONFIGURATION

DATA_PATH = "data/train_transaction.csv"

print("=" * 70)
print("RISKSHIELD MEMORY-EFFICIENT MODEL COMPARISON")
print("=" * 70)

# 1. LOAD ONLY REQUIRED COLUMNS

print("\nLoading only required columns...")

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

# 2. SORT TEMPORALLY

print("\nSorting transactions chronologically...")

df = df.sort_values(
    "TransactionDT"
).reset_index(drop=True)

print("Transactions sorted.")

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

print("\nCreating temporal split...")

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
    f"Training transactions: "
    f"{len(train_df)}"
)

print(
    f"Testing transactions: "
    f"{len(test_df)}"
)

print(
    f"Training time: "
    f"{train_df['TransactionDT'].min()} → "
    f"{train_df['TransactionDT'].max()}"
)

print(
    f"Testing time: "
    f"{test_df['TransactionDT'].min()} → "
    f"{test_df['TransactionDT'].max()}"
)

# 5. LEAKAGE-SAFE BEHAVIORAL FEATURES

print("\nCreating leakage-safe behavioral features...")


# ------------------------------------------------------------
# CARD FREQUENCY
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# ADDRESS FREQUENCY
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# EMAIL FREQUENCY
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# PRODUCT AVERAGE AMOUNT
# ------------------------------------------------------------

product_mean = (
    train_df
    .groupby("ProductCD")["TransactionAmt"]
    .mean()
)

train_df["amount_vs_product_mean"] = (
    train_df["TransactionAmt"]
    /
    train_df["ProductCD"]
    .map(product_mean)
)

test_df["amount_vs_product_mean"] = (
    test_df["TransactionAmt"]
    /
    test_df["ProductCD"]
    .map(product_mean)
)


# ------------------------------------------------------------
# CARD AVERAGE AMOUNT
# ------------------------------------------------------------

card_mean = (
    train_df
    .groupby("card1")["TransactionAmt"]
    .mean()
)

train_df["amount_vs_card_mean"] = (
    train_df["TransactionAmt"]
    /
    train_df["card1"]
    .map(card_mean)
)

test_df["amount_vs_card_mean"] = (
    test_df["TransactionAmt"]
    /
    test_df["card1"]
    .map(card_mean)
)


# Clean infinite values

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


print(
    "Leakage-safe behavioral features created."
)

# 6. SELECT FINAL FEATURES

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
    f"\nNumber of features: "
    f"{len(features)}"
)

# 7. CATEGORICAL / NUMERICAL FEATURES

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

# 8. PREPROCESSOR FACTORY

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

# 9. CLASS WEIGHT

positive_weight = (
    (y_train == 0).sum()
    /
    (y_train == 1).sum()
)

print(
    f"\nPositive class weight: "
    f"{positive_weight:.2f}"
)

# 10. RESULTS STORAGE

results = []

# 11. LOGISTIC REGRESSION

print("\n")
print("=" * 70)
print("RUNNING: LOGISTIC REGRESSION")
print("=" * 70)

print("Training...")

logistic_model = Pipeline(
    steps=[

        (
            "preprocessor",
            create_preprocessor()
        ),

        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)

logistic_model.fit(
    X_train,
    y_train
)

print("Training complete.")

logistic_prob = (
    logistic_model
    .predict_proba(X_test)[:, 1]
)

logistic_pred = (
    logistic_prob >= 0.5
).astype(int)


logistic_roc = roc_auc_score(
    y_test,
    logistic_prob
)

logistic_pr = average_precision_score(
    y_test,
    logistic_prob
)

logistic_precision = precision_score(
    y_test,
    logistic_pred,
    zero_division=0
)

logistic_recall = recall_score(
    y_test,
    logistic_pred,
    zero_division=0
)

logistic_f1 = f1_score(
    y_test,
    logistic_pred,
    zero_division=0
)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    logistic_pred
).ravel()

results.append({

    "Model": "Logistic Regression",

    "ROC-AUC": logistic_roc,

    "PR-AUC": logistic_pr,

    "Precision": logistic_precision,

    "Recall": logistic_recall,

    "F1": logistic_f1,

    "TP": tp,

    "FP": fp,

    "FN": fn,

    "TN": tn
})

print(
    f"ROC-AUC: {logistic_roc:.4f}"
)

print(
    f"PR-AUC: {logistic_pr:.4f}"
)


del logistic_model
gc.collect()

# 12. HISTOGRAM GRADIENT BOOSTING

print("\n")
print("=" * 70)
print("RUNNING: HISTGRADIENTBOOSTING")
print("=" * 70)

print("Training...")

hist_model = Pipeline(
    steps=[

        (
            "preprocessor",
            create_preprocessor()
        ),

        (
            "model",
            HistGradientBoostingClassifier(

                max_iter=200,

                learning_rate=0.08,

                max_leaf_nodes=31,

                l2_regularization=1.0,

                random_state=42
            )
        )
    ]
)

hist_model.fit(
    X_train,
    y_train
)

print("Training complete.")

hist_prob = (
    hist_model
    .predict_proba(X_test)[:, 1]
)

hist_pred = (
    hist_prob >= 0.5
).astype(int)


hist_roc = roc_auc_score(
    y_test,
    hist_prob
)

hist_pr = average_precision_score(
    y_test,
    hist_prob
)

hist_precision = precision_score(
    y_test,
    hist_pred,
    zero_division=0
)

hist_recall = recall_score(
    y_test,
    hist_pred,
    zero_division=0
)

hist_f1 = f1_score(
    y_test,
    hist_pred,
    zero_division=0
)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    hist_pred
).ravel()

results.append({

    "Model": "HistGradientBoosting",

    "ROC-AUC": hist_roc,

    "PR-AUC": hist_pr,

    "Precision": hist_precision,

    "Recall": hist_recall,

    "F1": hist_f1,

    "TP": tp,

    "FP": fp,

    "FN": fn,

    "TN": tn
})

print(
    f"ROC-AUC: {hist_roc:.4f}"
)

print(
    f"PR-AUC: {hist_pr:.4f}"
)


del hist_model
gc.collect()

# 13. XGBOOST

print("\n")
print("=" * 70)
print("RUNNING: XGBOOST")
print("=" * 70)

print("Training...")

xgb_model = Pipeline(
    steps=[

        (
            "preprocessor",
            create_preprocessor()
        ),

        (
            "model",
            XGBClassifier(

                n_estimators=300,

                max_depth=6,

                learning_rate=0.05,

                subsample=0.8,

                colsample_bytree=0.8,

                objective="binary:logistic",

                eval_metric="aucpr",

                scale_pos_weight=positive_weight,

                tree_method="hist",

                n_jobs=2,

                random_state=42
            )
        )
    ]
)

xgb_model.fit(
    X_train,
    y_train
)

print("Training complete.")

xgb_prob = (
    xgb_model
    .predict_proba(X_test)[:, 1]
)

xgb_pred = (
    xgb_prob >= 0.5
).astype(int)


xgb_roc = roc_auc_score(
    y_test,
    xgb_prob
)

xgb_pr = average_precision_score(
    y_test,
    xgb_prob
)

xgb_precision = precision_score(
    y_test,
    xgb_pred,
    zero_division=0
)

xgb_recall = recall_score(
    y_test,
    xgb_pred,
    zero_division=0
)

xgb_f1 = f1_score(
    y_test,
    xgb_pred,
    zero_division=0
)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    xgb_pred
).ravel()

results.append({

    "Model": "XGBoost",

    "ROC-AUC": xgb_roc,

    "PR-AUC": xgb_pr,

    "Precision": xgb_precision,

    "Recall": xgb_recall,

    "F1": xgb_f1,

    "TP": tp,

    "FP": fp,

    "FN": fn,

    "TN": tn
})

print(
    f"ROC-AUC: {xgb_roc:.4f}"
)

print(
    f"PR-AUC: {xgb_pr:.4f}"
)


del xgb_model
gc.collect()

# 14. FINAL COMPARISON

results_df = pd.DataFrame(
    results
)

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
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# 15. BEST MODEL

best = results_df.iloc[0]

print("\n")
print("=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    f"Model: {best['Model']}"
)

print(
    f"ROC-AUC: {best['ROC-AUC']:.4f}"
)

print(
    f"PR-AUC: {best['PR-AUC']:.4f}"
)

print(
    f"Precision: {best['Precision']:.4f}"
)

print(
    f"Recall: {best['Recall']:.4f}"
)

print(
    f"F1: {best['F1']:.4f}"
)

# 16. SAVE RESULTS

results_df.to_csv(
    "model_comparison_results.csv",
    index=False
)

print(
    "\nResults saved to "
    "'model_comparison_results.csv'"
)

print(
    "\nRiskShield model comparison complete!"
)
