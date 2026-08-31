import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

# 1. LOAD DATA

print("=" * 70)
print("RISKSHIELD CLEANING EXPERIMENT")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(
    "data/train_transaction.csv",
    nrows=100_000
)

print("Dataset loaded:", df.shape)

# 2. SORT CHRONOLOGICALLY

print("\nSorting transactions chronologically...")

df = df.sort_values("TransactionDT").reset_index(drop=True)

print("Transactions sorted.")

# 3. BASIC FEATURES

print("\nCreating basic features...")

df["log_amount"] = np.log1p(df["TransactionAmt"])

df["transaction_day"] = (
    df["TransactionDT"] // (24 * 60 * 60)
)

df["transaction_time"] = (
    df["TransactionDT"] % (24 * 60 * 60)
)

df["day_of_week"] = (
    df["transaction_day"] % 7
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

print("Basic features created.")

# 4. TEMPORAL SPLIT

print("\nCreating temporal train/test split...")

split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

print(
    "Training transactions:",
    len(train_df)
)

print(
    "Testing transactions:",
    len(test_df)
)

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

# 5. TARGET

y_train = train_df["isFraud"]
y_test = test_df["isFraud"]

# 6. DETERMINE CLEANING CANDIDATES

print("\nDetermining cleaning candidates...")

# IMPORTANT:
# We calculate these statistics using TRAINING DATA ONLY.

train_missing_percentage = (
    train_df.isna().mean() * 100
)

train_unique_counts = (
    train_df.nunique(dropna=False)
)

train_near_constant = []

for column in train_df.columns:

    if column == "isFraud":
        continue

    frequencies = (
        train_df[column]
        .value_counts(
            normalize=True,
            dropna=False
        )
    )

    if len(frequencies) > 0:

        if frequencies.iloc[0] >= 0.99:

            train_near_constant.append(column)


constant_features = [
    column
    for column in train_df.columns
    if train_unique_counts[column] == 1
    and column != "isFraud"
]

high_missing_features = [
    column
    for column in train_df.columns
    if train_missing_percentage[column] > 90
    and column != "isFraud"
]


print(
    "\nConstant features:",
    constant_features
)

print(
    "\nNear-constant features:",
    train_near_constant
)

print(
    "\nHigh-missing features (>90%):",
    high_missing_features
)

# 7. FEATURE GROUPS

base_features = [
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
    "day_of_week"
]


# Make sure all features exist.

base_features = [
    feature
    for feature in base_features
    if feature in train_df.columns
]

# 8. EXPERIMENT DEFINITIONS

experiments = {}


# ------------------------------------------------------------
# A. BASELINE
# ------------------------------------------------------------

experiments["A_Baseline"] = base_features.copy()


# ------------------------------------------------------------
# B. REMOVE CONSTANT FEATURES
# ------------------------------------------------------------

experiments["B_Remove_Constant"] = [
    feature
    for feature in base_features
    if feature not in constant_features
]


# ------------------------------------------------------------
# C. REMOVE NEAR-CONSTANT FEATURES
# ------------------------------------------------------------

experiments["C_Remove_Near_Constant"] = [
    feature
    for feature in base_features
    if feature not in train_near_constant
]


# ------------------------------------------------------------
# D. REMOVE HIGH-MISSING FEATURES
# ------------------------------------------------------------

experiments["D_Remove_High_Missing"] = [
    feature
    for feature in base_features
    if feature not in high_missing_features
]


# ------------------------------------------------------------
# E. REMOVE CONSTANT + NEAR-CONSTANT
# ------------------------------------------------------------

experiments["E_Aggressive_Cleaning"] = [
    feature
    for feature in base_features
    if (
        feature not in constant_features
        and feature not in train_near_constant
        and feature not in high_missing_features
    )
]

# 9. TRAINING FUNCTION

def run_experiment(
    name,
    features
):

    print("\n")
    print("=" * 70)
    print("Running:", name)
    print("Number of features:", len(features))
    print("=" * 70)

    X_train = train_df[features]
    X_test = test_df[features]

    categorical_features = (
        X_train
        .select_dtypes(
            include=["object"]
        )
        .columns
        .tolist()
    )

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
        ],
        sparse_threshold=0
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

    print("Training complete.")

    print("Generating predictions...")

    probabilities = (
        pipeline.predict_proba(
            X_test
        )[:, 1]
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
        "\nROC-AUC:",
        round(roc_auc, 6)
    )

    print(
        "PR-AUC:",
        round(pr_auc, 6)
    )

    return {
        "Model": name,
        "Features": len(features),
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc
    }

# 10. RUN EXPERIMENTS

results = []

for name, features in experiments.items():

    result = run_experiment(
        name,
        features
    )

    results.append(result)

# 11. RESULTS

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "PR-AUC",
    ascending=False
)

print("\n")
print("=" * 70)
print("FINAL CLEANING EXPERIMENT RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

# 12. INTERPRETATION

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

best_model = results_df.iloc[0]

print(
    f"\nBest model by PR-AUC: "
    f"{best_model['Model']}"
)

print(
    f"PR-AUC: "
    f"{best_model['PR-AUC']:.6f}"
)

print(
    "\nRemember:"
)

print(
    "- Higher PR-AUC is generally better for our fraud problem."
)

print(
    "- ROC-AUC tells us overall ranking ability."
)

print(
    "- PR-AUC is especially important because fraud is rare."
)

print(
    "- We are using the SAME temporal test period for every experiment."
)

print(
    "- Therefore, the comparison is fair."
)