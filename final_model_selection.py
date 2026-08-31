import warnings
warnings.filterwarnings("ignore")

import gc
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from feature_engineering import (
    FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    create_basic_features,
    create_behavioral_mappings,
    apply_behavioral_features
)

# CONFIGURATION

DATA_PATH = "data/train_transaction.csv"

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "riskshield_histgradientboosting.pkl"
)

METRICS_PATH = os.path.join(
    MODEL_DIR,
    "final_model_metrics.csv"
)

THRESHOLD_PATH = os.path.join(
    MODEL_DIR,
    "riskshield_threshold.csv"
)

MAPPINGS_PATH = os.path.join(
    MODEL_DIR,
    "behavioral_mappings.pkl"
)

RANDOM_STATE = 42


print("=" * 70)
print("RISKSHIELD FINAL MODEL SELECTION")
print("=" * 70)

# 1. LOAD DATA

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


print(
    f"Dataset loaded: {df.shape}"
)

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

# 3. BASIC FEATURE ENGINEERING

print("\nCreating basic features...")


df = create_basic_features(df)

# 4. TEMPORAL TRAIN / TEST SPLIT

print("\nCreating temporal train/test split...")


split_index = int(
    len(df) * 0.80
)


train_df = (
    df.iloc[:split_index]
    .copy()
)


test_df = (
    df.iloc[split_index:]
    .copy()
)


print(
    f"Training rows: {len(train_df):,}"
)

print(
    f"Testing rows:  {len(test_df):,}"
)


print(
    f"\nTraining time:"
    f" {train_df['TransactionDT'].min()}"
    f" → {train_df['TransactionDT'].max()}"
)

print(
    f"Testing time:"
    f" {test_df['TransactionDT'].min()}"
    f" → {test_df['TransactionDT'].max()}"
)

# 5. CREATE LEAKAGE-SAFE BEHAVIORAL MAPPINGS

print("\nCreating leakage-safe behavioral mappings...")


behavioral_mappings = (
    create_behavioral_mappings(
        train_df
    )
)

# 6. APPLY BEHAVIORAL FEATURES

print("\nApplying behavioral features...")


train_df = apply_behavioral_features(
    train_df,
    behavioral_mappings
)


test_df = apply_behavioral_features(
    test_df,
    behavioral_mappings
)


print(
    "Behavioral features applied."
)

# 7. PREPARE FINAL FEATURES

X_train = train_df[FEATURES].copy()

X_test = test_df[FEATURES].copy()

y_train = train_df["isFraud"].copy()

y_test = test_df["isFraud"].copy()


print(
    f"\nFinal feature count: {len(FEATURES)}"
)

# 8. VERIFY FEATURE SET

print("\nFeature verification:")

for index, feature in enumerate(
    FEATURES,
    start=1
):

    print(
        f"{index:02d}. {feature}"
    )



# 9. CREATE PREPROCESSOR


print(
    "\nCreating preprocessing pipeline..."
)


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


preprocessor = ColumnTransformer(
    transformers=[

        (
            "num",

            numeric_transformer,

            NUMERICAL_FEATURES
        ),

        (
            "cat",

            categorical_transformer,

            CATEGORICAL_FEATURES
        )

    ]
)

# 10. FINAL HISTGRADIENTBOOSTING MODEL

print("\n")
print("=" * 70)
print("TRAINING FINAL HISTGRADIENTBOOSTING MODEL")
print("=" * 70)


final_model = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",

            HistGradientBoostingClassifier(

                max_iter=200,

                learning_rate=0.08,

                max_leaf_nodes=31,

                l2_regularization=1.0,

                random_state=RANDOM_STATE

            )

        )

    ]
)


print("Training...")


final_model.fit(
    X_train,
    y_train
)


print("Training complete.")

# 11. GENERATE PROBABILITIES

print("\nGenerating fraud probabilities...")


probabilities = (
    final_model
    .predict_proba(X_test)[:, 1]
)

# 12. MODEL PERFORMANCE

roc_auc = roc_auc_score(
    y_test,
    probabilities
)


pr_auc = average_precision_score(
    y_test,
    probabilities
)


print("\n")
print("=" * 70)
print("MODEL PROBABILITY PERFORMANCE")
print("=" * 70)


print(
    f"ROC-AUC: {roc_auc:.4f}"
)

print(
    f"PR-AUC : {pr_auc:.4f}"
)

# 13. THRESHOLD ANALYSIS

print("\n")
print("=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)


thresholds = np.arange(
    0.05,
    0.51,
    0.05
)


threshold_results = []


for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)


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


    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions
    ).ravel()


    threshold_results.append({

        "Threshold": threshold,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "TP": tp,

        "FP": fp,

        "FN": fn,

        "TN": tn

    })


threshold_df = pd.DataFrame(
    threshold_results
)


print(
    threshold_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# 14. SELECT OPERATING THRESHOLD

best_threshold_row = (
    threshold_df
    .sort_values(
        "F1",
        ascending=False
    )
    .iloc[0]
)


best_threshold = (
    best_threshold_row["Threshold"]
)


print("\n")
print("=" * 70)
print("SELECTED OPERATING THRESHOLD")
print("=" * 70)


print(
    f"Threshold : {best_threshold:.2f}"
)

print(
    f"Precision : "
    f"{best_threshold_row['Precision']:.4f}"
)

print(
    f"Recall    : "
    f"{best_threshold_row['Recall']:.4f}"
)

print(
    f"F1        : "
    f"{best_threshold_row['F1']:.4f}"
)

# 15. FINAL PREDICTIONS

final_predictions = (
    probabilities >= best_threshold
).astype(int)


tn, fp, fn, tp = confusion_matrix(
    y_test,
    final_predictions
).ravel()


print("\n")
print("=" * 70)
print("FINAL CONFUSION MATRIX")
print("=" * 70)


print(
    f"True Positives : {tp:,}"
)

print(
    f"False Positives: {fp:,}"
)

print(
    f"False Negatives: {fn:,}"
)

print(
    f"True Negatives : {tn:,}"
)

# 16. CREATE MODEL DIRECTORY

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

# 17. SAVE MODEL

print("\nSaving final model...")


joblib.dump(
    final_model,
    MODEL_PATH
)


print(
    f"Model saved to:"
    f"\n{MODEL_PATH}"
)

# 18. SAVE BEHAVIORAL MAPPINGS

print("\nSaving behavioral mappings...")


joblib.dump(
    behavioral_mappings,
    MAPPINGS_PATH
)


print(
    f"Behavioral mappings saved to:"
    f"\n{MAPPINGS_PATH}"
)

# 19. SAVE THRESHOLD

threshold_config = pd.DataFrame({

    "Model": [
        "HistGradientBoosting"
    ],

    "Threshold": [
        best_threshold
    ],

    "PR_AUC": [
        pr_auc
    ],

    "ROC_AUC": [
        roc_auc
    ],

    "Precision": [
        best_threshold_row["Precision"]
    ],

    "Recall": [
        best_threshold_row["Recall"]
    ],

    "F1": [
        best_threshold_row["F1"]
    ]

})


threshold_config.to_csv(
    THRESHOLD_PATH,
    index=False
)


print(
    f"Threshold configuration saved to:"
    f"\n{THRESHOLD_PATH}"
)

# 20. SAVE METRICS

metrics = pd.DataFrame({

    "Model": [
        "HistGradientBoosting"
    ],

    "Features": [
        len(FEATURES)
    ],

    "ROC_AUC": [
        roc_auc
    ],

    "PR_AUC": [
        pr_auc
    ],

    "Threshold": [
        best_threshold
    ],

    "Precision": [
        best_threshold_row["Precision"]
    ],

    "Recall": [
        best_threshold_row["Recall"]
    ],

    "F1": [
        best_threshold_row["F1"]
    ],

    "TP": [
        tp
    ],

    "FP": [
        fp
    ],

    "FN": [
        fn
    ],

    "TN": [
        tn
    ]

})


metrics.to_csv(
    METRICS_PATH,
    index=False
)


print(
    f"Final metrics saved to:"
    f"\n{METRICS_PATH}"
)

# 21. CLEANUP

del df
del train_df
del test_df
del X_train
del X_test
del y_train
del y_test

gc.collect()

# 22. COMPLETE

print("\n")
print("=" * 70)
print("RISKSHEILD FINAL MODEL SELECTION COMPLETE")
print("=" * 70)


print(
    "\nFinal model:"
    " HistGradientBoosting"
)

print(
    f"Final features:"
    f" {len(FEATURES)}"
)

print(
    f"Final PR-AUC:"
    f" {pr_auc:.4f}"
)

print(
    f"Final threshold:"
    f" {best_threshold:.2f}"
)

print(
    "\nSaved artifacts:"
)

print(
    f"  Model     : {MODEL_PATH}"
)

print(
    f"  Mappings  : {MAPPINGS_PATH}"
)

print(
    f"  Threshold : {THRESHOLD_PATH}"
)

print(
    f"  Metrics   : {METRICS_PATH}"
)

print(
    "\nRiskShield is ready for inference."
)