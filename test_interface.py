import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from predict import (
    predict_transaction,
    RAW_FEATURES
)

# CONFIGURATION

DATA_PATH = "data/train_transaction.csv"

NUM_SAMPLES = 5

# LOAD TRANSACTIONS

print("=" * 70)
print("RISKSHEILD REAL TRANSACTION VALIDATION")
print("=" * 70)

print("\nLoading sample transactions...")


required_columns = RAW_FEATURES + ["isFraud"]


df = pd.read_csv(
    DATA_PATH,
    usecols=required_columns
)

# SELECT LEGITIMATE + FRAUD TRANSACTIONS

legitimate = (
    df[df["isFraud"] == 0]
    .head(NUM_SAMPLES)
)


fraud = (
    df[df["isFraud"] == 1]
    .head(NUM_SAMPLES)
)


test_df = pd.concat(
    [
        legitimate,
        fraud
    ],
    ignore_index=True
)


print(
    f"\nTesting {len(legitimate)} legitimate "
    f"+ {len(fraud)} fraudulent transactions."
)

# RUN RISKSHEILD

results = []


print("\nRunning RiskShield inference...")


for index, row in test_df.iterrows():

    actual_label = int(
        row["isFraud"]
    )


    transaction = (
        row[RAW_FEATURES]
        .to_dict()
    )


    result = predict_transaction(
        transaction
    )


    predicted_label = (

        1
        if result["decision"] != "APPROVE"
        else 0

    )


    correct = (
        actual_label == predicted_label
    )


    results.append({

        "Actual": (
            "FRAUD"
            if actual_label == 1
            else "LEGIT"
        ),

        "Probability":
            result["fraud_probability"],

        "Risk Score":
            result["risk_score"],

        "Risk Level":
            result["risk_level"],

        "Decision":
            result["decision"],

        "Prediction":
            (
                "FRAUD"
                if predicted_label == 1
                else "LEGIT"
            ),

        "Correct":
            "YES"
            if correct
            else "NO"

    })

# DISPLAY RESULTS

results_df = pd.DataFrame(
    results
)


print("\n")
print("=" * 70)
print("RISKSHEILD VALIDATION RESULTS")
print("=" * 70)


print(
    results_df.to_string(
        index=False,
        formatters={
            "Probability":
                lambda x: f"{x:.2%}",

            "Risk Score":
                lambda x: f"{x:.2f}"
        }
    )
)

# SUMMARY

correct_count = (
    results_df["Correct"] == "YES"
).sum()


total_count = len(
    results_df
)


accuracy = (
    correct_count
    /
    total_count
)


print("\n")
print("=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)


print(
    f"\nCorrect predictions : "
    f"{correct_count}/{total_count}"
)


print(
    f"Validation accuracy : "
    f"{accuracy:.2%}"
)

# SEPARATE FRAUD PERFORMANCE

fraud_results = results_df[
    results_df["Actual"] == "FRAUD"
]


fraud_detected = (
    fraud_results["Prediction"] == "FRAUD"
).sum()


print(
    f"\nFraud detected      : "
    f"{fraud_detected}/{len(fraud_results)}"
)

# SEPARATE LEGITIMATE PERFORMANCE

legit_results = results_df[
    results_df["Actual"] == "LEGIT"
]


legit_approved = (
    legit_results["Prediction"] == "LEGIT"
).sum()


print(
    f"Legitimate approved: "
    f"{legit_approved}/{len(legit_results)}"
)


print("\n")
print("=" * 70)
print(
    "RISKSHEILD REAL TRANSACTION VALIDATION COMPLETE"
)
print("=" * 70)