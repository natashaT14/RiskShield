import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# CONFIGURATION

RESULTS_PATH = (
    "models/batch_inference_results.csv"
)

OUTPUT_PATH = (
    "models/cost_sensitive_analysis.csv"
)

print("=" * 70)
print("RISKSHEILD COST-SENSITIVE DECISION ANALYSIS")
print("=" * 70)

# 1. LOAD BATCH RESULTS

print("\nLoading batch inference results...")

df = pd.read_csv(
    RESULTS_PATH
)

print(
    f"Transactions loaded: {len(df):,}"
)

# 2. BASIC DATA

y_true = (
    df["ActualFraud"]
    .values
)

probability = (
    df["FraudProbability"]
    .values
)


total_transactions = len(df)

# 3. THRESHOLDS

thresholds = np.arange(
    0.01,
    0.51,
    0.01
)

# 4. COST SCENARIOS

cost_scenarios = {

    "5_to_1": {
        "false_negative": 5,
        "false_positive": 1
    },

    "10_to_1": {
        "false_negative": 10,
        "false_positive": 1
    },

    "20_to_1": {
        "false_negative": 20,
        "false_positive": 1
    }

}

all_results = []

# 5. THRESHOLD ANALYSIS

for scenario_name, costs in cost_scenarios.items():

    fn_cost = costs["false_negative"]

    fp_cost = costs["false_positive"]


    for threshold in thresholds:

        predicted = (
            probability >= threshold
        ).astype(int)


        tn, fp, fn, tp = (
            confusion_matrix(
                y_true,
                predicted
            ).ravel()
        )


        precision = precision_score(
            y_true,
            predicted,
            zero_division=0
        )


        recall = recall_score(
            y_true,
            predicted,
            zero_division=0
        )


        f1 = f1_score(
            y_true,
            predicted,
            zero_division=0
        )


        # ----------------------------------------------------
        # COST
        # ----------------------------------------------------

        total_cost = (

            fn * fn_cost

            +

            fp * fp_cost

        )


        cost_per_transaction = (
            total_cost
            /
            total_transactions
        )


        # ----------------------------------------------------
        # RISK ENGINE DECISIONS
        # ----------------------------------------------------

        approve = (
            probability < threshold
        ).sum()


        review = (
            (probability >= threshold)
            &
            (probability < 0.50)
        ).sum()


        block = (
            probability >= 0.50
        ).sum()


        approval_rate = (
            approve
            /
            total_transactions
        )


        review_rate = (
            review
            /
            total_transactions
        )


        block_rate = (
            block
            /
            total_transactions
        )


        fraud_detected = tp

        fraud_missed = fn


        all_results.append({

            "Scenario":
                scenario_name,

            "Threshold":
                threshold,

            "TP":
                tp,

            "FP":
                fp,

            "FN":
                fn,

            "TN":
                tn,

            "Precision":
                precision,

            "Recall":
                recall,

            "F1":
                f1,

            "FraudDetected":
                fraud_detected,

            "FraudMissed":
                fraud_missed,

            "ApproveRate":
                approval_rate,

            "ReviewRate":
                review_rate,

            "BlockRate":
                block_rate,

            "TotalCost":
                total_cost,

            "CostPerTransaction":
                cost_per_transaction

        })

# 6. RESULTS DATAFRAME

results_df = pd.DataFrame(
    all_results
)

# 7. FIND OPTIMAL THRESHOLDS

print("\n")
print("=" * 70)
print("OPTIMAL THRESHOLD BY COST SCENARIO")
print("=" * 70)


for scenario in cost_scenarios.keys():

    scenario_df = results_df[
        results_df["Scenario"]
        == scenario
    ]


    best = (
        scenario_df
        .sort_values(
            "TotalCost"
        )
        .iloc[0]
    )


    print(
        f"\nScenario: "
        f"FN:FP = "
        f"{cost_scenarios[scenario]['false_negative']}:"
        f"{cost_scenarios[scenario]['false_positive']}"
    )


    print(
        f"Optimal threshold : "
        f"{best['Threshold']:.2f}"
    )


    print(
        f"Total cost        : "
        f"{best['TotalCost']:,.0f}"
    )


    print(
        f"Cost / transaction: "
        f"{best['CostPerTransaction']:.4f}"
    )


    print(
        f"Precision         : "
        f"{best['Precision']:.4f}"
    )


    print(
        f"Recall            : "
        f"{best['Recall']:.4f}"
    )


    print(
        f"F1                : "
        f"{best['F1']:.4f}"
    )


    print(
        f"Fraud detected    : "
        f"{best['FraudDetected']:,}"
    )


    print(
        f"Fraud missed      : "
        f"{best['FraudMissed']:,}"
    )

# 8. CURRENT POLICY

current_threshold = 0.10

print("\n")
print("=" * 70)
print("CURRENT RISKSHEILD POLICY")
print("=" * 70)


current_df = results_df[
    np.isclose(
        results_df["Threshold"],
        current_threshold
    )
]


for scenario in cost_scenarios.keys():

    row = current_df[
        current_df["Scenario"]
        == scenario
    ].iloc[0]


    print(
        f"\n{scenario}"
    )


    print(
        f"Threshold : "
        f"{row['Threshold']:.2f}"
    )


    print(
        f"Precision : "
        f"{row['Precision']:.4f}"
    )


    print(
        f"Recall    : "
        f"{row['Recall']:.4f}"
    )


    print(
        f"F1        : "
        f"{row['F1']:.4f}"
    )


    print(
        f"Total cost: "
        f"{row['TotalCost']:,.0f}"
    )

# 9. SAVE RESULTS

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n")
print("=" * 70)
print("RESULTS SAVED")
print("=" * 70)


print(
    f"\nSaved to:\n"
    f"{OUTPUT_PATH}"
)


print("\n")
print("=" * 70)
print("COST-SENSITIVE ANALYSIS COMPLETE")
print("=" * 70)