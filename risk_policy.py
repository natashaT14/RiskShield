import os
import pandas as pd

# CONFIGURATION

POLICY_PATH = "models/risk_policy.csv"

# DEFAULT POLICY

DEFAULT_POLICY = {
    "model": "HistGradientBoosting",
    "threshold": 0.10,

    # Cost assumptions are configurable.
    # They are NOT used to calculate the ML probability.
    # They are used for decision-policy analysis.
    "false_negative_cost": 10,
    "false_positive_cost": 1,

    # Risk-engine decision bands
    "low_upper": 0.10,
    "medium_upper": 0.50
}

# CREATE POLICY FILE IF IT DOES NOT EXIST

def create_default_policy():

    os.makedirs(
        "models",
        exist_ok=True
    )

    policy_df = pd.DataFrame([

        {
            "Model":
                DEFAULT_POLICY["model"],

            "Threshold":
                DEFAULT_POLICY["threshold"],

            "FN_Cost":
                DEFAULT_POLICY[
                    "false_negative_cost"
                ],

            "FP_Cost":
                DEFAULT_POLICY[
                    "false_positive_cost"
                ],

            "Low_Upper":
                DEFAULT_POLICY["low_upper"],

            "Medium_Upper":
                DEFAULT_POLICY["medium_upper"]
        }

    ])

    policy_df.to_csv(
        POLICY_PATH,
        index=False
    )

    print(
        f"Default risk policy created:\n"
        f"{POLICY_PATH}"
    )

# LOAD POLICY

def load_policy():

    if not os.path.exists(
        POLICY_PATH
    ):

        create_default_policy()


    policy_df = pd.read_csv(
        POLICY_PATH
    )


    if policy_df.empty:

        raise ValueError(
            "Risk policy file is empty."
        )


    row = policy_df.iloc[0]


    policy = {

        "model":
            str(row["Model"]),

        "threshold":
            float(row["Threshold"]),

        "false_negative_cost":
            float(row["FN_Cost"]),

        "false_positive_cost":
            float(row["FP_Cost"]),

        "low_upper":
            float(row["Low_Upper"]),

        "medium_upper":
            float(row["Medium_Upper"])
    }


    # ========================================================
    # VALIDATION
    # ========================================================

    threshold = policy["threshold"]

    low_upper = policy["low_upper"]

    medium_upper = policy["medium_upper"]


    if not 0 < threshold < 1:

        raise ValueError(
            "Operating threshold must be between 0 and 1."
        )


    if not 0 < low_upper < 1:

        raise ValueError(
            "Low risk boundary must be between 0 and 1."
        )


    if not 0 < medium_upper <= 1:

        raise ValueError(
            "Medium risk boundary must be between 0 and 1."
        )


    if low_upper >= medium_upper:

        raise ValueError(
            "Low risk boundary must be "
            "less than Medium risk boundary."
        )


    if (
        policy["false_negative_cost"] <= 0
        or
        policy["false_positive_cost"] <= 0
    ):

        raise ValueError(
            "Risk costs must be positive."
        )


    return policy

# RISK DECISION

def evaluate_risk(
    fraud_probability
):

    policy = load_policy()


    probability = float(
        fraud_probability
    )


    # --------------------------------------------------------
    # Validate probability
    # --------------------------------------------------------

    if not 0 <= probability <= 1:

        raise ValueError(
            "Fraud probability must be between 0 and 1."
        )


    # --------------------------------------------------------
    # Risk score
    # --------------------------------------------------------

    risk_score = (
        probability * 100
    )


    # --------------------------------------------------------
    # Risk level + decision
    # --------------------------------------------------------

    if probability < policy["low_upper"]:

        risk_level = "LOW"

        decision = "APPROVE"

        reason = "LOW_FRAUD_PROBABILITY"


    elif probability < policy["medium_upper"]:

        risk_level = "MEDIUM"

        decision = "REVIEW"

        reason = "ELEVATED_FRAUD_PROBABILITY"


    else:

        risk_level = "HIGH"

        decision = "BLOCK"

        reason = "HIGH_FRAUD_PROBABILITY"


    return {

        "fraud_probability":
            probability,

        "risk_score":
            round(
                risk_score,
                2
            ),

        "risk_level":
            risk_level,

        "decision":
            decision,

        "reason":
            reason
    }

# MAIN TEST

if __name__ == "__main__":

    print("=" * 70)
    print("RISKSHEILD RISK POLICY")
    print("=" * 70)


    policy = load_policy()


    print("\nLoaded policy:")

    for key, value in policy.items():

        print(
            f"{key}: {value}"
        )


    print("\n")
    print("=" * 70)
    print("POLICY DECISION TEST")
    print("=" * 70)


    test_probabilities = [

        0.02,
        0.08,
        0.10,
        0.18,
        0.35,
        0.50,
        0.73,
        0.95

    ]


    for probability in test_probabilities:

        result = evaluate_risk(
            probability
        )


        print(
            f"\nProbability : "
            f"{probability * 100:.2f}%"
        )

        print(
            f"Risk Score  : "
            f"{result['risk_score']:.2f}"
        )

        print(
            f"Risk Level  : "
            f"{result['risk_level']}"
        )

        print(
            f"Decision    : "
            f"{result['decision']}"
        )

        print(
            f"Reason      : "
            f"{result['reason']}"
        )


    print("\n")
    print("=" * 70)
    print("RISK POLICY TEST COMPLETE")
    print("=" * 70)