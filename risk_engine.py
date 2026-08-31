import pandas as pd

from risk_policy import load_policy

# RISK ENGINE

class RiskEngine:

    def __init__(self):

        self.policy = load_policy()


    # ========================================================
    # EVALUATE TRANSACTION
    # ========================================================

    def evaluate(
        self,
        fraud_probability
    ):

        probability = float(
            fraud_probability
        )


        # ----------------------------------------------------
        # Validate probability
        # ----------------------------------------------------

        if not 0 <= probability <= 1:

            raise ValueError(
                "Fraud probability must be "
                "between 0 and 1."
            )


        # ----------------------------------------------------
        # Risk score
        # ----------------------------------------------------

        risk_score = (
            probability * 100
        )


        # ----------------------------------------------------
        # Policy boundaries
        # ----------------------------------------------------

        low_upper = (
            self.policy["low_upper"]
        )

        medium_upper = (
            self.policy["medium_upper"]
        )


        # ----------------------------------------------------
        # Decision
        # ----------------------------------------------------

        if probability < low_upper:

            risk_level = "LOW"

            decision = "APPROVE"

            reason = (
                "LOW_FRAUD_PROBABILITY"
            )


        elif probability < medium_upper:

            risk_level = "MEDIUM"

            decision = "REVIEW"

            reason = (
                "ELEVATED_FRAUD_PROBABILITY"
            )


        else:

            risk_level = "HIGH"

            decision = "BLOCK"

            reason = (
                "HIGH_FRAUD_PROBABILITY"
            )


        # ----------------------------------------------------
        # Return complete risk assessment
        # ----------------------------------------------------

        return {

            "fraud_probability":
                round(
                    probability,
                    6
                ),

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

# TEST

if __name__ == "__main__":

    print("=" * 70)
    print("RISKSHEILD RISK ENGINE")
    print("=" * 70)


    engine = RiskEngine()


    print("\nLoaded policy:")

    print(
        f"Operating threshold: "
        f"{engine.policy['threshold']:.2f}"
    )

    print(
        f"Low upper boundary: "
        f"{engine.policy['low_upper']:.2f}"
    )

    print(
        f"Medium upper boundary: "
        f"{engine.policy['medium_upper']:.2f}"
    )


    print("\n")
    print("=" * 70)
    print("RISK ENGINE TEST")
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

        result = engine.evaluate(
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
    print("RISKSHEILD RISK ENGINE TEST COMPLETE")
    print("=" * 70)