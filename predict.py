import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import pandas as pd

from feature_engineering import (
    create_basic_features,
    apply_behavioral_features,
    FEATURES
)

from risk_engine import RiskEngine

# CONFIGURATION

MODEL_PATH = (
    "models/riskshield_histgradientboosting.pkl"
)

MAPPINGS_PATH = (
    "models/behavioral_mappings.pkl"
)

THRESHOLD_PATH = (
    "models/riskshield_threshold.csv"
)

# REQUIRED RAW FEATURES
RAW_FEATURES = [

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
    "dist2"
]

# LOAD MODEL

def load_model():

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(
        MODEL_PATH
    )

# LOAD BEHAVIORAL MAPPINGS

def load_mappings():

    if not os.path.exists(MAPPINGS_PATH):

        raise FileNotFoundError(
            f"Behavioral mappings not found: "
            f"{MAPPINGS_PATH}"
        )

    return joblib.load(
        MAPPINGS_PATH
    )

# LOAD THRESHOLD

def load_threshold():

    if not os.path.exists(THRESHOLD_PATH):

        raise FileNotFoundError(
            f"Threshold configuration not found: "
            f"{THRESHOLD_PATH}"
        )

    threshold_df = pd.read_csv(
        THRESHOLD_PATH
    )

    return float(
        threshold_df.loc[
            0,
            "Threshold"
        ]
    )

# VALIDATE INPUT

def validate_transaction(
    transaction
):

    missing_features = [

        feature

        for feature in RAW_FEATURES

        if feature not in transaction
    ]

    if missing_features:

        raise ValueError(
            "Missing required features: "
            + ", ".join(missing_features)
        )

# PREPARE FEATURES

def prepare_features(
    transaction,
    mappings
):

    # --------------------------------------------------------
    # Convert transaction to DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(
        [transaction]
    )


    # --------------------------------------------------------
    # Validate raw features
    # --------------------------------------------------------

    validate_transaction(
        transaction
    )


    # --------------------------------------------------------
    # Basic features
    # --------------------------------------------------------

    df = create_basic_features(
        df
    )


    # --------------------------------------------------------
    # Behavioral features
    # --------------------------------------------------------

    df = apply_behavioral_features(
        df,
        mappings
    )


    # --------------------------------------------------------
    # Final 26 features
    # --------------------------------------------------------

    return df[
        FEATURES
    ]

# PREDICT TRANSACTION


def predict_transaction(
    transaction
):

    print(
        "\nPreparing RiskShield inference..."
    )


    # --------------------------------------------------------
    # Load artifacts
    # --------------------------------------------------------

    model = load_model()

    mappings = load_mappings()

    threshold = load_threshold()


    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    X = prepare_features(
        transaction,
        mappings
    )


    # --------------------------------------------------------
    # Generate probability
    # --------------------------------------------------------

    fraud_probability = (

        model
        .predict_proba(X)[:, 1][0]

    )


    # --------------------------------------------------------
    # Risk Engine
    # --------------------------------------------------------

    risk_engine = RiskEngine(

        approve_threshold=threshold,

        block_threshold=0.50

    )


    result = risk_engine.assess(
        fraud_probability
    )


    # --------------------------------------------------------
    # Add transaction information
    # --------------------------------------------------------

    result["transaction_amount"] = (
        float(
            transaction["TransactionAmt"]
        )
    )


    return result

# DEMO

if __name__ == "__main__":

    print("=" * 70)
    print("RISKSHEILD TRANSACTION INFERENCE")
    print("=" * 70)


    # --------------------------------------------------------
    # Load one real transaction from dataset
    # --------------------------------------------------------

    print(
        "\nLoading sample transaction..."
    )


    sample_columns = RAW_FEATURES


    sample = pd.read_csv(

        "data/train_transaction.csv",

        usecols=sample_columns,

        skiprows=lambda x: x not in [0, 1]

    )


    transaction = (
        sample.iloc[0]
        .to_dict()
    )


    print(
        "\nTransaction loaded."
    )


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    result = predict_transaction(
        transaction
    )


    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RISKSHEILD RISK ASSESSMENT")
    print("=" * 70)


    print(
        f"\nTransaction Amount : "
        f"{result['transaction_amount']:.2f}"
    )

    print(
        f"Fraud Probability  : "
        f"{result['fraud_probability']:.2%}"
    )

    print(
        f"Risk Score         : "
        f"{result['risk_score']:.2f}"
    )

    print(
        f"Risk Level         : "
        f"{result['risk_level']}"
    )

    print(
        f"Decision            : "
        f"{result['decision']}"
    )

    print(
        f"Reason              : "
        f"{result['reason']}"
    )


    print("\n")
    print("=" * 70)
    print(
        "RISKSHEILD INFERENCE COMPLETE"
    )
    print("=" * 70)