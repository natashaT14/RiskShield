import numpy as np
import pandas as pd

# RISKSHIELD FEATURE ENGINEERING

FEATURES = [

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


CATEGORICAL_FEATURES = [

    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain"
]


NUMERICAL_FEATURES = [

    feature

    for feature in FEATURES

    if feature not in CATEGORICAL_FEATURES
]

# BASIC FEATURES

def create_basic_features(df):

    df = df.copy()

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

    return df

# CREATE BEHAVIORAL MAPPINGS

def create_behavioral_mappings(train_df):

    mappings = {}

    # --------------------------------------------------------
    # CARD FREQUENCY
    # --------------------------------------------------------

    mappings["card_counts"] = (
        train_df["card1"]
        .value_counts()
        .to_dict()
    )

    # --------------------------------------------------------
    # ADDRESS FREQUENCY
    # --------------------------------------------------------

    mappings["address_counts"] = (
        train_df["addr1"]
        .value_counts()
        .to_dict()
    )

    # --------------------------------------------------------
    # EMAIL FREQUENCY
    # --------------------------------------------------------

    mappings["email_counts"] = (
        train_df["P_emaildomain"]
        .value_counts()
        .to_dict()
    )

    # --------------------------------------------------------
    # PRODUCT MEAN
    # --------------------------------------------------------

    mappings["product_mean"] = (
        train_df
        .groupby("ProductCD")["TransactionAmt"]
        .mean()
        .to_dict()
    )

    # --------------------------------------------------------
    # CARD MEAN
    # --------------------------------------------------------

    mappings["card_mean"] = (
        train_df
        .groupby("card1")["TransactionAmt"]
        .mean()
        .to_dict()
    )

    return mappings

# APPLY BEHAVIORAL FEATURES

def apply_behavioral_features(
    df,
    mappings
):

    df = df.copy()

    # --------------------------------------------------------
    # CARD FREQUENCY
    # --------------------------------------------------------

    df["card_frequency"] = (
        df["card1"]
        .map(mappings["card_counts"])
        .fillna(0)
    )

    # --------------------------------------------------------
    # ADDRESS FREQUENCY
    # --------------------------------------------------------

    df["address_frequency"] = (
        df["addr1"]
        .map(mappings["address_counts"])
        .fillna(0)
    )

    # --------------------------------------------------------
    # EMAIL FREQUENCY
    # --------------------------------------------------------

    df["email_frequency"] = (
        df["P_emaildomain"]
        .map(mappings["email_counts"])
        .fillna(0)
    )

    # --------------------------------------------------------
    # AMOUNT VS PRODUCT MEAN
    # --------------------------------------------------------

    df["amount_vs_product_mean"] = (

        df["TransactionAmt"]

        /

        df["ProductCD"]
        .map(mappings["product_mean"])

    )

    # --------------------------------------------------------
    # AMOUNT VS CARD MEAN
    # --------------------------------------------------------

    df["amount_vs_card_mean"] = (

        df["TransactionAmt"]

        /

        df["card1"]
        .map(mappings["card_mean"])

    )

    # --------------------------------------------------------
    # CLEAN RATIOS
    # --------------------------------------------------------

    ratio_features = [

        "amount_vs_product_mean",
        "amount_vs_card_mean"

    ]

    for column in ratio_features:

        df[column] = (
            df[column]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(1)
        )

    return df



# COMPLETE FEATURE ENGINEERING


def build_features(
    df,
    mappings
):

    df = create_basic_features(df)

    df = apply_behavioral_features(
        df,
        mappings
    )

    return df[FEATURES].copy()