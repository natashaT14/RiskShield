from risk_engine import RiskEngine

# TEST RISK ENGINE

engine = RiskEngine(
    approve_threshold=0.10,
    block_threshold=0.50
)


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


print("=" * 70)
print("RISKSHEILD RISK ENGINE TEST")
print("=" * 70)


for probability in test_probabilities:

    result = engine.assess(
        probability
    )

    print(
        f"\nProbability: "
        f"{probability:.2%}"
    )

    print(
        f"Risk Score: "
        f"{result['risk_score']}"
    )

    print(
        f"Risk Level: "
        f"{result['risk_level']}"
    )

    print(
        f"Decision: "
        f"{result['decision']}"
    )

    print(
        f"Reason: "
        f"{result['reason']}"
    )