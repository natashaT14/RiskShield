import pandas as pd

TRANSACTION_PATH = "data/train_transaction.csv"
IDENTITY_PATH = "data/train_identity.csv"

print("Loading transaction data...")
transaction = pd.read_csv(TRANSACTION_PATH)

print("\nLoading identity data...")
identity = pd.read_csv(IDENTITY_PATH)

print("\n========== TRANSACTION DATA ==========")
print("Shape:", transaction.shape)

print("\nColumns:")
print(transaction.columns.tolist())

print("\nData types:")
print(transaction.dtypes)

print("\nMissing values - top 30:")
print(transaction.isnull().sum().sort_values(ascending=False).head(30))

print("\nTarget distribution:")
print(transaction["isFraud"].value_counts())

print("\nTarget percentage:")
print(transaction["isFraud"].value_counts(normalize=True) * 100)

print("\n========== IDENTITY DATA ==========")
print("Shape:", identity.shape)

print("\nColumns:")
print(identity.columns.tolist())

print("\nMissing values - top 30:")
print(identity.isnull().sum().sort_values(ascending=False).head(30))

print("\n========== SAMPLE TRANSACTION ==========")
print(transaction.head(3).to_string())

print("\n========== SAMPLE IDENTITY ==========")
print(identity.head(3).to_string())