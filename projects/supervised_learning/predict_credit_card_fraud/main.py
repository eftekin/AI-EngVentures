import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
transactions = pd.read_csv("transactions_modified.csv")
print(transactions.head())
transactions = transactions.dropna()
print(transactions.info())

# How many fraudulent transactions?
print(transactions["isFraud"].info())
fraudulent_transactions = transactions["isFraud"].sum()
print("fraudulent transactions:", fraudulent_transactions)

# Plot fraudulent vs non-fraudulent transactions
plt.figure(figsize=(6, 4))
seaborn.countplot(x="isFraud", data=transactions)
plt.title("Fraudulent vs Non-Fraudulent Transactions")
plt.xlabel("Is Fraud")
plt.ylabel("Count")
plt.show()

# Summary statistics on amount column
print(transactions["amount"].describe())

# Plot the distribution of transaction amounts
plt.figure(figsize=(8, 6))
seaborn.histplot(transactions["amount"], bins=50, kde=True)
plt.title("Distribution of Transaction Amounts")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.show()

# Create isPayment field
transactions["isPayment"] = 0
transactions.loc[transactions["type"].isin(["PAYMENT", "DEBIT"]), "isPayment"] = 1


# Create isMovement field
transactions["isMovement"] = 0
transactions.loc[transactions["type"].isin(["CASH_OUT", "TRANSFER"]), "isMovement"] = 1


# Create accountDiff field
transactions["accountDiff"] = abs(
    transactions["oldbalanceDest"] - transactions["oldbalanceOrg"]
)

# Create features and label variables
features = transactions[["amount", "isPayment", "isMovement", "accountDiff"]]
label = transactions["isFraud"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features, label, train_size=0.7)

mask = ~y_train.isna()
X_train = X_train[mask]
y_train = y_train[mask]


# Normalize the features variables
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit the model to the training data
model = LogisticRegression()
model.fit(X_train, y_train)

# Score the model on the training data
print("model score on training data:", model.score(X_train, y_train))

# Score the model on the test data
print("model score on test data:", model.score(X_test, y_test))

# Print the model coefficients
print("model coefficients:", model.coef_)

# New transaction data
transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])

your_transaction = np.array([2023.3, 1.0, 0.0, 2021.2])


# Combine new transactions into a single array
sample_transactions = np.stack(
    (transaction1, transaction2, transaction3, your_transaction)
)

# Convert the new transactions array into a DataFrame to preserve feature names for scaling
sample_transactions = pd.DataFrame(sample_transactions, columns=features.columns)

# Normalize the new transactions
sample_transactions = scaler.transform(sample_transactions)

# Predict fraud on the new transactions
print("predicted fraud on the new transactions:", model.predict(sample_transactions))

# Show probabilities on the new transactions
print(
    "probabilities on the new transactions:\n", model.predict_proba(sample_transactions)
)
