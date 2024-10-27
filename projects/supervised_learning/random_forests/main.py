import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Define column names for the dataset
col_names = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

# Load dataset and assign column names
df = pd.read_csv("../../../assets/csv/adult.data", header=None, names=col_names)

# Display distribution of income (target variable)
print("Income distribution:\n", df.income.value_counts(normalize=True))

# Clean whitespace in categorical columns
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].str.strip()

# Select features and target
feature_cols = ["age", "capital-gain", "capital-loss", "hours-per-week", "sex", "race"]
X = pd.get_dummies(df[feature_cols], drop_first=True)
y = np.where(df.income == "<=50K", 0, 1)

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Instantiate, train, and score a Random Forest classifier with default parameters
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
print(f"Accuracy score (default Random Forest): {rf.score(x_test, y_test) * 100:.2f}%")

# Tune the 'max_depth' hyperparameter over a range from 1 to 25 and record accuracy scores
np.random.seed(0)
accuracy_train = []
accuracy_test = []
depths = range(1, 26)
for depth in depths:
    rf = RandomForestClassifier(max_depth=depth)
    rf.fit(x_train, y_train)
    accuracy_test.append(accuracy_score(y_test, rf.predict(x_test)))
    accuracy_train.append(accuracy_score(y_train, rf.predict(x_train)))

# Determine and display the best depth and accuracy for the test set
best_acc = np.max(accuracy_test)
best_depth = depths[np.argmax(accuracy_test)]
print(f"Optimal max depth: {best_depth}")
print(f"Highest test set accuracy: {best_acc * 100:.2f}%")

# Plot accuracy scores for test and train sets across different depths
plt.plot(depths, accuracy_test, "bo--", label="Test Accuracy")
plt.plot(depths, accuracy_train, "r*:", label="Train Accuracy")
plt.legend()
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.title("Random Forest Accuracy vs. Max Depth")
plt.show()

# Save the best Random Forest model and display top 5 feature importances
best_rf = RandomForestClassifier(max_depth=best_depth)
best_rf.fit(x_train, y_train)
feature_imp_df = pd.DataFrame(
    zip(x_train.columns, best_rf.feature_importances_),
    columns=["Feature", "Importance"],
)
print(
    "Top 5 Important Features:\n",
    feature_imp_df.sort_values("Importance", ascending=False).head(),
)

# Create new binned education feature and redefine feature columns
df["education_bin"] = pd.cut(
    df["education-num"],
    bins=[0, 9, 13, 16],
    labels=["HS or less", "College to Bachelors", "Masters or more"],
)
feature_cols = [
    "age",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "sex",
    "race",
    "education_bin",
]
X = pd.get_dummies(df[feature_cols], drop_first=True)

# Redo train-test split with updated features
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Re-evaluate 'max_depth' tuning with the additional features
accuracy_train = []
accuracy_test = []
depths = range(1, 10)
for depth in depths:
    rf = RandomForestClassifier(max_depth=depth)
    rf.fit(x_train, y_train)
    accuracy_test.append(accuracy_score(y_test, rf.predict(x_test)))
    accuracy_train.append(accuracy_score(y_train, rf.predict(x_train)))

# Display best depth and accuracy after adding new features
best_acc = np.max(accuracy_test)
best_depth = depths[np.argmax(accuracy_test)]
print(f"Optimal max depth with additional features: {best_depth}")
print(f"Highest test set accuracy with additional features: {best_acc * 100:.2f}%")

# Plot accuracy scores for the updated feature set
plt.figure()
plt.plot(depths, accuracy_test, "bo--", label="Test Accuracy")
plt.plot(depths, accuracy_train, "r*:", label="Train Accuracy")
plt.legend()
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.title("Random Forest Accuracy vs. Max Depth (With Additional Features)")
plt.show()

# Display top 5 important features with new feature set
best_rf = RandomForestClassifier(max_depth=best_depth)
best_rf.fit(x_train, y_train)
feature_imp_df = pd.DataFrame(
    zip(x_train.columns, best_rf.feature_importances_),
    columns=["Feature", "Importance"],
)
print(
    "Top 5 Important Features (With Additional Features):\n",
    feature_imp_df.sort_values("Importance", ascending=False).head(),
)
