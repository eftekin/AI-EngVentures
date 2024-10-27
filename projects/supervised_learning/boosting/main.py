import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load the dataset from the UCI Machine Learning Repository
path_to_data = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)

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

# Read the dataset into a DataFrame
df = pd.read_csv(path_to_data, header=None, names=col_names)

# Display the first few rows of the DataFrame to understand its structure
print("Preview of the dataset:\n", df.head())

# Clean string columns by stripping extra whitespace
for c in df.select_dtypes(include=["object"]).columns:
    df[c] = df[c].str.strip()

# Define the target variable (the outcome we want to predict)
target_column = "income"

# Define the feature columns we want to use for prediction
raw_feature_cols = [
    "age",
    "education-num",
    "workclass",
    "hours-per-week",
    "sex",
    "race",
]

# Print the distribution of the target variable to understand the class imbalance
print("Income distribution:\n", df[target_column].value_counts(normalize=True))

# Check the data types of the feature columns
print("Data types of features:\n", df[raw_feature_cols].dtypes)

# Prepare the feature matrix by converting categorical variables to dummy/indicator variables
X = pd.get_dummies(df[raw_feature_cols], drop_first=True)
print("Feature matrix preview:\n", X.head(n=5))

# Convert the target variable to a binary format
y = np.where(df[target_column] == "<=50K", 0, 1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123
)

# Create a decision stump (a shallow decision tree) for AdaBoost
decision_stump = DecisionTreeClassifier(max_depth=1)

# Create the AdaBoost classifier with the correct parameter for the base estimator
ada_classifier = AdaBoostClassifier(
    estimator=decision_stump, algorithm="SAMME"
)  # Specify the algorithm


# Create the Gradient Boosting Classifier
grad_classifier = GradientBoostingClassifier()

# Fit the AdaBoost model on the training data and make predictions on the test set
ada_classifier.fit(X_train, y_train)
y_pred_ada = ada_classifier.predict(X_test)

# Fit the Gradient Boosting model on the training data and make predictions on the test set
grad_classifier.fit(X_train, y_train)
y_pred_grad = grad_classifier.predict(X_test)

# Calculate and print the accuracy and F1 score for both models
print(f"AdaBoost accuracy: {accuracy_score(y_test, y_pred_ada):.4f}")
print(f"AdaBoost F1-score: {f1_score(y_test, y_pred_ada):.4f}")

print(f"Gradient Boost accuracy: {accuracy_score(y_test, y_pred_grad):.4f}")
print(f"Gradient Boost F1-score: {f1_score(y_test, y_pred_grad):.4f}")

# Perform hyperparameter tuning for the AdaBoost model using GridSearchCV
n_estimators_list = [10, 30, 50, 70, 90]
estimator_parameters = {"n_estimators": n_estimators_list}

# Create a GridSearchCV object for AdaBoost
ada_gridsearch = GridSearchCV(
    ada_classifier, estimator_parameters, cv=5, scoring="accuracy", verbose=True
)
ada_gridsearch.fit(X_train, y_train)

# Extract the mean test scores from the grid search results for plotting
ada_scores_list = ada_gridsearch.cv_results_["mean_test_score"]

# Plot the mean test scores against the number of estimators
plt.scatter(n_estimators_list, ada_scores_list)
plt.title("AdaBoost Mean Test Scores vs. Number of Estimators")
plt.xlabel("Number of Estimators")
plt.ylabel("Mean Test Score")
plt.grid(True)  # Add grid for better visualization
plt.show()  # Display the plot
