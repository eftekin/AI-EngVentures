# Import required libraries
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the dataset from the ARFF file
data = arff.loadarff("../../assets/csv/bone-marrow.arff")
df = pd.DataFrame(data[0])

# Drop the 'Disease' column since it is not part of the prediction task
df.drop(columns=["Disease"], inplace=True)

# Convert all columns to numeric; non-convertible entries become NaN
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# For binary columns, encode values as 0 and 1 for consistency
for col in df.columns[df.nunique() == 2]:
    df[col] = (df[col] == 1) * 1.0

# Examine the count of unique values in each column to understand data types
print("Unique values per column:\n", df.nunique())

# Separate the target variable (survival status) and features
y = df["survival_status"]
X = df.drop(columns=["survival_time", "survival_status"])

# Identify numeric and categorical columns based on unique value counts
num_cols = X.columns[X.nunique() > 7]
cat_cols = X.columns[X.nunique() <= 7]

# Display columns with missing values for awareness
print("Columns with missing values:\n", X.columns[X.isnull().sum() > 0])

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a pipeline for categorical data processing: fill missing values with mode and apply one-hot encoding
cat_vals = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "ohe",
            OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore"),
        ),
    ]
)

# Create a pipeline for numerical data processing: fill missing values with mean and standardize
num_vals = Pipeline(
    [("imputer", SimpleImputer(strategy="mean")), ("scale", StandardScaler())]
)

# ColumnTransformer to apply categorical and numerical transformations separately
preprocess = ColumnTransformer(
    transformers=[
        ("cat_process", cat_vals, cat_cols),
        ("num_process", num_vals, num_cols),
    ]
)

# Define the main pipeline combining preprocessing, PCA, and Logistic Regression for prediction
pipeline = Pipeline(
    [("preprocess", preprocess), ("pca", PCA()), ("clf", LogisticRegression())]
)

# Fit the pipeline to the training data
pipeline.fit(x_train, y_train)

# Evaluate the pipeline accuracy on the test data
print("Pipeline Accuracy on Test Set:", pipeline.score(x_test, y_test))

# Define a parameter grid for hyperparameter tuning, including logistic regression regularization and PCA components
search_space = [
    {
        "clf": [LogisticRegression()],
        "clf__C": np.logspace(-4, 2, 10),
        "pca__n_components": np.linspace(30, 37, 3).astype(int),
    }
]

# Perform grid search for hyperparameter optimization
gs = GridSearchCV(pipeline, search_space, cv=5)
gs.fit(x_train, y_train)

# Store the best model from the grid search
best_model = gs.best_estimator_

# Display key attributes of the best model
print("Best Classification Model:", best_model.named_steps["clf"])
print("Best Model Hyperparameters:", best_model.named_steps["clf"].get_params())
print("Selected PCA Components:", best_model.named_steps["pca"].n_components)

# Output the final accuracy score of the optimized model on the test set
print("Best Model Accuracy on Test Set:", best_model.score(x_test, y_test))
