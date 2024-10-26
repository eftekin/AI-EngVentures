import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("../../../assets/csv/wine_quality.csv")
print("Columns in dataset:", df.columns)

# Separate the target variable and features
y = df["quality"]
features = df.drop(columns=["quality"])

# Standardize the feature data
scaler = StandardScaler().fit(features)
X = scaler.transform(features)

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=99
)

# Fit a logistic regression classifier without regularization
clf_no_reg = LogisticRegression(penalty=None)
clf_no_reg.fit(X_train, y_train)

# Plot the coefficients for the model without regularization
predictors = features.columns
coefficients = clf_no_reg.coef_.ravel()
coef = pd.Series(coefficients, predictors).sort_values()
coef.plot(kind="bar", title="Coefficients (No Regularization)")
plt.tight_layout()
plt.show()
plt.clf()

# Evaluate training and test performance using F1 score
y_pred_test = clf_no_reg.predict(X_test)
y_pred_train = clf_no_reg.predict(X_train)
print("Training F1 Score:", f1_score(y_train, y_pred_train))
print("Testing F1 Score:", f1_score(y_test, y_pred_test))

# Fit a default logistic regression model (L2-regularized)
clf_default = LogisticRegression()
clf_default.fit(X_train, y_train)

# Evaluate training and test performance for ridge-regularized model
y_pred_train_ridge = clf_default.predict(X_train)
y_pred_test_ridge = clf_default.predict(X_test)
print("Ridge-regularized Training F1 Score:", f1_score(y_train, y_pred_train_ridge))
print("Ridge-regularized Testing F1 Score:", f1_score(y_test, y_pred_test_ridge))

# Coarse-grained hyperparameter tuning for regularization strength (C parameter)
training_scores = []
test_scores = []
C_values = [0.0001, 0.001, 0.01, 0.1, 1]
for C in C_values:
    clf = LogisticRegression(C=C)
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)
    training_scores.append(f1_score(y_train, y_pred_train))
    test_scores.append(f1_score(y_test, y_pred_test))

# Plot training and test F1 scores as a function of C
plt.plot(C_values, training_scores, label="Training Score")
plt.plot(C_values, test_scores, label="Test Score")
plt.xscale("log")
plt.xlabel("C")
plt.legend()
plt.show()
plt.clf()

# Create a parameter grid for fine-tuning C with GridSearchCV
C_values = np.logspace(-4, -2, 100)
param_grid = {"C": C_values}

# Use GridSearchCV with L2 penalty to find the best C value
clf_gs = LogisticRegression()
gs = GridSearchCV(clf_gs, param_grid=param_grid, scoring="f1", cv=5)
gs.fit(X_train, y_train)

# Print optimal C value and corresponding score
print("Best C value:", gs.best_params_, "Best F1 Score:", gs.best_score_)

# Validate the best classifier found by GridSearchCV
clf_best = LogisticRegression(C=gs.best_params_["C"])
clf_best.fit(X_train, y_train)
y_pred_best = clf_best.predict(X_test)
print("Best Model Testing F1 Score:", f1_score(y_test, y_pred_best))

# Implement L1 hyperparameter tuning with LogisticRegressionCV
C_values = np.logspace(-2, 2, 100)
clf_l1 = LogisticRegressionCV(
    Cs=C_values, cv=5, penalty="l1", scoring="f1", solver="liblinear"
)
clf_l1.fit(X, y)

# Print best C value and corresponding coefficients for L1-regularized model
print("Best C value for L1:", clf_l1.C_)
print("Best-fit coefficients for L1 regularization:", clf_l1.coef_)

# Plot the tuned L1 coefficients
coefficients = clf_l1.coef_.ravel()
coef = pd.Series(coefficients, predictors).sort_values()
coef.plot(kind="bar", title="Coefficients for Tuned L1")
plt.tight_layout()
plt.show()
plt.clf()
