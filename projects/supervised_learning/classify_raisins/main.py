import pandas as pd
from scipy.stats import uniform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
raisins = pd.read_csv("../../../assets/csv/raisin.csv")
raisins.head()

# Define predictor and target variables
X = raisins.drop("Class", axis=1)
y = raisins["Class"]

# Display dataset overview
print("Number of features:", X.shape[1])
print("Total number of samples:", len(y))
print("Samples in class '1':", y.sum())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=19)

# Initialize Decision Tree model
tree = DecisionTreeClassifier()

# Define parameter grid for GridSearchCV
parameters = {"min_samples_split": [2, 3, 4], "max_depth": [3, 5, 7]}

# Run GridSearchCV for hyperparameter tuning
grid = GridSearchCV(tree, parameters)
grid.fit(X_train, y_train)

# Display best model and hyperparameters found by GridSearchCV
print("Best Decision Tree model:", grid.best_estimator_)
print("Best cross-validation score:", grid.best_score_)
print("Test accuracy of the final model:", grid.score(X_test, y_test))

# Summarize GridSearchCV results
grid_results_df = pd.concat(
    [
        pd.DataFrame(grid.cv_results_["params"]),
        pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["Score"]),
    ],
    axis=1,
)
print(grid_results_df)

# Initialize Logistic Regression model
lr = LogisticRegression(solver="liblinear", max_iter=1000)

# Define hyperparameter distributions for RandomizedSearchCV
distributions = {"penalty": ["l1", "l2"], "C": uniform(loc=0, scale=100)}

# Run RandomizedSearchCV for hyperparameter tuning
clf = RandomizedSearchCV(lr, distributions, n_iter=8)
clf.fit(X_train, y_train)

# Display best model and score from RandomizedSearchCV
print("Best Logistic Regression model:", clf.best_estimator_)
print("Best cross-validation score:", clf.best_score_)

# Summarize RandomizedSearchCV results
random_search_results_df = pd.concat(
    [
        pd.DataFrame(clf.cv_results_["params"]),
        pd.DataFrame(clf.cv_results_["mean_test_score"], columns=["Accuracy"]),
    ],
    axis=1,
)
print(random_search_results_df.sort_values("Accuracy", ascending=False))
