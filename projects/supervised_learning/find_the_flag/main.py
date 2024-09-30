import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Column names for the dataset
cols = [
    "name",
    "landmass",
    "zone",
    "area",
    "population",
    "language",
    "religion",
    "bars",
    "stripes",
    "colours",
    "red",
    "green",
    "blue",
    "gold",
    "white",
    "black",
    "orange",
    "mainhue",
    "circles",
    "crosses",
    "saltires",
    "quarters",
    "sunstars",
    "crescent",
    "triangle",
    "icon",
    "animate",
    "text",
    "topleft",
    "botright",
]

# Load the flag dataset from UCI repository
df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data",
    names=cols,
)

# Predictor variables (flag colors, shapes, etc.)
var = [
    "red",
    "green",
    "blue",
    "gold",
    "white",
    "black",
    "orange",
    "mainhue",
    "bars",
    "stripes",
    "circles",
    "crosses",
    "saltires",
    "quarters",
    "sunstars",
    "triangle",
    "animate",
]

# Display the count of countries by landmass (continents)
print("Number of countries by landmass (continent):")
print(df["landmass"].value_counts())

# Create a new dataframe with only flags from Europe (landmass 3) and Oceania (landmass 6)
df_36 = df[df["landmass"].isin([3, 6])]

# Display the average values of the predictors for Europe and Oceania
print("\nAverage values of predictors for Europe (3) and Oceania (6):")
## Select only numeric columns for the mean calculation
numeric_cols = df_36[var].select_dtypes(include=[np.number]).columns
print(df_36.groupby("landmass")[numeric_cols].mean())


# Create binary labels for Europe (1) and Oceania (0)
labels = (df_36["landmass"] == 3).astype(int)

# Display the data types of the predictors
print("\nData types of the predictors:")
print(df_36[var].dtypes)

# Convert categorical predictors into dummy variables
data = pd.get_dummies(df_36[var])

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, random_state=1, test_size=0.4
)

# Fit a decision tree for different max_depth values (1-20) and store accuracy scores
depths = range(1, 21)
acc_depth = []

for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(train_data, train_labels)
    acc_depth.append(clf.score(test_data, test_labels))

# Plot the accuracy vs. tree depth
plt.plot(depths, acc_depth)
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree Accuracy vs Max Depth")
plt.show()

# Find the highest accuracy and corresponding tree depth
max_acc = np.max(acc_depth)
best_depth = depths[np.argmax(acc_depth)]
print(f"\nBest accuracy: {max_acc:.3f} at max depth: {best_depth}")

# Refit the decision tree with the best depth and visualize the tree
clf = DecisionTreeClassifier(max_depth=best_depth)
clf.fit(train_data, train_labels)

plt.figure(figsize=(20, 10))
tree.plot_tree(
    clf, filled=True, feature_names=data.columns, class_names=["Oceania", "Europe"]
)
plt.title("Decision Tree with Best Depth")
plt.show()

# Prune the decision tree using cost complexity pruning (ccp_alpha) and evaluate accuracy
ccp = np.linspace(0, 0.05, 50)
acc_pruned = []

for alpha in ccp:
    clf = DecisionTreeClassifier(max_depth=best_depth, ccp_alpha=alpha)
    clf.fit(train_data, train_labels)
    acc_pruned.append(clf.score(test_data, test_labels))

# Plot the accuracy vs. ccp_alpha (pruning parameter)
plt.plot(ccp, acc_pruned)
plt.xlabel("ccp_alpha")
plt.ylabel("Accuracy")
plt.title("Decision Tree Accuracy vs ccp_alpha (Pruning)")
plt.show()

# Find the highest accuracy and corresponding ccp_alpha value
max_acc_pruned = np.max(acc_pruned)
best_alpha = ccp[np.argmax(acc_pruned)]
print(f"\nBest pruned accuracy: {max_acc_pruned:.3f} at ccp_alpha: {best_alpha:.4f}")

# Fit a decision tree model using the optimal max_depth and ccp_alpha values
clf = DecisionTreeClassifier(max_depth=best_depth, ccp_alpha=best_alpha)
clf.fit(train_data, train_labels)

# Visualize the final decision tree
plt.figure(figsize=(20, 10))
tree.plot_tree(
    clf, filled=True, feature_names=data.columns, class_names=["Oceania", "Europe"]
)
plt.title("Final Pruned Decision Tree")
plt.show()
