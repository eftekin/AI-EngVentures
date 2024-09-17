import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# load and investigate the data here:
df = pd.read_csv("tennis_stats.csv")
print(df.head())

# Exploratory analysis: Visualize the relationship between BreakPointsOpportunities and Winnings
plt.scatter(df["BreakPointsOpportunities"], df["Winnings"])
plt.xlabel("Break Points Opportunities")
plt.ylabel("Winnings")
plt.title("Break Points Opportunities vs Winnings")
plt.show()


### Single feature linear regression

# Select feature and outcome columns
features = df[["FirstServeReturnPointsWon"]]
outcome = df[["Winnings"]]

# Split the data into training and test sets
features_train, features_test, outcome_train, outcome_test = train_test_split(
    features, outcome, train_size=0.8
)

# Create and train the linear regression model
model = LinearRegression()
model.fit(features_train, outcome_train)

# Evaluate the model's performance on the test set
score = model.score(features_test, outcome_test)
print(f"Model Test Score: {score}")

# Visualize the model's predictions
predictions = model.predict(features_test)
plt.scatter(outcome_test, predictions, alpha=0.4)
plt.xlabel("Actual Winnings")
plt.ylabel("Predicted Winnings")
plt.title("Actual vs Predicted Winnings")
plt.show()

## Create linear regression models using different single features and compare their performance.

# Select feature and outcome columns
features_list = ["BreakPointsOpportunities", "Aces", "DoubleFaults"]
outcome = df[["Winnings"]]

# Create and evaluate a model for each feature
for feature in features_list:
    features = df[[feature]]

    # Split the data into training and test sets
    features_train, features_test, outcome_train, outcome_test = train_test_split(
        features, outcome, train_size=0.8
    )

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(features_train, outcome_train)

    # Evaluate the model's performance on the test set
    score = model.score(features_test, outcome_test)
    print(f"Model Test Score with {feature}: {score}")

    # Visualize the model's predictions
    predictions = model.predict(features_test)
    plt.scatter(outcome_test, predictions, alpha=0.4)
    plt.xlabel("Actual Winnings")
    plt.ylabel("Predicted Winnings")
    plt.title(f"Actual vs Predicted Winnings ({feature})")
    plt.show()

### Two feature linear regressions

# Select features and outcome columns
features = df[["BreakPointsOpportunities", "FirstServeReturnPointsWon"]]
outcome = df[["Winnings"]]

# Split the data into training and test sets
features_train, features_test, outcome_train, outcome_test = train_test_split(
    features, outcome, train_size=0.8
)

# Create and train the linear regression model
model = LinearRegression()
model.fit(features_train, outcome_train)

# Evaluate the model's performance on the test set
score = model.score(features_test, outcome_test)
print(f"Two Feature Model Test Score: {score}")

# Visualize the model's predictions
predictions = model.predict(features_test)
plt.scatter(outcome_test, predictions, alpha=0.4)
plt.xlabel("Actual Winnings")
plt.ylabel("Predicted Winnings")
plt.title("Actual vs Predicted Winnings (Two Features)")
plt.show()

### Multiple feature linear regression

# Select features and outcome columns
features = df[
    [
        "FirstServe",
        "FirstServePointsWon",
        "FirstServeReturnPointsWon",
        "SecondServePointsWon",
        "SecondServeReturnPointsWon",
        "Aces",
        "BreakPointsConverted",
        "BreakPointsFaced",
        "BreakPointsOpportunities",
        "BreakPointsSaved",
        "DoubleFaults",
        "ReturnGamesPlayed",
        "ReturnGamesWon",
        "ReturnPointsWon",
        "ServiceGamesPlayed",
        "ServiceGamesWon",
        "TotalPointsWon",
        "TotalServicePointsWon",
    ]
]
outcome = df[["Winnings"]]

# Split the data into training and test sets
features_train, features_test, outcome_train, outcome_test = train_test_split(
    features, outcome, train_size=0.8
)

# Create and train the linear regression model
model = LinearRegression()
model.fit(features_train, outcome_train)

# Evaluate the model's performance on the test set
score = model.score(features_test, outcome_test)
print(f"Multiple Feature Model Test Score: {score}")

# Visualize the model's predictions
predictions = model.predict(features_test)
plt.scatter(outcome_test, predictions, alpha=0.4)
plt.xlabel("Actual Winnings")
plt.ylabel("Predicted Winnings")
plt.title("Actual vs Predicted Winnings (Multiple Features)")
plt.show()
