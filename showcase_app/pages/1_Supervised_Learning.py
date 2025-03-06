import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn import tree
from sklearn.datasets import fetch_20newsgroups, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.title("Supervised Learning Projects")
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "Cancer Classifier",
        "Honey Production",
        "Fraud Detection",
        "Tennis Ace",
        "Flag Classifier",
        "Email Similarity",
        "Wine Quality",
        "Income Predictor",
    ]
)


def create_honey_plots():
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        return fig, (ax1, ax2)
    except Exception as e:
        st.error(f"Error creating plots: {str(e)}")
        return None, None


def analyze_fraud_detection():
    try:
        # Load the data
        transactions = pd.read_csv("assets/csv/transactions_modified.csv")
        transactions = transactions.dropna()

        # Create features
        transactions["isPayment"] = 0
        transactions.loc[
            transactions["type"].isin(["PAYMENT", "DEBIT"]), "isPayment"
        ] = 1
        transactions["isMovement"] = 0
        transactions.loc[
            transactions["type"].isin(["CASH_OUT", "TRANSFER"]), "isMovement"
        ] = 1
        transactions["accountDiff"] = abs(
            transactions["oldbalanceDest"] - transactions["oldbalanceOrg"]
        )

        # Prepare features and labels
        features = transactions[["amount", "isPayment", "isMovement", "accountDiff"]]
        label = transactions["isFraud"]

        return transactions, features, label
    except Exception as e:
        st.error(f"Error loading fraud detection data: {str(e)}")
        return None, None, None


def analyze_tennis_data():
    try:
        # Load the data
        df = pd.read_csv("assets/csv/tennis_stats.csv")

        # Create features for different models
        single_feature = df[["FirstServeReturnPointsWon"]]
        two_features = df[["BreakPointsOpportunities", "FirstServeReturnPointsWon"]]
        multiple_features = df[
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

        return df, single_feature, two_features, multiple_features, outcome
    except Exception as e:
        st.error(f"Error loading tennis data: {str(e)}")
        return None, None, None, None, None


def analyze_flag_data():
    try:
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

        # Load the data
        df = pd.read_csv(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data",
            names=cols,
        )

        # Predictor variables - removed 'mainhue' as it's causing issues
        var = [
            "red",
            "green",
            "blue",
            "gold",
            "white",
            "black",
            "orange",
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

        # Filter for Europe and Oceania
        df_36 = df[df["landmass"].isin([3, 6])]

        # Ensure all features are numeric
        for col in var:
            df_36[col] = pd.to_numeric(df_36[col], errors="coerce")

        # Drop any rows with NaN values after conversion
        df_36 = df_36.dropna(subset=var)

        labels = (df_36["landmass"] == 3).astype(int)

        # No need for get_dummies since all features are now numeric
        data = df_36[var]

        return df_36, data, labels, var
    except Exception as e:
        st.error(f"Error loading flag data: {str(e)}")
        return None, None, None, None


def analyze_email_data():
    try:
        # Load the training and test data
        train_emails = fetch_20newsgroups(
            categories=["comp.sys.ibm.pc.hardware", "rec.sport.hockey"],
            subset="train",
            shuffle=True,
            random_state=108,
        )
        test_emails = fetch_20newsgroups(
            categories=["comp.sys.ibm.pc.hardware", "rec.sport.hockey"],
            subset="test",
            shuffle=True,
            random_state=108,
        )

        # Initialize and fit CountVectorizer
        counter = CountVectorizer()
        counter.fit(train_emails.data + test_emails.data)

        # Transform data
        train_counts = counter.transform(train_emails.data)
        test_counts = counter.transform(test_emails.data)

        return train_emails, test_emails, counter, train_counts, test_counts
    except Exception as e:
        st.error(f"Error loading email data: {str(e)}")
        return None, None, None, None, None


def analyze_wine_data():
    try:
        # Load the dataset
        df = pd.read_csv("assets/csv/wine_quality.csv")

        # Separate target and features
        y = df["quality"]
        features = df.drop(columns=["quality"])

        # Standardize features
        scaler = StandardScaler().fit(features)
        X = scaler.transform(features)

        return df, X, y, features.columns
    except Exception as e:
        st.error(f"Error loading wine data: {str(e)}")
        return None, None, None, None


with tab1:
    code = """
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load data and split
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=100
)

# Train model
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)
accuracy = classifier.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
    """

    st.code(code, language="python")

    if st.button("Run Cancer Classifier"):
        data = load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=100
        )
        classifier = KNeighborsClassifier(n_neighbors=3)
        classifier.fit(X_train, y_train)
        accuracy = classifier.score(X_test, y_test)
        st.write(f"Accuracy: {accuracy:.4f}")

    # Add GitHub link at the bottom of tab1
    st.markdown("---")
    st.markdown(
        '<a href="https://github.com/eftekin/AI-EngVentures/blob/main/projects/supervised_learning/cancer_classifier/main.py" target="_blank"><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="25" style="vertical-align: middle;"/></a> | <a href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html">Dataset: Breast Cancer Wisconsin</a>',
        unsafe_allow_html=True,
    )

with tab2:
    code = """
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load and prepare data
df = pd.read_csv(
    "https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv",
    storage_options={"User-Agent": "Mozilla/5.0"},
)

# Calculate mean production per year
prod_per_year = df.groupby("year").totalprod.mean().reset_index()

# Prepare data for model
X = prod_per_year["year"].values.reshape(-1, 1)
y = prod_per_year["totalprod"]

# Create and fit model
regr = LinearRegression()
regr.fit(X, y)

# Make future predictions
X_future = np.array(range(2013, 2051)).reshape(-1, 1)
future_predict = regr.predict(X_future)
    """

    st.code(code, language="python")

    if st.button("Run Honey Production Analysis"):
        try:
            df = pd.read_csv(
                "https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv",
                storage_options={"User-Agent": "Mozilla/5.0"},
            )

            prod_per_year = df.groupby("year").totalprod.mean().reset_index()

            X = prod_per_year["year"].values.reshape(-1, 1)
            y = prod_per_year["totalprod"]

            # Create and train model
            regr = LinearRegression()
            regr.fit(X, y)

            # Show model results
            st.write("Model Parameters:")
            st.write(f"Slope: {regr.coef_[0]:.2f}")
            st.write(f"Intercept: {regr.intercept_:.2f}")

            # Create plots
            fig, (ax1, ax2) = create_honey_plots()
            if fig is not None:
                # Historical data with regression line
                ax1.scatter(X, y)
                ax1.plot(X, regr.predict(X), color="red")
                ax1.set_xlabel("Year")
                ax1.set_ylabel("Total Production")
                ax1.set_title("Historical Honey Production with Regression")

                # Future predictions
                X_future = np.array(range(2013, 2051)).reshape(-1, 1)
                future_predict = regr.predict(X_future)

                ax2.plot(X_future, future_predict, color="blue")
                ax2.set_xlabel("Year")
                ax2.set_ylabel("Predicted Production")
                ax2.set_title("Predicted Honey Production (2013-2050)")

                st.pyplot(fig)
                plt.close(fig)  # Properly close the figure
        except Exception as e:
            st.error(f"Error processing honey production data: {str(e)}")

    # Update GitHub link with correct username
    st.markdown("---")
    st.markdown(
        '<a href="https://github.com/eftekin/AI-EngVentures/blob/main/projects/supervised_learning/honey_production/main.py" target="_blank"><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="25" style="vertical-align: middle;"/></a> | <a href="https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv">Dataset: Honey Production</a>',
        unsafe_allow_html=True,
    )

with tab3:
    st.header("Fraud Detection Analysis")

    code = """
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and prepare data
transactions = pd.read_csv("transactions_modified.csv")
transactions = transactions.dropna()

# Create features
transactions["isPayment"] = transactions["type"].isin(["PAYMENT", "DEBIT"]).astype(int)
transactions["isMovement"] = transactions["type"].isin(["CASH_OUT", "TRANSFER"]).astype(int)
transactions["accountDiff"] = abs(transactions["oldbalanceDest"] - transactions["oldbalanceOrg"])

# Prepare and split data
features = transactions[["amount", "isPayment", "isMovement", "accountDiff"]]
label = transactions["isFraud"]
X_train, X_test, y_train, y_test = train_test_split(features, label, train_size=0.7)

# Train model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
    """

    st.code(code, language="python")

    if st.button("Run Fraud Detection Analysis"):
        transactions, features, label = analyze_fraud_detection()
        if transactions is not None:
            try:
                # Create visualizations
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Fraud Distribution")
                    fig1, ax1 = plt.subplots(figsize=(6, 4))
                    sns.countplot(x="isFraud", data=transactions)
                    plt.title("Fraudulent vs Non-Fraudulent")
                    st.pyplot(fig1)
                    plt.close(fig1)

                with col2:
                    st.subheader("Transaction Amounts")
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    sns.histplot(transactions["amount"], bins=50, kde=True)
                    plt.title("Distribution of Amounts")
                    st.pyplot(fig2)
                    plt.close(fig2)

                # Model training and evaluation
                X_train, X_test, y_train, y_test = train_test_split(
                    features, label, train_size=0.7
                )
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model = LogisticRegression()
                model.fit(X_train_scaled, y_train)

                st.subheader("Model Performance")
                st.write(f"Training Score: {model.score(X_train_scaled, y_train):.4f}")
                st.write(f"Test Score: {model.score(X_test_scaled, y_test):.4f}")

                st.success("Fraud detection analysis completed successfully!")
            except Exception as e:
                st.error(f"Error during fraud detection analysis: {str(e)}")

    st.markdown("---")
    st.markdown(
        '<a href="https://github.com/eftekin/AI-EngVentures/blob/main/projects/supervised_learning/predict_credit_card_fraud/main.py" target="_blank"><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="25" style="vertical-align: middle;"/></a> | <a href="https://github.com/eftekin/AI-EngVentures/blob/main/assets/csv/transactions_modified.csv">Dataset: Financial Fraud</a>',
        unsafe_allow_html=True,
    )

with tab4:
    st.header("Tennis Ace Analysis")

    code = """
import pandas as pd
from sklearn.linear.model import LinearRegression
from sklearn.model_selection import train_test_split

# Load tennis stats data
df = pd.read_csv("tennis_stats.csv")

# Train single feature model
features = df[["FirstServeReturnPointsWon"]]
outcome = df[["Winnings"]]
features_train, features_test, outcome_train, outcome_test = train_test_split(
    features, outcome, train_size=0.8
)
model = LinearRegression()
model.fit(features_train, outcome_train)
score = model.score(features_test, outcome_test)
    """

    st.code(code, language="python")

    if st.button("Run Tennis Analysis"):
        df, single_feature, two_features, multiple_features, outcome = (
            analyze_tennis_data()
        )
        if df is not None:
            try:
                # Create visualizations
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Break Points vs Winnings")
                    fig1, ax1 = plt.subplots(figsize=(6, 4))
                    plt.scatter(
                        df["BreakPointsOpportunities"], df["Winnings"], alpha=0.5
                    )
                    plt.xlabel("Break Points Opportunities")
                    plt.ylabel("Winnings")
                    st.pyplot(fig1)
                    plt.close(fig1)

                # Train and evaluate models
                models = {
                    "Single Feature": (single_feature, "FirstServeReturnPointsWon"),
                    "Two Features": (two_features, "BreakPoints & FirstServe"),
                    "Multiple Features": (multiple_features, "All Features"),
                }

                st.subheader("Model Performance")
                for model_name, (features, desc) in models.items():
                    X_train, X_test, y_train, y_test = train_test_split(
                        features, outcome, train_size=0.8
                    )
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    score = model.score(X_test, y_test)
                    st.write(f"{model_name} ({desc}) - R² Score: {score:.4f}")

                st.success("Tennis analysis completed successfully!")
            except Exception as e:
                st.error(f"Error during tennis analysis: {str(e)}")

    st.markdown("---")
    st.markdown(
        '<a href="https://github.com/eftekin/AI-EngVentures/blob/main/projects/supervised_learning/tennis_ace/main.py" target="_blank"><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="25" style="vertical-align: middle;"/></a> | <a href="https://github.com/eftekin/AI-EngVentures/blob/main/assets/csv/tennis_stats.csv">Dataset: Tennis Statistics</a>',
        unsafe_allow_html=True,
    )

with tab5:
    st.header("Flag Classification")

    code = """
# Load and prepare flag data
df_36, data, labels, var = analyze_flag_data()

# Split the data
train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, random_state=1, test_size=0.4
)

# Train model with optimal parameters
clf = DecisionTreeClassifier(max_depth=3, ccp_alpha=0.02)
clf.fit(train_data, train_labels)
accuracy = clf.score(test_data, test_labels)
    """

    st.code(code, language="python")

    if st.button("Run Flag Classification"):
        df_36, data, labels, var = analyze_flag_data()
        if df_36 is not None:
            try:
                # Split data and train model
                train_data, test_data, train_labels, test_labels = train_test_split(
                    data, labels, random_state=1, test_size=0.4
                )

                # Create visualization of distributions
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Flag Features Distribution")
                    fig1, ax1 = plt.subplots(figsize=(6, 4))
                    df_36[var].mean().sort_values().plot(kind="bar")
                    plt.xticks(rotation=45)
                    plt.title("Average Feature Values")
                    st.pyplot(fig1)
                    plt.close(fig1)

                # Train model with optimal parameters
                clf = DecisionTreeClassifier(max_depth=3, ccp_alpha=0.02)
                clf.fit(train_data, train_labels)

                # Show model performance
                st.subheader("Model Performance")
                st.write(f"Training Score: {clf.score(train_data, train_labels):.4f}")
                st.write(f"Test Score: {clf.score(test_data, test_labels):.4f}")

                # Visualize decision tree
                fig2, ax2 = plt.subplots(figsize=(15, 10))
                tree.plot_tree(
                    clf,
                    filled=True,
                    feature_names=data.columns,
                    class_names=["Oceania", "Europe"],
                )
                st.pyplot(fig2)
                plt.close(fig2)

                st.success("Flag classification completed successfully!")
            except Exception as e:
                st.error(f"Error during flag classification: {str(e)}")

    st.markdown("---")
    st.markdown(
        '<a href="https://github.com/eftekin/AI-EngVentures/blob/main/projects/supervised_learning/find_the_flag/main.py" target="_blank"><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="25" style="vertical-align: middle;"/></a> | <a href="https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data">Dataset: Flag Database</a>',
        unsafe_allow_html=True,
    )

with tab6:
    st.header("Email Similarity Analysis")

    code = """
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load and prepare email data
train_emails, test_emails, counter, train_counts, test_counts = analyze_email_data()

# Train classifier
classifier = MultinomialNB()
classifier.fit(train_counts, train_emails.target)

# Evaluate model
accuracy = classifier.score(test_counts, test_emails.target)
print(f"Model Accuracy: {accuracy:.4f}")
    """

    st.code(code, language="python")

    # Initialize session state variables
    if "email_classifier" not in st.session_state:
        st.session_state["email_classifier"] = None
        st.session_state["email_counter"] = None
        st.session_state["email_categories"] = None
        st.session_state["model_trained"] = False

    # Model training section
    if st.button("Run Email Analysis"):
        train_emails, test_emails, counter, train_counts, test_counts = (
            analyze_email_data()
        )
        if train_emails is not None:
            try:
                # Train model
                classifier = MultinomialNB()
                classifier.fit(train_counts, train_emails.target)

                # Store in session state
                st.session_state["email_classifier"] = classifier
                st.session_state["email_counter"] = counter
                st.session_state["email_categories"] = train_emails.target_names
                st.session_state["model_trained"] = True

                # Show model performance
                st.subheader("Model Performance")
                accuracy = classifier.score(test_counts, test_emails.target)
                st.write(f"Test Accuracy: {accuracy:.4f}")
                st.success("Email analysis completed successfully!")
            except Exception as e:
                st.error(f"Error during email analysis: {str(e)}")

    # Email classification section
    st.subheader("Try it yourself!")
    user_email = st.text_area(
        "Enter a text to classify:\n"
        + "Example for hardware: 'I need help with my computer's memory upgrade'\n"
        + "Example for hockey: 'The game was amazing, what a goal in the last period!'"
    )

    if st.button("Classify Text"):
        if not st.session_state["model_trained"]:
            st.warning("Please run the email analysis first!")
        elif not user_email:
            st.warning("Please enter some text to classify")
        else:
            try:
                email_count = st.session_state["email_counter"].transform([user_email])
                prediction = st.session_state["email_classifier"].predict(email_count)
                category = st.session_state["email_categories"][prediction[0]]
                st.success(f"The email is classified as: {category}")
            except Exception as e:
                st.error(f"Error classifying text: {str(e)}")

    st.markdown("---")
    st.markdown(
        '<a href="https://github.com/eftekin/AI-EngVentures/blob/main/projects/supervised_learning/email_similarity/main.py" target="_blank"><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="25" style="vertical-align: middle;"/></a> | <a href="https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset">Dataset: 20 Newsgroups</a>',
        unsafe_allow_html=True,
    )

with tab7:
    st.header("Wine Quality Analysis")

    code = """
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

# Load and prepare wine data
df, X, y, predictors = analyze_wine_data()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)

# Train models with different regularization
clf_no_reg = LogisticRegression(penalty=None)
clf_default = LogisticRegression()  # L2 regularization
clf_no_reg.fit(X_train, y_train)
clf_default.fit(X_train, y_train)
    """

    st.code(code, language="python")

    if st.button("Run Wine Analysis"):
        df, X, y, predictors = analyze_wine_data()
        if df is not None:
            try:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=99
                )

                # Train models
                clf_no_reg = LogisticRegression(penalty=None)
                clf_default = LogisticRegression()

                clf_no_reg.fit(X_train, y_train)
                clf_default.fit(X_train, y_train)

                # Plot coefficients without regularization
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Model Coefficients (No Regularization)")
                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    coefficients = clf_no_reg.coef_.ravel()
                    coef = pd.Series(coefficients, predictors).sort_values()
                    coef.plot(kind="bar")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig1)
                    plt.close(fig1)

                # Show model performance
                st.subheader("Model Performance")

                # No regularization results
                y_pred_test = clf_no_reg.predict(X_test)
                y_pred_train = clf_no_reg.predict(X_train)
                st.write("Without Regularization:")
                st.write(f"Training F1 Score: {f1_score(y_train, y_pred_train):.4f}")
                st.write(f"Testing F1 Score: {f1_score(y_test, y_pred_test):.4f}")

                # L2 regularization results
                y_pred_train_ridge = clf_default.predict(X_train)
                y_pred_test_ridge = clf_default.predict(X_test)
                st.write("\nWith L2 Regularization:")
                st.write(
                    f"Training F1 Score: {f1_score(y_train, y_pred_train_ridge):.4f}"
                )
                st.write(f"Testing F1 Score: {f1_score(y_test, y_pred_test_ridge):.4f}")

                st.success("Wine quality analysis completed successfully!")
            except Exception as e:
                st.error(f"Error during wine analysis: {str(e)}")

    st.markdown("---")
    st.markdown(
        '<a href="https://github.com/eftekin/AI-EngVentures/blob/main/projects/supervised_learning/predict_wine_quality/main.py" target="_blank"><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="25" style="vertical-align: middle;"/></a> | <a href="https://github.com/eftekin/AI-EngVentures/blob/main/assets/csv/wine_quality.csv">Dataset: Wine Quality</a>',
        unsafe_allow_html=True,
    )

with tab8:
    st.header("Income Prediction with Random Forest")

    code = """
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and prepare data
col_names = ["age", "workclass", "fnlwgt", "education", "education-num", 
            "marital-status", "occupation", "relationship", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
df = pd.read_csv("adult.data", header=None, names=col_names)

# Clean and prepare features
feature_cols = ["age", "capital-gain", "capital-loss", "hours-per-week", "sex", "race"]
X = pd.get_dummies(df[feature_cols], drop_first=True)
y = np.where(df.income.str.strip() == "<=50K", 0, 1)

# Split and train model
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
rf = RandomForestClassifier(max_depth=8)
rf.fit(x_train, y_train)
accuracy = rf.score(x_test, y_test)
    """

    st.code(code, language="python")

    def analyze_income_data():
        try:
            # Load and prepare data
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
            df = pd.read_csv("assets/csv/adult.data", header=None, names=col_names)

            # Clean whitespace
            for col in df.select_dtypes(include=["object"]).columns:
                df[col] = df[col].str.strip()

            # Create features
            feature_cols = [
                "age",
                "capital-gain",
                "capital-loss",
                "hours-per-week",
                "sex",
                "race",
            ]
            X = pd.get_dummies(df[feature_cols], drop_first=True)
            y = np.where(df.income == "<=50K", 0, 1)

            return df, X, y, feature_cols
        except Exception as e:
            st.error(f"Error loading income data: {str(e)}")
            return None, None, None, None

    if st.button("Run Income Analysis"):
        df, X, y, feature_cols = analyze_income_data()
        if df is not None:
            try:
                # Split data
                x_train, x_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=1
                )

                # Train model and get predictions
                rf = RandomForestClassifier(max_depth=8)
                rf.fit(x_train, y_train)

                # Display model performance
                st.subheader("Model Performance")
                st.write(
                    f"Training Accuracy: {accuracy_score(y_train, rf.predict(x_train)):.4f}"
                )
                st.write(
                    f"Test Accuracy: {accuracy_score(y_test, rf.predict(x_test)):.4f}"
                )

                # Plot feature importance
                fig, ax = plt.subplots(figsize=(10, 6))
                feature_imp = pd.DataFrame(
                    {"Feature": X.columns, "Importance": rf.feature_importances_}
                ).sort_values("Importance", ascending=False)

                sns.barplot(data=feature_imp, x="Importance", y="Feature")
                plt.title("Feature Importance in Random Forest Model")
                st.pyplot(fig)
                plt.close(fig)

                # Display income distribution
                st.subheader("Income Distribution")
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                sns.countplot(data=df, x="income")
                plt.title("Distribution of Income Categories")
                st.pyplot(fig2)
                plt.close(fig2)

                st.success("Income analysis completed successfully!")

            except Exception as e:
                st.error(f"Error during income analysis: {str(e)}")

    st.markdown("---")
    st.markdown(
        '<a href="https://github.com/eftekin/AI-EngVentures/blob/main/projects/supervised_learning/random_forests/main.py" target="_blank"><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="25" style="vertical-align: middle;"/></a> | <a href="https://github.com/eftekin/AI-EngVentures/blob/main/assets/csv/adult.data">Dataset: Adult Income</a>',
        unsafe_allow_html=True,
    )
