from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the training and test data for two categories: hardware and hockey
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

# Initialize a CountVectorizer to convert email text to a matrix of token counts
counter = CountVectorizer()

# Fit the CountVectorizer to the combined training and test data
counter.fit(train_emails.data + test_emails.data)

# Transform the training and test data into count matrices
train_counts = counter.transform(train_emails.data)
test_counts = counter.transform(test_emails.data)

# Initialize a Multinomial Naive Bayes classifier
classifier = MultinomialNB()

# Train the classifier using the training data
classifier.fit(train_counts, train_emails.target)

# Evaluate the classifier's accuracy on the test data
accuracy = classifier.score(test_counts, test_emails.target)
print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")


# Practical Part: Test the classifier with new email inputs
def classify_email(email):
    """
    Takes a new email as input, processes it, and returns the predicted category.
    """
    email_count = counter.transform([email])  # Transform the email into count matrix
    prediction = classifier.predict(email_count)  # Make a prediction
    category = train_emails.target_names[prediction[0]]  # Get the category name
    return category


# Get a new email input from the user
test_email = input("Enter an email to classify (hardware or hockey related): ")

# Classify the input email
predicted_category = classify_email(test_email)
print(f"The email is classified as: {predicted_category}")
