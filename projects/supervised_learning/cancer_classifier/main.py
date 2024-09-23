import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load the breast cancer dataset
breast_cancer_data = load_breast_cancer()

# Print the first data entry (feature values for one sample)
print("First data sample (feature values):")
print(breast_cancer_data.data[0])

# Print the feature names of the dataset
print("\nFeature names:")
print(breast_cancer_data.feature_names)

# Print the target values (0 = malignant, 1 = benign)
print("\nTarget values (0 = malignant, 1 = benign):")
print(breast_cancer_data.target)

# Print the target names (malignant and benign labels)
print("\nTarget names (malignant = 0, benign = 1):")
print(breast_cancer_data.target_names)

# Split the dataset into training and test sets
training_data, test_data, training_label, test_label = train_test_split(
    breast_cancer_data.data, breast_cancer_data.target, test_size=0.2, random_state=100
)

# Print the length of the training labels
print("\nNumber of training labels:")
print(len(training_label))

# Print the length of the training data
print("\nNumber of training data samples:")
print(len(training_data))

# Create a KNeighborsClassifier model with k=3
classifier = KNeighborsClassifier(n_neighbors=3)

# Train the model using the training data and labels
classifier.fit(training_data, training_label)

# Print the accuracy of the model on the test set
print("\nModel accuracy on test set with k=3:")
print(classifier.score(test_data, test_label))

# Test accuracy for different values of k
k_list = list(range(1, 101))
accuracies = []

# Train and evaluate the model for each value of k from 1 to 100
for k in k_list:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(training_data, training_label)
    accuracies.append(model.score(test_data, test_label))

# Plot the accuracy against different values of k
plt.plot(k_list, accuracies)
plt.title("Breast Cancer Classifier Accuracy")
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.show()
