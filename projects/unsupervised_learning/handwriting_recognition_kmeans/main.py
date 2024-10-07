import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

# Load the digits dataset
digits = datasets.load_digits()

# Print the description of the dataset to understand what it contains
print("Dataset description:\n", digits.DESCR)

# Print the data and target (the actual digits labels)
print(
    "\nFirst few samples of digit data (each row represents a digit image as a flattened array):\n",
    digits.data[:5],
)
print("\nCorresponding digit labels:\n", digits.target[:5])

# Visualize the first 64 digit images from the dataset
fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    # Display the image using binary colormap
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation="nearest")
    # Label each image with the corresponding digit
    ax.text(0, 7, str(digits.target[i]))

plt.show()

# Create a KMeans model to cluster the digits into 10 clusters (one for each digit)
model = KMeans(n_clusters=10, random_state=42)
model.fit(digits.data)

# Visualize the cluster centers, which represent the "average" digit for each cluster
fig = plt.figure(figsize=(8, 3))
fig.suptitle("Cluster Center Images", fontsize=14, fontweight="bold")

for i in range(10):
    ax = fig.add_subplot(2, 5, 1 + i)
    # Each cluster center is reshaped back into an 8x8 image
    ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)

plt.show()

# New samples to predict
new_samples = np.array(
    [
        [
            0.00,
            1.15,
            2.97,
            2.14,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            7.39,
            6.94,
            7.55,
            3.66,
            0.00,
            0.00,
            0.00,
            0.00,
            1.14,
            0.08,
            5.87,
            5.18,
            0.00,
            0.00,
            0.00,
            0.00,
            0.84,
            4.80,
            7.62,
            3.96,
            1.98,
            0.15,
            0.00,
            0.00,
            7.24,
            7.62,
            7.62,
            7.40,
            7.62,
            4.57,
            0.00,
            0.00,
            1.14,
            0.46,
            1.60,
            4.12,
            7.24,
            4.50,
            0.00,
            0.00,
            3.66,
            7.40,
            7.62,
            6.41,
            3.51,
            0.23,
            0.00,
            0.00,
            2.29,
            2.74,
            0.69,
            0.00,
            0.00,
            0.00,
            0.00,
        ],
        [
            0.00,
            0.00,
            0.00,
            0.00,
            0.38,
            0.08,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.08,
            6.63,
            2.90,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            2.52,
            7.47,
            0.92,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            6.10,
            4.96,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            1.98,
            7.62,
            1.37,
            0.00,
            0.00,
            0.00,
            0.00,
            0.46,
            6.56,
            7.32,
            5.87,
            7.32,
            4.19,
            0.00,
            0.00,
            0.91,
            5.87,
            4.42,
            3.74,
            5.79,
            4.80,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            2.59,
            3.35,
            0.00,
        ],
        [
            0.00,
            0.00,
            0.00,
            0.00,
            0.99,
            6.48,
            1.07,
            0.00,
            0.00,
            0.00,
            0.00,
            0.08,
            5.87,
            6.17,
            0.30,
            0.00,
            0.00,
            0.00,
            0.00,
            2.67,
            7.47,
            1.22,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            6.25,
            4.73,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            1.60,
            7.62,
            4.88,
            0.15,
            0.00,
            0.00,
            0.00,
            0.00,
            3.05,
            7.62,
            7.62,
            1.45,
            0.00,
            0.00,
            0.00,
            0.00,
            0.99,
            5.03,
            3.73,
            0.07,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
        ],
        [
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            1.60,
            7.32,
            0.53,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            5.57,
            5.79,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            1.22,
            7.63,
            1.91,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            2.97,
            7.47,
            3.43,
            1.37,
            0.00,
            0.00,
            0.00,
            0.00,
            5.64,
            7.48,
            6.56,
            7.55,
            2.21,
            0.00,
            0.00,
            0.00,
            2.90,
            7.47,
            6.40,
            7.62,
            2.21,
            0.00,
            0.00,
            0.00,
            0.23,
            2.75,
            3.05,
            1.75,
            0.00,
            0.00,
            0.00,
        ],
    ]
)

# Predict the labels for new samples (these are new handwritten digit images represented as arrays)
new_labels = model.predict(new_samples)

# Output the predicted cluster labels
print("\nPredicted cluster labels for new samples:", new_labels)

# Map the cluster labels to actual digit numbers and print them
print("Predicted digits for the new samples: ", end="")
for i in range(len(new_labels)):
    if new_labels[i] == 0:
        print(0, end="")
    elif new_labels[i] == 1:
        print(9, end="")
    elif new_labels[i] == 2:
        print(2, end="")
    elif new_labels[i] == 3:
        print(1, end="")
    elif new_labels[i] == 4:
        print(6, end="")
    elif new_labels[i] == 5:
        print(8, end="")
    elif new_labels[i] == 6:
        print(4, end="")
    elif new_labels[i] == 7:
        print(5, end="")
    elif new_labels[i] == 8:
        print(7, end="")
    elif new_labels[i] == 9:
        print(3, end="")
