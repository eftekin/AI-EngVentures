import pandas as pd
from sklearn.preprocessing import StandardScaler

# Import data
reviews = pd.read_csv("reviews.csv")

# Print column names
print("Column names in the dataset:")
print(reviews.columns)

# Print .info
print("\nInformation about the dataset:")
print(reviews.info())

# Look at the counts of recommended
print("\nValue counts of 'recommended' feature:")
print(reviews["recommended"].value_counts())

# Create binary dictionary
binary_dict = {True: 1, False: 0}

# Transform column
reviews["recommended"] = reviews["recommended"].map(binary_dict)

# Print your transformed column
print("\nTransformed 'recommended' feature:")
print(reviews["recommended"].value_counts())

# Look at the counts of rating
print("\nValue counts of 'rating' feature:")
print(reviews["rating"].value_counts())

# Create dictionary
rating_dict = {
    "Loved it": 5,
    "Liked it": 4,
    "Was okay": 3,
    "Not great": 2,
    "Hated it": 1,
}

# Transform rating column
reviews["rating"] = reviews["rating"].map(rating_dict)

# Print your transformed column values
print("\nTransformed 'rating' feature:")
print(reviews["rating"].value_counts())

# Get the number of categories in a feature
print("\nValue counts of 'department_name' feature:")
print(reviews["department_name"].value_counts())

# Perform get_dummies
one_hot = pd.get_dummies(reviews["department_name"])

# Join the new columns back onto the original
reviews = reviews.join(one_hot)

# Print column names
print("\nColumn names after one-hot encoding 'department_name':")
print(reviews.columns)

# Transform review_date to date-time data
reviews["review_date"] = pd.to_datetime(reviews["review_date"])

# Print review_date data type
print("\nData type of 'review_date' after transformation:")
print(reviews["review_date"].dtype)

# Get numerical columns
reviews = reviews[
    [
        "clothing_id",
        "age",
        "recommended",
        "rating",
        "Bottoms",
        "Dresses",
        "Intimate",
        "Jackets",
        "Tops",
        "Trend",
    ]
].copy()

# Reset index
reviews = reviews.set_index("clothing_id")

# Instantiate standard scaler
scaler = StandardScaler()

# Fit transform data
scaled_reviews = scaler.fit_transform(reviews)
print("\nScaled numerical features:")
print(scaled_reviews)
