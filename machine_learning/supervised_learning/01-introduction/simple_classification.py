import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Load the data
photo_id_times = pd.read_csv("../../../assets/csv/photo_id_times.csv")

# Separate the data into independent and dependent variables
X = np.array(photo_id_times["Time to id photo"]).reshape(-1, 1)
y = photo_id_times["Class"]

# Create a model and fit it to the data
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

# Someone who identifies a picture in 5 seconds can pass as a human
time_to_identify_picture = 6

# Make a prediction based on how long it takes to identify a picture
y_pred = neigh.predict(np.array(time_to_identify_picture).reshape(1, -1))

if y_pred == 1:
    print("We think you're a robot.")
else:
    print("Welcome, human!")
