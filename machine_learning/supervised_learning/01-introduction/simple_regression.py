import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the data
housing_data = pd.read_csv("../../../assets/csv/housing_data.csv")
X = housing_data[["Sq ft", "Burglaries"]]
y = housing_data["Rent"]

# Create the model
reg = LinearRegression()

# Train the model
reg.fit(X, y)

square_footage = 950
number_of_burglaries = 2

y_pred = reg.predict(np.array([square_footage, number_of_burglaries]).reshape(1, 2))

print(y_pred)
