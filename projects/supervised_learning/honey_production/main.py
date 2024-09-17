import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv(
    "https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv",
    storage_options={"User-Agent": "Mozilla/5.0"},
)

print("First few rows of the DataFrame:")
print(df.head())

prod_per_year = df.groupby("year").totalprod.mean().reset_index()
print("\nMean total production per year:")
print(prod_per_year)

X = prod_per_year["year"].values.reshape(-1, 1)
print("\nYears reshaped for model input:")
print(X)

y = prod_per_year["totalprod"]
print("\nTotal production values:")
print(y)

plt.scatter(X, y)
plt.xlabel("Year")
plt.ylabel("Total Production")
plt.title("Total Honey Production per Year")
plt.show()
plt.clf()


regr = LinearRegression()
print("\nLinear regression model created.")

regr.fit(X, y)
print("\nModel fitted to the data.")

print("\nSlope of the line:", regr.coef_[0])
print("Intercept of the line:", regr.intercept_)

y_predict = regr.predict(X)
print("\nPredicted total production values:")
print(y_predict)

plt.scatter(X, y)
plt.plot(X, y_predict, color="red")
plt.xlabel("Year")
plt.ylabel("Total Production")
plt.title("Total Honey Production per Year with Linear Regression Line")
plt.show()
plt.clf()


X_future = np.array(range(2013, 2051)).reshape(-1, 1)
print("\nFuture years reshaped for prediction:")
print(X_future)

future_predict = regr.predict(X_future)
print("\nPredicted total production values for future years:")
print(future_predict)

plt.plot(X_future, future_predict, color="blue")
plt.xlabel("Year")
plt.ylabel("Predicted Total Production")
plt.title("Predicted Honey Production from 2013 to 2050")
plt.show()
plt.clf()
