import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error

# Predict house prices (data/housing.csv)

FILE_PATH = "data/housing.csv"

data = pd.read_csv(FILE_PATH)

# Quick overviewe of the dataset

print(data.head())

# Quick overview of the descriptive analytics of the dataset

print(data.info())

# mean, std, min, max, 25%, 50%, 75%

print(data.describe())

# Features and target
# Features -> inputs to the model (region, year, number of rooms etc.) X matrix 
# Target -> the house price y vector [100.000, 200.000, 300.000]

X = data[['sqft_living', 'bedrooms', 'bathrooms']]
y = data['price']

# Split the dataset to Train/Test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=84)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Evaluation of the model

mse = mean_squared_error(y_test, predictions)
rmse = root_mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print(f"Mean Squared error: {mse}")
print(f"Root Mean Squared error: {rmse}")
print(f"Mean Absolute error: {mae}")