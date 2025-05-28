import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv("housing_price_dataset.csv")
print (df)
data={
    'X1' : df['Avg. Area Income'],
    'X2' : df['Avg. Area House Age'],
    'X3' : df['Avg. Area Number of Rooms'],
    'X4' : df['Avg. Area Number of Bedrooms'],
    'X5' : df['Area Population'],
    'Y' : df['Price']
}
df = pd.DataFrame(data)
df.tail()
df.head()
X = df[['X1','X2','X3','X4','X5']]
Y = df['Y']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)

mse = mean_squared_error(Y_test, predictions)
r_squared = model.score(X_test, Y_test)

print(f"On the test data, the model's Mean Squared Error (MSE) is: {mse:.2f}")
print(f"The R-squared value for the model on the test data is: {r_squared:.2f}")
print(f"Model Coefficients (weights for X1, X2, X3): {model.coef_}")
print(f"Model Intercept (the predicted value when all inputs are zero): {model.intercept_}")

new_sample = np.array([[5, 6, 7, 8, 9]]) 
new_prediction = model.predict(new_sample)
print(f"For the new sample input {new_sample[0]}, the model predicts a value of:{new_prediction[0]:.2f}")

print("Training set size:", X_train.shape, Y_train.shape)
print("Testing set size:", X_test.shape, Y_test.shape)
