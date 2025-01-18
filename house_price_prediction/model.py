
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

# Load the dataset
df = pd.read_csv('train.csv')
  

# Eliminating NaN or missing input numbers 
df.fillna(method ='ffill', inplace = True)

# Extract the total_bill and tip columns
X = df['LotArea'].values.reshape(-1, 1)  # Reshape for sklearn
y = df['SalePrice'].values

model = LinearRegression()
model.fit(X, y)

# Predict y values
y_pred = model.predict(X)

print(f"Intercept (ω0): {model.intercept_}")
print(f"Slope (ω1): {model.coef_[0]}")


# Plot the original data points and the regression line
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.5, label='Original data')
plt.plot(X, y_pred, color='red', label='Regression line')
plt.title('LotArea vs SalePrice')
plt.xlabel('LotArea')
plt.ylabel('SalePrice')
plt.legend()
plt.grid(True)


X = df[["LotArea"]]
y = df[["SalePrice"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.7)

model = linear_model.LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("The MSE on test set is {0:.4f}".format(mean_squared_error(y_test, y_pred)))

plt.show()