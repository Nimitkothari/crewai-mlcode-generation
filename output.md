Here is the starter Python code for the project:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Load the dataset
df = pd.read_csv('house_prices.csv')

# Split the data into training and testing sets
X = df.drop(['price'], axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models
model_linear_regression = LinearRegression()
model_decision_tree = DecisionTreeRegressor()
model_random_forest = RandomForestRegressor()
model_gradient_boosting = GradientBoostingRegressor()

# Train the models
model_linear_regression.fit(X_train, y_train)
model_decision_tree.fit(X_train, y_train)
model_random_forest.fit(X_train, y_train)
model_gradient_boosting.fit(X_train, y_train)

# Make predictions
y_pred_linear_regression = model_linear_regression.predict(X_test)
y_pred_decision_tree = model_decision_tree.predict(X_test)
y_pred_random_forest = model_random_forest.predict(X_test)
y_pred_gradient_boosting = model_gradient_boosting.predict(X_test)

# Evaluate the models
from sklearn.metrics import mean_squared_error
mse_linear_regression = mean_squared_error(y_test, y_pred_linear_regression)
mse_decision_tree = mean_squared_error(y_test, y_pred_decision_tree)
mse_random_forest = mean_squared_error(y_test, y_pred_random_forest)
mse_gradient_boosting = mean_squared_error(y_test, y_pred_gradient_boosting)

print("Linear Regression MSE:", mse_linear_regression)
print("Decision Tree MSE:", mse_decision_tree)
print("Random Forest MSE:", mse_random_forest)
print("Gradient Boosting MSE:", mse_gradient_boosting)
```

Summary: The provided dataset is suitable for regression problems and can be used to train various machine learning models, including Linear Regression, Decision Trees, Random Forest, and Gradient Boosting. The code provided is a starting point for the project and can be customized and extended as needed.