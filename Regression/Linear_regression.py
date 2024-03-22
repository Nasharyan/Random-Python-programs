import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def single_linear_regression_sklearn(x, y):
    # Reshape x for sklearn if it's a 1D array
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    
    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(x, y)
    
    # Predict response vector
    y_pred = model.predict(x)
    
    # Calculate R-squared value
    r_squared = r2_score(y, y_pred)
    
    # Return R-squared value, intercept, slope, and predicted response vector
    return r_squared, model.intercept_, model.coef_, y_pred

# Generate random data for linear regression
np.random.seed(0) # For constant results
x = np.random.rand(100)  # Generate 100 random values between 0 and 1
y = 2 * x + 1 + np.random.randn(100) * 0.2   # Generate y values with a linear relationship to x and some noise

# Perform linear regression using scikit-learn
r_squared, b_0, b_1, y_pred = single_linear_regression_sklearn(x, y)

# Print estimated coefficients
print("Estimated coefficients:\nb_0 = {} \nb_1 = {}".format(b_0, b_1))

# Print R-squared value
print("R-squared: {}".format(r_squared))

# Calculate percentage of variance explained
variance_explained_percentage = r_squared * 100

# Print percentage of variance explained
print("Percentage of Variance Explained: {}%".format(variance_explained_percentage))

# Plotting regression line and data points
plt.scatter(x, y, color="b", marker="o", s=30, label='Data Points')  # Scatter plot of data points
plt.plot(x, y_pred, color="r", label='Regression Line')  # Plot regression line
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression using scikit-learn')
plt.show()
