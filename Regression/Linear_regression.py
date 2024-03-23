import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def single_linear_regression(x,y,n):
    
    m_x = np.mean(x)
    m_y = np.mean(y)

    s_n = np.sum(y * x) - n * m_y * m_x
    s_d = np.sum(x * x) - n * m_x * m_x

    # calculating regression coefficients
    b_1 = s_n / s_d
    b_0 = m_y - b_1 * m_x
    # predicted response vector
    y_pred = b_0 + b_1 * x

    # R-squared calculation
    ss_total = np.sum((y - np.mean(y))**2)
    ss_residual = np.sum((y - y_pred)**2)
    r_squared = 1 - (ss_residual / ss_total)
    return r_squared,b_0,b_1,y_pred


# Generate random data for linear regression
np.random.seed(0) # For constant results
x = np.random.rand(100)  # Generate 100 random values between 0 and 1
y = 2 * x + 1 + np.random.randn(100) * 0.2   # Generate y values with a linear relationship to x and some noise
# If using custom data make sure the x and y are numerical values
n = np.size(x)
r_squared,b_0, b_1,y_pred=single_linear_regression(x,y,n)
# observations / data
print("Estimated coefficients:\nb_0 = {} \nb_1 = {}".format(b_0, b_1))

# print R-squared value
print("R-squared: {}".format(r_squared))

# Calculate percentage of variance explained
variance_explained_percentage = r_squared * 100

# print percentage of variance explained
print("Percentage of Variance Explained: {}%".format(variance_explained_percentage))

# plotting regression line
# plotting the actual points as a scatter plot
plt.scatter(x, y, color="b", marker="o", s=30)

# plotting the regression line
plt.plot(x, y_pred, color="r")

# putting labels
plt.xlabel('x')
plt.ylabel('y')

# function to show plot
plt.show()
