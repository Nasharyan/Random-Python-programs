import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score


# Generate random data
np.random.seed(1)
X = 6 * np.random.rand(200, 1) - 3
noise = 0.2*np.random.randn(200, 1)
y = np.cos(X) + noise



# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Linear Regression
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print("R2 Score (Linear Regression):", r2_score(y_test, y_pred))



# Polynomial Regression
degree = 2
poly = PolynomialFeatures(degree=degree, include_bias=True)
x_train_trans = poly.fit_transform(x_train)
x_test_trans = poly.transform(x_test)

lr_poly = LinearRegression()
lr_poly.fit(x_train_trans, y_train)
y_pred_poly = lr_poly.predict(x_test_trans)

print("R2 Score (Polynomial Regression):", r2_score(y_test, y_pred_poly))
print("Coefficients:", lr_poly.coef_)
print("Intercept:", lr_poly.intercept_)


# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot the original data
axs[0].plot(X, y, 'b.')
axs[0].set_title('Original Data')
axs[0].set_xlabel("X")
axs[0].set_ylabel("Y")

# Plot Linear Regression results
axs[1].plot(x_train, lr.predict(x_train), color="r", label="Linear Regression")
axs[1].plot(X, y, "b.", label="Original Data")
axs[1].set_title('Linear Regression')
axs[1].set_xlabel("X")
axs[1].set_ylabel("Y")
axs[1].legend()

# Plot Polynomial Regression results
X_new = np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly.transform(X_new)
y_new = lr_poly.predict(X_new_poly)

axs[2].plot(X_new, y_new, "r-", linewidth=2, label="Polynomial Regression")
axs[2].plot(x_train, y_train, "b.", label='Training points')
axs[2].plot(x_test, y_test, "g.", label='Testing points')
axs[2].set_title('Polynomial Regression')
axs[2].set_xlabel("X")
axs[2].set_ylabel("Y")
axs[2].legend()

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()
