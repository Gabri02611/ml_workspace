import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv


iter = 1000
gamma = 0.01
points=100
# Create an array 'x_poly' with 100 evenly spaced values between -10 and 10.
# Reshape it to be a column vector (100 rows, 1 column).
x_poly = np.linspace(-10, 10, points).reshape(100, 1)

# Generate random coefficients 'w' for a quadratic polynomial.
# 'w' will have 3 coefficients, representing the quadratic term, linear term, and intercept.
w = np.random.rand(3)
print(w)
def evaluate_poly(X, w):
    """
    Parameters
    ----------
    X : array of float dim n x d
        Matrix containing the dataset
    w : array of float of dim deg
        Coefficients of the polynomial
    """
    # this is the actual degree - 1 because we have to consider the intercept
    deg = len(w)

    # column i is the i-th power of the datapoints
    X_pow = np.concatenate([np.power(X, i) for i in range(1, deg)], axis=1)
    return np.dot(X_pow, w[1:]) + w[0]

# Evaluate the quadratic polynomial defined by 'w' on the 'x_poly' data points.
y_poly = evaluate_poly(x_poly, w)
# add noise to the dataset

# Generate random noise 'eps' from a normal distribution with mean 0 and standard deviation 1.
# The length of 'eps' matches the number of data points in 'y_poly.'
eps = np.random.normal(0, 1, len(y_poly))
y_poly_noise = y_poly + eps
print(len(y_poly_noise))
print(len(y_poly))

def SquaredLoss(X,y,w): 
    return 1/len(y) * 

def PolGradientDescent(X,y,iter,gamma,n,d,tol = 1e-6):
    W = np.zeros((iter,d))
    L = np.zeros(iter)
    w = np.random.normal(0, 0.1, d)
    W[0] = w
    L[0] = SquaredLoss(X,y,w)
    for i in range(iter):
        w  = w + gamma *2/n * X.T.dot(y - X.dot(w))
        W[i] = w
        loss = L[0] = SquaredLoss(X,y,w)
        L[i] = loss
        if loss < tol:
            break
    return i,W,L


plt.plot(x_poly, y_poly_noise, 'o', markersize=4, c='orange')
plt.plot(x_poly, y_poly, '-', c='blue')
plt.title('Quadratic Noisy Data')

plt.show()