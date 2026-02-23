import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
def datagen(d, points, m, M, w, sigma):
    """
    Parameters
    ----------
    d : int
        Dimension of each data sample
    points : int
        Number of points to be generated
    m : float
        Lower bound for the domain of the data points
    M : float
        Upper bound for the domain of the data points
    w : float array of dim d
        Vector of weights of the linear model
    sigma : float
        Standard deviation of the noise eps
    """
    X = np.zeros((points, d))
    for i in range(points):
        X[i,:] = np.random.uniform(m, M, d)
    eps = np.random.normal(0, sigma, points)
    y = np.dot(X, w) + eps
    return X, y
def SquareLoss(X,y,w):
    """
    Parameters
    ----------
    X : array of float dim n x d
        Matrix containing the dataset
    y : array of float of dim n
        Vector containing the ground truth value of each data point
    w : array of float of dim d
        Weights of the fitted line
    """
    return np.linalg.norm(y - X@w, 2)
def GradientDescent(X,y,iter,gamma,n,d,tol = 1e-6):
    """
    Parameters
    ----------
    X : array of float dim n x d
        Matrix containing the dataset
    y : array of float of dim n
        Vector containing the ground truth value of each data point
    iter : int
        Number of GD iterations
    gamma : float
        Learning rate
    points : int
        Number of points in our dataset
    d : int
        Dimensionality of each data point in the dataset
    """
    # Initialize arrays to store weights and losses
    W = np.zeros((iter,d)) # To store weights at each iteration
    L = np.zeros(iter) # To store losses at each iteration

    # Initialize the weight vector w with small random values
    w = np.random.normal(0, 0.1, d)
    # GD iterations
    W[0] = w
    L[0] = SquareLoss(X,y,w)/n
    for i in range(iter):
        w = w + gamma * 2/n * X.T.dot(y - X.dot(w))
        W[i] = w
        loss = SquareLoss(X, y, w) / n
        L[i] = loss
        if loss < tol:
            break
        
    return i,W, L
d=1
sigma = 1
points = 1000
m = -10
M = 10
lam = 0.01
iter = 1000
n_points = 100
gamma = 0.001
w = np.random.normal(0, 1, d)
X_reg, y_reg = datagen(d, points, m, M, w, sigma)
d = np.shape(X_reg)[1]



# Apply Gradient Descent (GD) to train a linear regression model
i, wgd, L = GradientDescent(X_reg, y_reg, iter, gamma, points, d)
plt.plot(L)
plt.xlabel('Iter')
plt.ylabel('Loss')
plt.title('Squared Loss')
# the last stored weights are the most updated ones
wpred = wgd[:,-1]
print(w, wpred, np.abs(w-wpred))
plt.show()

