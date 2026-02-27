import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

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

def Poly_SquaredLoss(x,y,w): 
    y_poly = evaluate_poly(x_poly, w)
    return np.mean((y -y_poly_noise)**2)

def PolGradientDescent(X,y,iter,gamma,n,d,tol = 1e-6):
    X_pow = np.concatenate([np.power(X, i) for i in range(1, len(y)+1)], axis=1)
    W = np.zeros((d,iter))
    L = np.zeros(iter)
    w = np.zeros(len(y)+1)
    W[0] = w
    print(W[0])
    L[0] = Poly_SquaredLoss(X,y,w)
    for i in range(iter):
        W[:, i] = w
        y_hat = evaluate_poly(X, w)
        w[0] = w[0] + (2*gamma/points)*np.sum(y - y_hat)
        for d in range(1,len(w)):
            q = X_pow[:,d]
            w[d] = w[d] + (2*gamma/points)*np.dot(q,(y-y_hat))
        L[i] = Poly_SquaredLoss(X,y,w)
        if L[i] < tol:
            break
    return i,W,L

iter = 100
gamma = 0.0005
points=100
x_poly = np.linspace(-10, 10, points).reshape(points, 1)
print(x_poly)
w = np.random.rand(3)
y_poly = evaluate_poly(x_poly, w)
eps = np.random.normal(0, 1, len(y_poly))
y_poly_noise = y_poly + eps

i, wgd, L = PolGradientDescent(x_poly, y_poly_noise, iter, gamma, points, len(y_poly_noise))
plt.plot(L)
plt.xlabel('Iter')
plt.ylabel('Loss')
plt.title('Squared Loss')
plt.show()