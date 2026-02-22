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

def CloseRegression(X, y):
    """
    Parameters
    ----------
    X : array of float dim n x d
        Matrix containing the dataset
    y : array of float of dim n
        Vector containing the ground truth value of each data point
    """
    xx =  np.linalg.inv(X.T@X)@X.T
    w =xx@y

    return w
def CloseRegressionReg(X,y,lam):
    w = np.linalg.inv(X.T @ X + lam * np.eye(X.shape[1])) @ X.T @ y
    return w
def 
# usage example





d = 1
w = np.random.normal(0, 1, d)
sigma = 1
points = 1000
m = -10
M = 10
lam = 0.01
###switch
Regression = 1
Ridge = 1
ddplot = 0
#####
X_reg, y_reg = datagen(d, points, m, M, w, sigma)
if Regression == 1:
    w1 = CloseRegression(X_reg,y_reg)
if Ridge == 1:
    w2 = CloseRegressionReg(X_reg,y_reg,lam)
# plotting the generated dataset

###single plot 
if ddplot == 0:
    plt.scatter(X_reg, y_reg, alpha = 0.4)
    plt.plot(X_reg, np.dot(X_reg, w1), '--', c='r',label = 'Non ridge')
    plt.plot(X_reg,np.dot(X_reg, w2),'--', c='g',label = 'ridge', alpha = 0.5)
    plt.plot(X_reg,np.dot(X_reg,w), label='Ground Truth',c ='y')
    plt.title('Data')
    plt.ylim([m, M])
    plt.legend()
    plt.show()
else: 
    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(X_reg, y_reg, alpha = 0.4)
    ax[0].plot(X_reg, np.dot(X_reg,w1),'--', c='r')
    ax[1].scatter(X_reg,y_reg, alpha = 0.4)
    ax[1].plot(X_reg, np.dot(X_reg,w2),'--', c='r')
    plt.show()
wdiff = np.abs(w1-w2)
print(w1,w2,wdiff)

