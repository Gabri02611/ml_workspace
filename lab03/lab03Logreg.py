import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

def LogisticLoss(labels,X,w):
    return -labels*np.log(sigmoid(X,w)) - (1-labels)*np.log(1-sigmoid(X,w))
def sigmoid(X,w):
     return 1/(1 + np.exp(-np.dot(X, w)))
def LogisticGD(X,labels,iter,gamma):
    d = np.shape(X)
    cost = np.zeros(iter)
    w = np.random.normal(0,0.01,d[1])
    W = np.zeros((iter,2))
    for i in range(iter):
        W[i] = w
        w = w - (gamma/d[0])*X.T @ (sigmoid(X,w)-labels)
        cost[i] = np.mean(LogisticLoss(labels,X,w))
    return W,cost, i
def mixGauss(means, sigmas, n):
    means = np.array(means)
    sigmas = np.array(sigmas)

    d = np.shape(means)[1]
    num_classes = sigmas.size

    data = np.full((n * num_classes, d), np.inf)
    labels = np.zeros(n * num_classes)

    for idx, sigma in enumerate(sigmas):
        data[idx * n: (idx + 1)* n] = np.random.multivariate_normal(mean = means[idx], cov = np.eye(d) * sigmas[idx] ** 2, size = n)
        labels [idx * n: (idx + 1) * n] = idx
    if(num_classes == 2):
            labels[labels == 0] = -1
    return data,labels
def labelsnoise(perc, labels):
    """
    Parameters
    ----------
    perc : float
        Percentage of labels to be flipped
    labels: array of int of dim n_classes
        Array containing labels idxs
    """
    points = np.shape(labels)[0]
    noisylabels = np.copy(np.squeeze(labels))
    n_flips = int(np.floor(points * perc / 100)) # floor: nearest integer by defect
    idx_to_flip = np.random.choice(points, size=n_flips, replace=False) # replace is false since the same index cannot be chosen twice
    noisylabels[idx_to_flip] = -noisylabels[idx_to_flip] # for binary this turns -1 into 1 and viceversa
    return noisylabels
means = [[3,0],[0,6]]
sigmas = [0.9,0.9]
iter = 100
gamma = 0.005
n = 100
X, labels = mixGauss(means,sigmas, n)
noisy_labels = labelsnoise(10,labels)
fig, ax = plt.subplots()
plt.scatter(X[:,0], X[:,1], s=70, c= noisy_labels, alpha=0.8)
plt.show()
labels_01 = np.where(noisy_labels == -1, 0, 1)
W, cost, i = LogisticGD(X, noisy_labels, iter, gamma)
plt.plot(cost, 'go', markersize=0.5)
plt.title('Loss')
plt.show()