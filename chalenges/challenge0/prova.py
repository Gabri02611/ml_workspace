import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import numpy.linalg as LA
from numpy.linalg import inv

def sigmoid(x,w):
    y = 1/(1+np.exp(-np.dot(x,w)))
    return y

x = np.linspace(-3,3,100)
w = 3
w1 = 2
w2 =1*np.exp()
y = sigmoid(x,w)
y1 = sigmoid (x,w1)
y2 = sigmoid (x,w2)
fig, ax = plt.subplots()
ax.plot(x, y, label='sigmoid function w ')
ax.plot(x, y1, label='sigmoid function w1 ')
ax.plot(x, y2, label='sigmoid function w2 ')
plt.legend()
plt.show()