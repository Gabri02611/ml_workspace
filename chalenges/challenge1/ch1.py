import os 
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.decomposition import PCA
from sklearn.cluster import k_means
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy.linalg as LA
from numpy.linalg import inv
data = open('data_banknote_authentication.txt')
datanp = pd.DataFrame(data)
#datasp = datanp[].__str__.split(',',expand = True)
print(datanp.describe())
datafp = datanp[0].str.split(',',expand = True)
print(datafp.())
