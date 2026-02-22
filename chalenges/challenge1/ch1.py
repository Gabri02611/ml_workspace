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

#carico il dataset e lo trasformo in un datframe
data = open('data_banknote_authentication.txt')
datanp = pd.DataFrame(data)

#utilizzo il delimitatore "," per dividere le colonne e rinomino gli indici di colonne con i loro nomi
datafp = pd.DataFrame(datanp[0].str.split(',',expand = True))
datafp = datafp.rename(columns={0 :"Variance", 1:"Skewness", 2 : "Curtosis", 3 : "Entropy", 4 : "Class"})
#raccolgo i dati che utilizzerò in una matrice
X = datafp.iloc[:,:-1].values
N = X.shape[0] #salvo il numero di punti di data



print(X)


