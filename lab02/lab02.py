import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from random import randint
from math import pi as PI

np.random.seed(42)
#deifnisco il numero di ossevazioni
NUM_OBS = 1000
#genero un array per inserire i valori della mia predizione
x = np.linspace(0,2,num = NUM_OBS)
# genero del noise attraverso una distribuzione normale
eps = np.random.normal(0,0.1, NUM_OBS)
#genero i dati con il noise
y = np.sin(PI * x) + eps
def mse(y_true,y_pred):
    return np.mean((y_true- y_pred)**2)
def test_param_range(x_train, y_train, x_test, y_test, m_min, m_max):
    mse_train = []
    mse_test  = []
    
    # per ogni configurazione di parametri calcolo mse su train e test
    # salvo tutto in una lista

    for m in range(m_min, m_max):
        #creo il fit dei dati con un polinomio di grado m (ottimizzato)
        model = np.polyfit(x_train, y_train, deg = m)
        #valuto il fit nei punti del dataset di training
        y_train_pred = np.polyval(model, x_train)
        #calcolo l'errore quadrtico medio tra i valori reali e quelli forniti dal modello
        mse_train.append(mse(y_train, y_train_pred))
        #ripeto la procedura sul dataset di test 
        y_test_pred = np.polyval(model, x_test)
        mse_test.append(mse(y_test, y_test_pred))

        
    return mse_train, mse_test
#divido il dataset in training e test
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 0)
#calcolo l'mse per il dataset di train e quello di test
mse_train, mse_test = test_param_range(x_train, y_train, x_test, y_test, 2, 8)
ms = [i for i in range(2,8)]
##plt.plot(ms, mse_train, ".-", label = "MSE Train")
#plt.plot(ms, mse_test, ".-", label = "MSE Test")
#creo la finestra del plot
fig = plt.figure(figsize=(7,7))
plt.title('y = sin(pix) + eps')
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(x_train,y_train, label = 'train_set', alpha = 0.6)
plt.scatter(x_test,y_test, label = 'test_set', alpha = 0.6)

#plt.show()


#plt.plot(ms, mse_train, ".-", label = "MSE Train")
#plt.plot(ms, mse_test, ".-", label = "MSE Test")
plt.legend()
plt.show()

print(np.size(mse))