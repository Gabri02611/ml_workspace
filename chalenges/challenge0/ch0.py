import os 
import numpy as np
import sys
print(sys.executable)
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import numpy.linalg as LA
from numpy.linalg import inv


#carico i dati tramite pandas
data = pd.read_csv('50_Startups.csv')

#con iloc creo un sottoinsieme di un dataset
#in questo caso sto prendendo tutte le colone tranne le ultime due 
features = data.iloc [:,:-2].values

#le label che cerco riguardano i nomi degli stati nella terza colonna
labels = data.iloc[:,3].values

#se voglio lavorare con un dataframe piuttosto che una matrice posso farlo 
#con il seguente comando
df = pd.DataFrame(data)

#voglio sapere la dimenisone del dataset utilizzo
df.shape
#questo comando restituisc euna tupla con 
# 1. il numero di righe nel dataset 
# 2. il numero di colonne del dataset

#nel dataset sono presenti dei dati mancanti ed è necessario modificare il dataset
#in modo che questi dati non interferiscano con la creazione del modello
#posso eliminare le righe con i dati mancanti o rimpiazzarli 
#utilizzo la funzione pandas replace per rimpiazzare gli 0.00 con la media
df.replace(to_replace = 0.00, value = df.mean(axis = 0, numeric_only = True), inplace = True)

#con .head() stampo le prime x righe del dataset, di default = 5
df.head(20)

#con .tail stampo invece le ultime x righe del dataset
df.tail()

#la challenge riguarda un problema di classificazione binaria, seleziono solo due stati come feature
df_sel = df[(df.State=="California") | (df.State == "Florida")]


#però gli algoritmi di machine learning utilizzano variabili numeriche non categoriche com e in questo caso
#per ovviare a questo problema si possono utilizzare dei metodi di encoding per rappresentare queti tipi di dati
#il più semplice è il one hot encoding che viene richiamato attraverso il seguente comando:
df_one = pd.get_dummies(df_sel["State"],dtype = int)

#con questo comando creo un dataframe di due colonne in cui ogni un acolonna avrà 1 per california e 0 per florida,
#vieceversa l'altra


#costruisco il dataset finale per il training

#1. concateno i due data frame  "df_one " e "df_sel" in un solo dataframe 
### axis = 1 indica che i dataframe vengono aggiunti come colonne
df_fin = pd.concat((df_one,df_sel), axis = 1)

#2. elimino le colonne "florida" in quanto dopo l'encoding posso mappare con 1
#le righe relative a california e 0 quelle relative a florida

df_fin = df_fin.drop(("Florida"), axis = 1)

#3. elimino la colonna "state" in quanto ho già l'encoding necessario  dato da df_one

df_fin = df_fin.drop(("State"), axis = 1)

#4.rinomino la colonna encoded California con state in modo da ripristinare la formattazione
#oriignale del daatset (5 colonne di cui un "State")

df_fin = df_fin.rename(columns = {"California" : "State"})


df_fin.head()


####NORMALIZZAZIONE
#normalizzo per evitare che ci siano problemi di scala tra i numeri del dataset

#normalizzo dividendo i valori di ogni colonna nel dataframe "df_fin" per il massimo valore assoluto della colonna
def absolute_maximum_scale(series):
    return series / series.abs().max()

#applico la mia funzione
for col in df_fin.columns:
    df_fin[col] = absolute_maximum_scale(df_fin[col])

#ATTENZIONE così facendo scalero anche la colonna "State", la quale però essendo composta da soloi 0 e 1 non cambierà di valore 
#ma asserà da int a float

df_fin.head()
df_fin.shape

####CLASSIFICAZIONE

#preparo il dataset
#salvo in y la colonna "State"
y = df_fin["State"]

#salvo in x tutto il resto 
x = df_fin.drop(["State"],axis=1)
#voglio lavorare con un vettore numpy invece di una serie pandas

y = y.values
x = x.values

#suddivido il dataset in un inseme di train e un insieme di test

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state =  0)
#print(x_train)
#print(y_train)
#i questa challenge devo usare il metodo di Regressione Logistica:

#con random_state che funge da seed per la regressione e il solver che indica il risolutore
#il comando fit addestra il modello di supervised learning
LR = LogisticRegression(random_state=0, solver='liblinear', penalty = 'l1').fit(x_train, y_train)
print(type(LR))
#calcolo le mie previsioni
predictions = LR.predict(x_test)
print("predicitons: ", predictions)
# Calculating and rounding the accuracy score of the Logistic Regression model on the test set.
#calcolo la accuracy del modello di regressione logistica sul set di test e lo arrotondo a 4 cifre decimali
#la funzione punteggio è calcolata confrontando i valori stimati con i valori reali (y_test)
# The score is calculated by comparing the predicted values to the actual values (y_test).
accuracy = round(LR.score(x_test, y_test), 4)
print(accuracy)

###CHALLENGE

#PLOT DEI RISULTATI
plot = 0
if plot == 1:
    dl = {"Florida","California", "California", "Florida"}
    fig = ConfusionMatrixDisplay.from_predictions(y_test,predictions,display_labels = dl)
#con descrizione degli assi e legenda

#con commento sensato relativo al plot 
##...
#MODELLO DI REGRESSIONE LOGISTICA + REGOLARIZZAZIONI

#genero i pesi della retta di regressione
def mycost(y,y_pred):
    return(np.mean(((y-y_pred))**2))
model_scrauso = 1
if model_scrauso == 1:
        def Loss(h,y):
            m = len(y)
            return -(1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))  

        def sigmoid(x):
            return (1/(1+np.exp(-x)))


        def fits(x,y,l_r = 0.01,it_max = 1000):
            #faccio il fitting al modello utilizzando gradient descent
            it = 0
            m,n = np.shape(x)
            weights = np.zeros(n)
            bias = 0 
            cost_history = []
            for i in range(it_max):
                z = np.dot(x,weights) + bias
                h = sigmoid(z)
                der_weights = (1/m) * np.dot(x.T,(h-y))
                der_bias = (1/m) * np.sum(h - y)
                weights -= l_r * der_weights
                bias -= l_r * der_bias
                cost_history.append(Loss(h,y))
            #print (cost_history)
            return weights, bias, cost_history
        
        def predict(x,weights,bias):
            return(sigmoid(np.dot(x,weights) + bias) >= 0.3).astype(int)
        def Ridge_regr(x,y,w,lam):
            return 
        predictions1 = predict(x_test,fits(x_train,y_train)[0],fits(x_train,y_train)[1])
        accuracy1 = np.mean(predictions1 == y_test)
        accuracy = np.mean(predictions == y_test)
        print(f"Model Accuracy: {accuracy1:.2f}")
        print(f"Model Accuracy: {accuracy:.2f}")

        
        plt.grid(True)
        plt.xlabel("Iterations")
        plt.ylabel("Cost")

fig,ax = plt.subplots(1,2)
ax[0].plot(fits(x_train,y_train)[2])
plt.grid(True)
ax[1].plot(fits(x_train,y_train)[2])
plt.grid(True)
plt.show()
#classification_report fornisce informazioni sulle principali grandezze necessarie per stilare il report
y_true = y_test
y_pred = LR.predict(x_test)
target_names = ['California','Florida']
print(classification_report(y_true,y_pred,target_names=target_names))

#IMPLEMENTAZIONE DELLA ROC CURVE PLOT + SPIEGAZIONE, anche con risultati non ottimali l'importante è che siano giustificati

