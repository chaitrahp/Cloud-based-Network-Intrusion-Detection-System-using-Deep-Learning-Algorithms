import pandas as pd  
import numpy as np  
#import statsmodels.api as sm
import sklearn.metrics 
import seaborn as sbs
import pickle
import matplotlib.pyplot as plt  
from keras.models import Sequential
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
import csv
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
#import statsmodels.formula.api as smf
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from pandas import DataFrame
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from keras.layers import Convolution1D,MaxPooling1D, Flatten


trainset = pd.read_csv('UNSW_NB15_training-set.csv')
testset = pd.read_csv('UNSW_NB15_testing-set.csv')
label_encoder = LabelEncoder()
proto = list(trainset['proto'])
trainset['proto'] = label_encoder.fit_transform(proto)

service = list(trainset['service'])
trainset['service'] = label_encoder.fit_transform(service)

state = list(trainset['state'])
trainset['state'] = label_encoder.fit_transform(state)

attack_cat = list(trainset['attack_cat'])
trainset['attack_cat'] = label_encoder.fit_transform(attack_cat)

proto = list(testset['proto'])
testset['proto'] = label_encoder.fit_transform(proto)

service = list(testset['service'])
testset['service'] = label_encoder.fit_transform(service)

state = list(testset['state'])
testset['state'] = label_encoder.fit_transform(state)

attack_cat = list(testset['attack_cat'])
testset['attack_cat'] = label_encoder.fit_transform(attack_cat)

del trainset['id']
del testset['id']
xtrain = trainset.iloc[:,0:42]
ytrain = trainset.iloc[:,43]

xtest = testset.iloc[:,0:42]
ytest = testset.iloc[:,43]

corr = xtrain.corr()
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False

columnsToUse = xtrain.columns[columns] 
xtrain = xtrain[columnsToUse]
xtest = xtest[columnsToUse]
X_train=np.array(xtrain)
X_test=np.array(xtest)
ytrain = np.array(ytrain)
ytest = np.array(ytest)


#print(X_test.shape[1])

# reshape input to be [samples, time steps, features]
xtrain = np.reshape(X_train, (X_train.shape[0],1,  X_train.shape[1]))
xtest = np.reshape(X_test, (X_test.shape[0],1, X_test.shape[1]))



batch_size = 32

model = Sequential()
model = Sequential()
model.add(Convolution1D(64, 3, padding="same",activation="relu",input_dim=29))
model.add(MaxPooling1D(pool_size=(2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))
print(model.summary())

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#checkpointer = callbacks.ModelCheckpoint(filepath="kddresults/lstm1layer/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='val_acc',mode='max')
#csv_logger = CSVLogger('training_set_iranalysis.csv',separator=',', append=False)

model.fit(xtrain, ytrain, batch_size=batch_size, epochs=5) #,callbacks=[checkpointer,csv_logger])
#smodel.save("kddresults/lstm1layer/fullmodel/lstm1layer_model.hdf5")

loss, accuracy = model.evaluate(xtest, ytest)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

pred = model.predict_classes(xtest)
proba = model.predict_proba(xtest)
accuracy = accuracy_score(ytest, pred)

recall = recall_score(ytest, pred , average="binary")
precision = precision_score(ytest, pred , average="binary")
f1 = f1_score(ytest, pred, average="binary")

print("----------------------------------------------")
print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("recall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)
