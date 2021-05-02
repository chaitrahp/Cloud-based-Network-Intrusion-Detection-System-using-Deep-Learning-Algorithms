#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from keras.models import Sequential
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.utils import to_categorical
from tensorflow.python.keras.metrics import Metric
from sklearn.utils import class_weight


# In[2]:


trainset = pd.read_csv('train.csv')
testset = pd.read_csv('test.csv')


# In[3]:


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

#del trainset['id']
#del testset['id']
cols=[]
for i in range(0,47): #47 : attack cat, 48: label
	cols.append(i)

cols.remove(0)
cols.remove(3)
cols.remove(2)
cols.remove(28)
cols.remove(29)

cols.remove(39)
#cols.append(48)
#cols.remove(3)
#cols.remove(1)
'''
x_train = trainset.iloc[:5000,cols]
y_train = trainset.iloc[:5000,48]

#print(x_train[0])
for i in x_train:
	print(i)

x_test = trainset.iloc[5000:7000,cols]
y_test = trainset.iloc[5000:7000,48]
'''

x_train = trainset.iloc[:,cols]
y_train = trainset.iloc[:,48]

x_test = testset.iloc[:,cols]
y_test = testset.iloc[:,48]

x_test = x_test.fillna(x_train.mean())
print(x_test.iloc[0:1,:]) #49 columns


corr = x_train.corr()

#print(corr)


'''columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False

columnsToUse = x_train.columns[columns] 

print(columnsToUse)
x_train = x_train[columnsToUse]
x_test = x_test[columnsToUse]
'''

X_train=np.array(x_train)
X_test=np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

X_train = np.reshape(X_train, (X_train.shape[0],1,  X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0],1, X_test.shape[1]))
np.where(np.isnan(X_test))


# In[6]:


batch_size = 32



# 1. define the network
model = Sequential()
model.add(LSTM(64))

# ----- 5. DENSE HIDDEN LAYER
model.add(Dense(64, activation="relu"))

# ----- 6. OUTPUT
model.add(Dense(1, activation="sigmoid"))

#print(model.get_config())


# try using different optimizers and different optimizer configs
#

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


#model.compile(loss='binary_crossentropy',optimizer='adam',metrics=[keras.metrics.Recall()])

weights = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)
#checkpointer = callbacks.ModelCheckpoint(filepath="kddresults/lstm1layer/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='val_acc',mode='max')
#csv_logger = CSVLogger('training_set_iranalysis.csv',separator=',', append=False)
model.fit(X_train, y_train, batch_size=batch_size, epochs=50, class_weight=weights) #,callbacks=[checkpointer,csv_logger])
#smodel.save("kddresults/lstm1layer/fullmodel/lstm1layer_model.hdf5")


# In[13]:


#fixing matrix of predictions:
weighted_test = model.predict(X_test)
for i in range(0,len(weighted_test)):
    for x in range(0,len(weighted_test[i])):
        if weighted_test[i][x]>=0.4:
            weighted_test[i][x]=1


# In[14]:


cm = (confusion_matrix(y_test,(weighted_test.round())))
#cm = np.argmax(cm,axis=1)
print("Confusion Matrix:\n\n",cm)  # 00:TN 01:FP 10:FN 11:TP


# In[15]:


def precision(cm):
    p = (cm[1][1]/((cm[0][1])+(cm[1][1]))) #TP/TP+FP
    if (str(p) == "nan"):
        print("Precision: ","0.00")
    else:
        print("Precision– ",p)
        
#do same for accuracy, recall and f1 score        
precision(cm)


# In[16]:


loss, accuracy = model.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))


# In[ ]:




