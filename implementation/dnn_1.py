import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
from keras.utils import to_categorical
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from keras.models import Sequential
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense,Input, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from tensorflow.python.keras.metrics import Metric
from sklearn.utils import class_weight

trainset = pd.read_csv('train.csv')
testset = pd.read_csv('test.csv')

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

# X_train = np.reshape(X_train, (X_train.shape[0],1,  X_train.shape[1]))
# X_test = np.reshape(X_test, (X_test.shape[0],1, X_test.shape[1]))



#np.where(np.isnan(X_test))


batch_size = 1024
'''
ins = Input(shape = 41)
#x = SimpleRNN(2048)(ins)
x = Dense(4096)(ins)
x = Activation('relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(2048)(x)
x = Activation('relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(1024)(x)
x = Activation('relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(512)(x)
x = Activation('relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256)(x)
x = Activation('relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
outs = Dense(1, activation = 'sigmoid')(x)

model = Model(ins, outs)
'''

# 1. define the network
model = Sequential()
model.add(Dense(512,input_dim=41,activation='relu'))   
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(1024,activation='relu'))  
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


#print(model.get_config())


# try using different optimizers and different optimizer configs
#

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


#model.compile(loss='binary_crossentropy',optimizer='adam',metrics=[keras.metrics.Recall()])

weights = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)
#checkpointer = callbacks.ModelCheckpoint(filepath="kddresults/lstm1layer/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='val_acc',mode='max')
#csv_logger = CSVLogger('training_set_iranalysis.csv',separator=',', append=False)
model.fit(X_train, y_train, batch_size=batch_size, epochs=25)#, class_weight=weights) #,callbacks=[checkpointer,csv_logger])
model.save("./dnnlayer_model.hdf5")



loss, accuracy = model.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

weighted_test = model.predict(X_test)
for i in range(0,len(weighted_test)):
    for x in range(0,len(weighted_test[i])):
        if weighted_test[i][x]>=0.4:
            weighted_test[i][x]=1

cm = (confusion_matrix(y_test,(weighted_test.round())))
#cm = np.argmax(cm,axis=1)
print("Confusion Matrix:\n\n",cm) 

def precision(cm):
    p = (cm[1][1]/((cm[0][1])+(cm[1][1]))) #TP/TP+FP
    if (str(p) == "nan"):
        print("Precision: ","0.00")
    else:
        print("Precision– ",p)
        
#do same for accuracy, recall and f1 score        
precision(cm)

