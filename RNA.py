
#%% Dependencies
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt

#%% Import Dataset
dataset = pd.read_excel('divorce.xlsx') #You need to change #directory accordingly
# dataset.head(10) #Return 10 rows of data
X = dataset.iloc[:, 1:54]
Y = dataset[['Class']]

#%% Normalizing the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#%% Splitting data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.6)

#%% Import resourcers for the neural network 
from keras.models import Sequential
from keras.layers import Dense

#%% Setting up the NN
model = Sequential()
model.add(Dense(10, input_dim=53, activation='sigmoid'))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=20, batch_size=len(X_train))

#%% Check performance
y_pred = model.predict(X_test)

#Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))

# No need to convert 'one hot encoded' test label to label
test = y_test
#for i in range(len(y_test)):
#    test.append(np.argmax(y_test[i]))

from sklearn.metrics import accuracy_score
a = accuracy_score(pred,test)
print('Accuracy is:', a*100)

#%% Plotting Accuracy
history = model.fit(X_train, y_train,validation_data = (X_test,y_test), epochs=20, batch_size=len(X_train))

import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#%% Plotting Loss
plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.title('Model loss') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()