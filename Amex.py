# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 16:05:48 2018

@author: Pranjal Bhaskare
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the train dataset
dataset = pd.read_csv('Training_dataset_Original_.csv')
X_train = dataset.iloc[:, 1:48].values
y_train = dataset.iloc[:, 48].values

# Importing the test Dataset
test_dataset = pd.read_csv('Leaderboard_dataset_.csv')
X_test = test_dataset.iloc[:, 1:48].values
# y_test = test_dataset.iloc[:, 48].values
person_add =  test_dataset.iloc[:, 0].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X_train[:, -1] = labelencoder_X_1.fit_transform(X_train[:, -1])

labelencoder_X_2 = LabelEncoder()
X_test[:, -1] = labelencoder_X_2.fit_transform(X_test[:, -1])

# Missing Data
from sklearn.preprocessing import Imputer
imputer1 = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer2 = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer1 = imputer1.fit(X_train)
imputer2 = imputer2.fit(X_test)
X_train = imputer1.transform(X_train)
X_test = imputer1.transform(X_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages

import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and hidden layer
classifier.add(Dense(output_dim = 24, init = 'uniform', activation = 'relu', input_dim = 47))

classifier.add(Dense(output_dim = 24, init = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = 24, init = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'relu'))

# classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 500)


# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
# acc = (cm[0][0] + cm[1][1])/8000
# print(acc)

# Saving the result
with open('blablabla_IIT_Madras_1.csv', "w") as f:
    for i in range(len(person_add)):
        f.write(str(person_add[i]) + "," + str(y_pred[i]) + "\n")
































