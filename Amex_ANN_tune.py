# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 16:05:48 2018

@author: Pranjal Bhaskare
"""

# Importing the libraries
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.activations import relu, sigmoid
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

# Importing the train dataset
dataset = pd.read_csv('Training_dataset_Original_.csv')
X_train = dataset.iloc[:, 1:48].values
y_train = dataset.iloc[:, 48].values

# Importing the test Dataset
test_dataset = pd.read_csv('Leaderboard_dataset_.csv')
X_test = test_dataset.iloc[:, 1:48].values
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


# import keras.backend as K

def create_model(layers, activation_fun, opt, los):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i == 0:
            model.add(Dense(nodes, input_dim=47))
            model.add(Activation(activation_fun))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation_fun))
    model.add(Dense(1, activation = 'sigmoid'))  # Note: no activation beyond this point

    model.compile(optimizer=opt, loss=los, metrics = ['accuracy'])
    return model


model = KerasRegressor(build_fn=create_model, verbose=1)

#TODO : try dropouts
layers = [[24], [32, 16], [36, 24, 12]]
activations = [sigmoid, relu]
opt = ['adam', 'adadelta']
los = ['binary_crossentropy', 'mse']
param_grid = dict(layers=layers, activation_fun=activations,opt=opt, los=los, batch_size = [10, 50, 100], epochs=[50, 100, 200])
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error')

grid_result = grid.fit(X_train, y_train)

i = 0
print([grid_result.best_score_, grid_result.best_params_])

with open('Tuning_result.txt', "w") as file:
    file.write("Best Score: " + str(grid_result.best_score_) + "\n")
    file.write("Best Params: " + str(grid_result.best_params_) + "\n\n\n")
    for params, mean_score, scores in grid_result.grid_scores_:
        i += 1
        file.write(str(i) + " " + str(scores.mean()) + " " + str(scores.std()) + ' ' + str(params) + "\n")


# Predicting the Test set results
y_pred = grid.predict(X_test)
y_pred = (y_pred > 0.5)

# Saving the result
with open('blablabla_IIT_Madras_7.csv', "w") as f:
    for i in range(len(person_add)):
        f.write(str(person_add[i]) + "," + str(y_pred[i]) + "\n")


