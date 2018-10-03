# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 16:05:48 2018

@author: Pranjal Bhaskare
"""


from Data_Cleaning import Data
from keras.models import Sequential
from keras.layers import Dense

flags = [True, False]
imputer ='median'
avgs = [55, 60]

for flag in flags:
    for avg in avgs:
        X_train, y_train, X_test, person_add = Data(imputer,avg,fetu_sel=flag )

        print("Starting: ", imputer, avg, flag)

        # Initialising the ANN
        classifier = Sequential()

        # Adding the input layer and hidden layer
        classifier.add(Dense(output_dim =40, init = 'uniform', activation = 'relu', input_dim = X_train.shape[1]))

        classifier.add(Dense(output_dim = 40, init = 'uniform', activation = 'relu'))

        classifier.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu'))

        classifier.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu'))

        # classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'relu'))

        classifier.add(Dense(output_dim =10, init = 'uniform', activation = 'relu'))

        # classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

        classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

        # Compiling the ANN
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

        # Fitting the ANN to the Training set
        classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)


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
        with open('result/blablabla_IIT_Madras_ANN_' + imputer + str(avg) + str(flag) +'.csv', "w") as f:
            for i in range(len(person_add)):
                if y_pred[i]:
                    val = 1
                else:
                    val = 0
                f.write(str(person_add[i]) + "," + str(val) + "\n")






























