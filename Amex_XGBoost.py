# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 16:05:48 2018

@author: Pranjal Bhaskare
"""

from Data_Cleaning import Data
from xgboost import XGBClassifier


flag = [[True, False]]
imputer = 'median'
avg = 55
for f in flag:
    X_train, y_train, X_test, person_add = Data(remove_null=f[0], fetu_sel=f[1])
    print("Starting: ", imputer, avg)

    classifier = XGBClassifier()
    classifier.fit(X_train, y_train)


    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)


    # Saving the result
    with open('result/blablabla_IIT_Madras_XG_' + imputer + str(avg) + str(f[0]) + str(f[1]) +'_1.csv', "w") as f:
        for i in range(len(person_add)):
            if y_pred[i]:
                val = 1
            else:
                val = 0
            f.write(str(person_add[i]) + "," + str(val) + "\n")
































