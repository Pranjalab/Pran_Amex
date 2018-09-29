from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def Data(imputer='mean', min_avg=45):

    # Importing the train dataset
    dataset = pd.read_csv('Training_dataset_Original_.csv')
    X_train = dataset.iloc[:, 1:48].values
    y_train = dataset.iloc[:, 48].values
    test_dataset = pd.read_csv('Leaderboard_dataset_.csv')
    X_test = test_dataset.iloc[:, 1:48].values
    person_add = test_dataset.iloc[:, 0].values

    # Encoding categorical data
    labelencoder_X_1 = LabelEncoder()
    X_train[:, -1] = labelencoder_X_1.fit_transform(X_train[:, -1])

    labelencoder_X_2 = LabelEncoder()
    X_test[:, -1] = labelencoder_X_2.fit_transform(X_test[:, -1])

    Null_Avg = []
    TN = X_train.shape[0]
    for i in range(X_train.shape[1]):
        val = pd.isnull(X_train[:, i]).sum()
        avg = (val/TN)*100
        Null_Avg.append(avg)
    j = 0
    for i in range(len(Null_Avg)):
        if Null_Avg[i] > min_avg:
            X_train = np.delete(X_train, i - j, 1)
            X_test = np.delete(X_test, i - j, 1)
            j += 1

    # Missing Data
    imputer1 = Imputer(missing_values='NaN', strategy=imputer, axis=0)
    imputer2 = Imputer(missing_values='NaN', strategy=imputer, axis=0)
    imputer1 = imputer1.fit(X_train)
    imputer2 = imputer2.fit(X_test)
    X_train = imputer1.transform(X_train)
    X_test = imputer1.transform(X_test)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, y_train, X_test, person_add