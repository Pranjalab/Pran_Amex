from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

def Data(imputer='median', min_avg=55, remove_null=True, fetu_sel=False):

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
    NullMask, mask = None, None
    if remove_null:
        print("Removing Null Values...")
        Null_Avg = []
        TN = X_train.shape[0]
        for i in range(X_train.shape[1]):
            val = pd.isnull(X_train[:, i]).sum()
            avg = (val/TN)*100
            Null_Avg.append(avg)
        j = 0
        NullMask = [True]*len(Null_Avg)
        for i in range(len(Null_Avg)):
            if Null_Avg[i] > min_avg:
                NullMask[i] = False
                X_train = np.delete(X_train, i - j, 1)
                X_test = np.delete(X_test, i - j, 1)
                j += 1

        j = 0
        no = X_train.shape[1]
        for i in range(len(X_train[0])):
            val = pd.isnull(X_train[i,:]).sum()
            avg = (val/no)*100
            if avg > 40:
                X_train = np.delete(X_train, i - j, 0)
                y_train = np.delete(y_train, i - j, 0)
                j += 1

    print(X_train.shape)

    # Missing Data
    imputer1 = Imputer(missing_values='NaN', strategy=imputer, axis=0)
    imputer2 = Imputer(missing_values='NaN', strategy=imputer, axis=0)
    imputer1 = imputer1.fit(X_train)
    imputer2 = imputer2.fit(X_test)
    X_train = imputer1.transform(X_train)
    X_test = imputer1.transform(X_test)

    if fetu_sel:
        print("Feature Selection...")
        select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='median')
        select.fit(X_train, y_train)
        print('The shape of old X_train is: ', X_train.shape)
        X_train = select.transform(X_train)
        print('The shape of new X_train is ', X_train.shape)

        mask = select.get_support()
        plt.matshow(mask.reshape(1, -1), cmap='gray_r')
        plt.xlabel('Index of Features')
        print(mask)

        X_test = select.transform(X_test)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # with open("Feture/f_ReNUll_" + str(remove_null) + "Fet_" + str(fetu_sel) + ".txt", "w") as file:
    #     file.write("Null Mask\n\n")
    #     if remove_null:
    #         for n in NullMask:
    #             file.write(str(n) + "\n")
    #     file.write('\n\nfet mask')
    #     if fetu_sel:
    #         for m in mask:
    #             file.write(str(m) + "\n")

    return X_train, y_train, X_test, person_add