# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def PCA_Data(remove_null=False, imputer='median'):
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
            if Null_Avg[i] > 55:
                NullMask[i] = False
                X_train = np.delete(X_train, i - j, 1)
                X_test = np.delete(X_test, i - j, 1)
                j += 1

        # j = 0
        # no = X_train.shape[1]
        # for i in range(len(X_train[0])):
        #     val = pd.isnull(X_train[i,:]).sum()
        #     avg = (val/no)*100
        #     if avg > 50:
        #         X_train = np.delete(X_train, i - j, 0)
        #         y_train = np.delete(y_train, i - j, 0)
        #         j += 1

    print(X_train.shape)

    # Missing Data
    imputer1 = Imputer(missing_values='NaN', strategy=imputer, axis=0)
    imputer2 = Imputer(missing_values='NaN', strategy=imputer, axis=0)
    imputer1 = imputer1.fit(X_train)
    imputer2 = imputer2.fit(X_test)
    X_train = imputer1.transform(X_train)
    X_test = imputer2.transform(X_test)

    # # Applying PCA
    # pca = PCA(n_components=3)
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)

    # Applying LDA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    lda = LDA(n_components=20)
    X_train = lda.fit_transform(X_train, y_train)
    X_test = lda.transform(X_test)

    # # Applying Kernel PCA
    # from sklearn.decomposition import KernelPCA
    # kpca = KernelPCA(n_components=2, kernel='rbf')
    # X_train = kpca.fit_transform(X_train)
    # X_test = kpca.transform(X_test)

    # explained_variance = pca.explained_variance_ratio_
    # for i in range(len(explained_variance)):
    #     print(i, round(float(explained_variance[i]),10))

    print('Shape of X_train: ', X_train.shape)

    return X_train, y_train, X_test, person_add


# PCA_Data()