# Initial imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.decomposition import PCA

import datetime
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import boxcox

random.seed(3)

# Imports for better visualization
from matplotlib import rcParams
#colorbrewer2 Dark2 qualitative color table
dark2_colors = [(0.10588235294117647, 0.6196078431372549, 0.4666666666666667),
                (0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
                (0.4588235294117647, 0.4392156862745098, 0.7019607843137254),
                (0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
                (0.4, 0.6509803921568628, 0.11764705882352941),
                (0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
                (0.6509803921568628, 0.4627450980392157, 0.11372549019607843)]

rcParams['figure.figsize'] = (8, 3)
rcParams['figure.dpi'] = 150
rcParams['axes.color_cycle'] = dark2_colors
rcParams['lines.linewidth'] = 2
rcParams['font.size'] = 14
rcParams['patch.edgecolor'] = 'white'
rcParams['patch.facecolor'] = dark2_colors[0]
rcParams['font.family'] = 'StixGeneral'
rcParams['axes.grid'] = True
rcParams['axes.facecolor'] = '#eeeeee'


# Importing the data
train=pd.read_csv("Data/Training_Dataset.csv")
test=pd.read_csv("Data/Leaderboard_Dataset.csv")

data=pd.concat([train, test])

data["mvar3"][data.mvar3 == 0] = np.nan
data["mvar3"] = data.groupby("mvar2").mvar3.transform(lambda x: x.fillna(x.median()))

# Dataset contains many variables which are very skewed so those variables were normalized using box-cox transformations
# Upper cap is set to 99 percentile of its value
# Boosting model tend to perform poorly when values of single type occur many times in a single columns
# so to compensate that random noise was also added to continuous features.

skewed = ["mvar3", "mvar9",
          "mvar16", "mvar17", "mvar18", "mvar19",
          "mvar20", "mvar21", "mvar22", "mvar23",
          "mvar24", "mvar25", "mvar26", "mvar27",
          "mvar28", "mvar29", "mvar30", "mvar31",
          "mvar32", "mvar33", "mvar34", "mvar35",
          "mvar36", "mvar37", "mvar38", "mvar39"]


def normalizing(X):
    data[X][data[X] < 0] = 0
    data[X][data[X] > data[X].quantile(0.99)] = data[X].quantile(0.99)
    data[X] = data[X].apply(lambda x: x + np.random.rand())

    data[X] = data[X].apply(lambda x: x + 1)
    data[X], _ = boxcox(data[X])


for i in skewed:
    normalizing(i)


# Feature engineering

data['electronics'] =  data.mvar16 + data.mvar17 + data.mvar18 + data.mvar19
data['travel'] =  data.mvar21 + data.mvar22 + data.mvar23 + data.mvar20
data['household'] =  data.mvar25 + data.mvar26 + data.mvar27 + data.mvar24
data['car'] =  data.mvar29 + data.mvar30 + data.mvar31 + data.mvar28
data['retail'] =  data.mvar33 + data.mvar34 + data.mvar35 + data.mvar32
data['total'] =  data.mvar36 + data.mvar37 + data.mvar38 + data.mvar39

data['per_supp'] = data.mvar43/ data.mvar40
data['per_elite'] = data.mvar44/ data.mvar41
data['per_credit'] = data.mvar45/ data.mvar42


# For family size >= 4, supp card is preferred
data['mvar2_flag'] = 0
data['mvar2_flag'][data.mvar2 >= 4] = 1

# If Number of club memberships >= 2--> High chances of getting card
data['mvar14_flag'] = 0
data['mvar14_flag'][data.mvar14 >= 2] = 1

data['total_internal_score'] = data.mvar7 + data.mvar8 + data.mvar11/7
data['total_club_memberships'] =  data.mvar14 + data.mvar15
data['payement_per_card'] =  (data.mvar13*1.0)/data.mvar4
data['fees_per_club'] =  (data.mvar6*1.0)/data.mvar14
data['Income_per_member'] = (data.mvar9*1.0)/data.mvar2
data['spending_per_member'] =  (data.mvar3*1.0)/data.mvar2


def preparing_data(train, test, flag):
    if (flag == 'Credit'):
        data["Y"] = abs(data.mvar51)
        train["Y"] = abs(train.mvar51)

    if (flag == 'Elite'):
        data["Y"] = abs(data.mvar50)
        train["Y"] = abs(train.mvar50)

    if (flag == 'Supp'):
        data["Y"] = abs(data.mvar49)
        train["Y"] = abs(train.mvar49)

    if (flag == 'None'):
        data["Y"] = abs(data.mvar49 + data.mvar50 + data.mvar51 - 1)
        train["Y"] = abs(train.mvar49 + train.mvar50 + train.mvar51 - 1)

    # Since Family size and Industry code in which the customer has spent the most in past contained many labels
    # so dummy encoding would create too many variables (curse of dimensionality).
    # Therefore each of these were replaced with two new features

    categorical = ["mvar12", "mvar2"]

    def replacing_categorical(X):
        x = train.groupby(X)["Y"].mean()
        y = train.groupby(X)["Y"].std()
        data[X + str("_mean")] = data[X].apply(lambda X1: x[X1])
        data[X + str("_std")] = data[X].apply(lambda X2: y[X2])

    #     del data[X]

    for i in categorical:
        replacing_categorical(i)

    train_new = data.iloc[0:train.shape[0]]
    test_new = data.iloc[train.shape[0]:data.shape[0]]

    def prepare_data(df, is_train):
        # Dropping cm_key
        df = df.drop(["cm_key"], axis=1)
        # Dropping mvar1 due to uniformity
        df = df.drop(["mvar1"], axis=1)
        if is_train:
            return df.drop(['mvar12', "Y", "mvar46", "mvar47", "mvar48", "mvar49", "mvar50", "mvar51"], axis=1), df['Y']
        return df.drop(['mvar12', "Y", "mvar46", "mvar47", "mvar48", "mvar49", "mvar50", "mvar51"], axis=1)

    train_features, train_target = prepare_data(train_new, 1)
    test_features = prepare_data(test_new, 0)

    return train_features, train_target, test_features


import xgboost
from sklearn.cross_validation import train_test_split

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, precision_score, recall_score


def model(train, test, flag):
    train_features, train_target, test_features = preparing_data(train, test, flag)
    Xtrain, Xtest, ytrain, ytest = train_test_split(train_features, train_target,
                                                    stratify=train_target, test_size=.3, random_state=100)

    xgb = XGBClassifier(n_estimators=1000, seed=100)
    xgb.fit(train_features, train_target)

    print("Training :" + str(accuracy_score(ytrain, xgb.predict(Xtrain))))
    print("Test :" + str(accuracy_score(ytest, xgb.predict(Xtest))))
    print("ROC_AUC :" + str(roc_auc_score(ytest, xgb.predict(Xtest))))
    print("precision_score :" + str(precision_score(ytest, xgb.predict(Xtest))))
    print("recall_score :" + str(recall_score(ytest, xgb.predict(Xtest))))
    print("confusion_matrix :" + str(confusion_matrix(ytest, xgb.predict(Xtest))))

    def myscorer(cm):
        False1 = cm[0][1]
        False2 = cm[1][0]
        return (3 * False1) + (False2), (False1 + False2)

    print("myscorer :" + str(myscorer(confusion_matrix(ytest, xgb.predict(Xtest)))))

    ax = xgboost.plot_importance(xgb)
    fig = ax.figure
    fig.set_size_inches(15, 15)

model(train,test,'Supp')

# Training on Supp data
train_features, train_target, test_features = preparing_data(train, test, "Supp")
xgb = XGBClassifier(n_estimators=1000, seed=100)
xgb.fit(train_features, train_target)
submission = pd.concat(
    [test["cm_key"], pd.DataFrame(xgb.predict_proba(test_features), columns=["Supp_NO", "Supp_Yes"])], axis=1)
submission.to_csv("Supp.csv", index=False)

# Training on Elite data
train_features, train_target, test_features = preparing_data(train,test, "Elite")
xgb = XGBClassifier(n_estimators= 1000, seed=100)
xgb.fit(train_features, train_target)
submission = pd.concat([test["cm_key"], pd.DataFrame(xgb.predict_proba(test_features), columns=["Elite_NO","Elite_Yes"])], axis=1)
submission.to_csv("Elite.csv", index=False)

# Training on credit data
train_features, train_target, test_features = preparing_data(train,test, "Credit")
xgb = XGBClassifier(n_estimators= 1000, seed=100)
xgb.fit(train_features, train_target)
submission = pd.concat([test["cm_key"], pd.DataFrame(xgb.predict_proba(test_features), columns=["Credit_NO","Credit_Yes"])], axis=1)
submission.to_csv("Credit.csv", index=False)

# Training on None  class data
train_features, train_target, test_features = preparing_data(train,test, "None")
xgb = XGBClassifier(n_estimators= 1000, seed=100)
xgb.fit(train_features, train_target)
submission = pd.concat([test["cm_key"], pd.DataFrame(xgb.predict_proba(test_features), columns=["None_NO","None_Yes"])], axis=1)
submission.to_csv("NOne.csv", index=False)

Supp = pd.read_csv("Supp.csv")
Credit = pd.read_csv("Credit.csv")
Elite = pd.read_csv("Elite.csv")
NOne = pd.read_csv("NOne.csv")