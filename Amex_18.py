# Initial imports
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import boxcox
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

Remove_NaN = True

# Importing the data
train=pd.read_csv("Data/Training_dataset_Original_.csv")
test=pd.read_csv("Data/Leaderboard_dataset_.csv")

data=pd.concat([train, test])

# Encoding categorical data
labelencoder = LabelEncoder()
train['mvar47'] = labelencoder.fit_transform(train['mvar47'])
test['mvar47'] = labelencoder.transform(test['mvar47'])

# Removing columns which has more then 60% of NaN value
Removed_feature = []
if Remove_NaN:
    Total_feature = test.axes[1]
    print("Removing Null Values...")
    TN = data.shape[0]
    for feature in Total_feature:
        val = pd.isnull(data[feature]).sum()
        avg = (val / TN) * 100
        if avg > 60:
            Removed_feature.append(feature)
            train = train.drop(feature, axis=1)
            test = test.drop(feature, axis=1)
    print("Shape of new train and test: ", train.shape, test.shape)
    print("Feature removed are: ", str(Removed_feature))

# Imputing Missing Data
#test.fillna(test.median(), inplace=True)
#train.fillna(train.median(), inplace=True)

# Normalizing Data
data=pd.concat([train, test])
skewed = ["mvar6", "mvar7","mvar8", "mvar9",
          "mvar10", "mvar11","mvar12", "mvar13",
          "mvar14", "mvar15","mvar21", "mvar22",
          "mvar23", "mvar24", "mvar25", "mvar26",
          "mvar27","mvar28", "mvar29", "mvar30", "mvar32"]

def normalizing(X):
    data[X][data[X] < 0] = 0
    data[X][data[X] > data[X].quantile(0.99)] = data[X].quantile(0.99)
    data[X] = data[X].apply(lambda x: x + np.random.rand())

    data[X] = data[X].apply(lambda x: x + 1)
    data[X], _ = boxcox(data[X])

for i in skewed:
    normalizing(i)

fe = False
if fe:
    # Feature engineering
    print("Feature engineering...")
    y = train.iloc[:,-1]
    Severity = ['mvar3','mvar4','mvar5']
    lda_Severity = LDA(n_components=5)
    lda_Severity = lda_Severity.fit(train[Severity], y)
    data['Severity'] = lda_Severity.transform(data[Severity])
    
    No_of_active = ['mvar16','mvar17','mvar19', 'mvar20','mvar18']
    lda_No_of_active = LDA(n_components=5)
    lda_No_of_active = lda_No_of_active.fit(train[No_of_active], y)
    data['No_of_active'] = lda_No_of_active.transform(data[No_of_active])
    
    Average_utilization = ['mvar21','mvar22','mvar23', 'mvar24']
    lda_Average_utilization = LDA(n_components=5)
    lda_Average_utilization = lda_Average_utilization.fit(train[Average_utilization], y)
    data['Average_utilization'] = lda_Average_utilization.transform(data[Average_utilization])
    
    No_of_active_line = ['mvar34','mvar35','mvar36']
    lda_No_of_active_line = LDA(n_components=5)
    lda_No_of_active_line = lda_No_of_active_line.fit(train[No_of_active_line], y)
    data['No_of_active_line'] = lda_No_of_active_line.transform(data[No_of_active_line])
    
    print("Shape of Data after Feature Engineering: ", data.shape)

#  Preparing Data
train_new = data.iloc[0:train.shape[0]]
test_new = data.iloc[train.shape[0]:data.shape[0]]

def prepare_data(df, is_train):
    if is_train:
        return df.drop(["application_key",'default_ind', 'mvar16','mvar18','mvar17', 'mvar39','mvar35','mvar46', 'mvar45'], axis=1), df.iloc[:,:2]
    return df.drop(["application_key",'default_ind', 'mvar16','mvar18','mvar17', 'mvar39','mvar35','mvar46', 'mvar45'], axis=1), df["application_key"]

train_features, train_target = prepare_data(train_new, 1)
test_features, test_key= prepare_data(test_new, 0)

import xgboost
from sklearn.cross_validation import train_test_split

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, precision_score, recall_score

Xtrain, Xtest, my_ytrain, my_ytest = train_test_split(train_features, train_target, test_size=.2, random_state=100)

ytrain = my_ytrain['default_ind']
ytest = my_ytest['default_ind']
yKey = my_ytest["application_key"]

params_fixed = {
    'objective': 'binary:logistic',
    'silent': 0,
    'n_estimators' :1000,
     'max_depth': 6,
     'reg_lambda': 0.9,
     'seed': 100,
     'learning_rate': 0.01,
     'reg_alpha': 0.01,
     'min_child_weight': 3
}

xgb = XGBClassifier(**params_fixed,)
xgb.fit(Xtrain, ytrain)

print("Training :" + str(accuracy_score(ytrain, xgb.predict(Xtrain))))
print("Test :" + str(accuracy_score(ytest, xgb.predict(Xtest))))
print("ROC_AUC :" + str(roc_auc_score(ytest, xgb.predict(Xtest))))
print("precision_score :" + str(precision_score(ytest, xgb.predict(Xtest))))
print("recall_score :" + str(recall_score(ytest, xgb.predict(Xtest))))
print("confusion_matrix :" + str(confusion_matrix(ytest, xgb.predict(Xtest))))


ax = xgboost.plot_importance(xgb)
fig = ax.figure
fig.set_size_inches(15, 15)


proba = xgb.predict_proba(test_features)[:, 1]
pred = []

U_ther, L_ther = 0.915, 0.11
test_proba = xgb.predict_proba(Xtest)[:, 1]
my_test, my_pre = [], []

with open('result/blablabla_IIT_Madras_XGB_test' + str(L_ther) + "_" + str(U_ther) + '.csv', "w") as file:
     for i in range(len(test_proba)):
         if test_proba[i] < L_ther:
             my_test.append(ytest.iloc[i])
             my_pre.append(0)
             file.write(str(yKey.iloc[i]) + "," + str(0) + "," +"\n")
         if test_proba[i] > U_ther:
             my_test.append(ytest.iloc[i])
             my_pre.append(1)
             file.write(str(yKey.iloc[i]) + "," + str(1) + "," +"\n")
             

print("My Test :" + str(accuracy_score(my_test, my_pre)))
print("My ROC_AUC :" + str(roc_auc_score(my_test, my_pre)))
print("My precision_score :" + str(precision_score(my_test, my_pre)))
print("My recall_score :" + str(recall_score(my_test, my_pre)))
print("My confusion_matrix :" + str(confusion_matrix(my_test, my_pre)))

U_ther, L_ther = 0.91, 0.11
'''
with open('result/blablabla_IIT_Madras_XGB_test_prob_ther_9_' + str(L_ther) + "_" + str(U_ther) + '.csv', "w") as f:
    for i in range(len(proba)):
     
         if proba[i] < L_ther:
             f.write(str(test_key.iloc[i]) + "," + str(0) + "," + str(proba[i]) + "\n")
         if proba[i] > U_ther:
             f.write(str(test_key.iloc[i]) + "," + str(1) + "," + str(proba[i]) + "\n")

'''
with open('result/blablabla_IIT_Madras_XGB_test_prob_ther_2000' + str(L_ther) + "_" + str(U_ther) + '.csv', "w") as f:
    for i in range(len(proba)):
        if proba[i] > 0.5:
            f.write(str(test_key.iloc[i]) + "," + str(1) + "," + str(proba[i]) + "\n")
        else:
            f.write(str(test_key.iloc[i]) + "," + str(0) + "," + str(proba[i]) + "\n")

