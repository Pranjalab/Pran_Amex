# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 03:43:02 2018

@author: Pranjal
"""
# Initial imports
import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import boxcox
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

random.seed(3)

# Imports for better visualization
from matplotlib import rcParams

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
from sklearn.grid_search import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, precision_score, recall_score

Xtrain, Xtest, my_ytrain, my_ytest = train_test_split(train_features, train_target, test_size=.2, random_state=100)

ytrain = my_ytrain['default_ind']
ytest = my_ytest['default_ind']
yKey = my_ytest["application_key"]

params_grid = {
    'max_depth': [6,9],
    'learning_rate': [0.05, 0.1],
    'gamma': [0.0,0.1],
    'reg_lambda' : [0.9, 1.1] 
}
params_fixed = {
    'objective': 'binary:logistic',
    'silent': 1,
    'n_estimators' :1000
}

seed = 100
bst_grid = GridSearchCV(
    estimator=XGBClassifier(**params_fixed, seed=seed),
    param_grid=params_grid,
    scoring='accuracy'
)


bst_grid.fit(Xtrain, ytrain)

scores = bst_grid.grid_scores_

print(scores)


print("Best accuracy obtained: {0}".format(bst_grid.best_score_))
print("Parameters:")
for key, value in bst_grid.best_params_.items():
    print("\t{}: {}".format(key, value))
    
    
    
    
    
    
    
    
    
    

print("Training :" + str(accuracy_score(ytrain, bst_grid.predict(Xtrain))))
print("Test :" + str(accuracy_score(ytest, bst_grid.predict(Xtest))))
print("ROC_AUC :" + str(roc_auc_score(ytest, bst_grid.predict(Xtest))))
print("precision_score :" + str(precision_score(ytest, bst_grid.predict(Xtest))))
print("recall_score :" + str(recall_score(ytest, bst_grid.predict(Xtest))))
print("confusion_matrix :" + str(confusion_matrix(ytest, bst_grid.predict(Xtest))))


ax = xgboost.plot_importance(xgb)
fig = ax.figure
fig.set_size_inches(15, 15)


proba = bst_grid.predict_proba(test_features)[:, 1]
pred = []

U_ther, L_ther = 0.88, 0.11
test_proba = bst_grid.predict_proba(Xtest)[:, 1]
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

L_ther_no = 0
U_ther_no = 0

with open('result/blablabla_IIT_Madras_XGB_test_prob_ther_5' + str(L_ther) + "_" + str(U_ther) + '.csv', "w") as f:
    for i in range(len(proba)):
        
         if proba[i] < L_ther:
             L_ther_no += 1
             f.write(str(test_key.iloc[i]) + "," + str(0) + "," + str(proba[i]) + "\n")
         if proba[i] > U_ther:
             U_ther_no += 1
             f.write(str(test_key.iloc[i]) + "," + str(1) + "," + str(proba[i]) + "\n")
'''
        if proba[i] > 0.5:
            f.write(str(test_key.iloc[i]) + "," + str(1) + "," + str(proba[i]) + "\n")
        else:
            f.write(str(test_key.iloc[i]) + "," + str(0) + "," + str(proba[i]) + "\n")
        '''


