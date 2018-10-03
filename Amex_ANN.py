# Initial imports
import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import boxcox
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

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
Remove_NaN = True
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
test.fillna(test.median(), inplace=True)
train.fillna(train.median(), inplace=True)

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

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_features = sc.fit_transform(train_features)
test_features = sc.transform(test_features)

# Model
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras.activations import relu, sigmoid
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, precision_score, recall_score

Xtrain, Xtest, my_ytrain, my_ytest = train_test_split(train_features, train_target, test_size=.2, random_state=100)

ytrain = my_ytrain['default_ind']
ytest = my_ytest['default_ind']
yKey = my_ytest["application_key"]

# import keras.backend as K
def create_model(layers, activation_fun, opt, los):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i == 0:
            model.add(Dense(nodes, input_dim= Xtrain.shape[1]))
            model.add(Activation(activation_fun))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation_fun))
    model.add(Dense(1, activation = 'sigmoid'))  # Note: no activation beyond this point

    model.compile(optimizer=opt, loss=los, metrics = ['accuracy'])
    return model


model = KerasRegressor(build_fn=create_model, verbose=1)
#TODO : try dropouts
layers = [[40,40,20,10]]
#activations = [sigmoid, relu]
activations = [relu]
opt = ['adam']
los = ['binary_crossentropy']
param_grid = dict(layers=layers, activation_fun=activations,opt=opt, los=los, batch_size = [100], epochs=[500])
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error')

grid_result = grid.fit(Xtrain, ytrain)
i = 0
print([grid_result.best_score_, grid_result.best_params_])

print("Training :" + str(accuracy_score(ytrain, (grid.predict(Xtrain)>0.4).astype(int))))
print("Test :" + str(accuracy_score(ytest, (grid.predict(Xtest)>0.4).astype(int))))
print("ROC_AUC :" + str(roc_auc_score(ytest, (grid.predict(Xtest)>0.4).astype(int))))
print("precision_score :" + str(precision_score(ytest, (grid.predict(Xtest)>0.4).astype(int))))
print("recall_score :" + str(recall_score(ytest, (grid.predict(Xtest)>0.4).astype(int))))
print("confusion_matrix :" + str(confusion_matrix(ytest, (grid.predict(Xtest)>0.4).astype(int))))

proba = grid.predict(test_features)
pred = []

U_ther, L_ther = 0.75, 0.09
test_proba = grid.predict(Xtest)
my_test, my_pre = [], []



with open('result/blablabla_IIT_Madras_ANN_' + str(L_ther) + "_" + str(U_ther) + '.csv', "w") as file:
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


             
with open('result/blablabla_IIT_Madras_ANN_.csv', "w") as f:
    for i in range(len(proba)):
         if proba[i] < 0.5:
             f.write(str(test_key.iloc[i]) + "," + str(0) + "," + str(proba[i]) + "\n")
         else:
             f.write(str(test_key.iloc[i]) + "," + str(1) + "," + str(proba[i]) + "\n")































