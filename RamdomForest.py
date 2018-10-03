print("Running Ramdom Forest")
from sklearn.ensemble import RandomForestClassifier
from Data_Cleaning import Data

flag = [[True, False]]
imputer = 'median'
avg = 55
for f in flag:
    X_train, y_train, X_test, person_add = Data(remove_null=f[0], fetu_sel=f[1])
    print("Starting: ", imputer, avg)

    classifier = RandomForestClassifier(verbose=1)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)

    # Saving the result
    with open('result/blablabla_IIT_Madras_RFC' + imputer + str(avg) + str(f) +'.csv', "w") as f:
        for i in range(len(person_add)):
            if y_pred[i]:
                val = 1
            else:
                val = 0
            f.write(str(person_add[i]) + "," + str(val) + "\n")


