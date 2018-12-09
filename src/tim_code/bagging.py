import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from format import get_data
from timeit import default_timer as timer
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

def main():
    #getting and splitting data
    print ("parsing data")
    patients, egm_matrix, cancer_onehot = get_data()
    train_X, test_X, train_Y, test_Y = train_test_split(egm_matrix,cancer_onehot, test_size=0.20, random_state=0)

    #form scaled data
    scaler = preprocessing.StandardScaler().fit(train_X)
    print (scaler)
    scaled_train_X = scaler.transform(train_X)
    scaled_test_X = scaler.transform(test_X)

    bagging = BaggingClassifier(DecisionTreeClassifier(max_depth=3), max_features=20000,
    							max_samples=1.0, n_estimators=11, random_state=2)

    bagging.fit(scaled_train_X, train_Y)

    pred_Y = bagging.predict(scaled_test_X)

    f1_score_bagging = f1_score(test_Y, pred_Y, average=None)
    print ("f1 score is: " + str(f1_score_bagging)) #max_features=20000 -> f1 score is: [0.77852349 0.47619048]

    mean_train_accuracy = bagging.score(scaled_train_X, train_Y)
    print ("mean_train_accuracy is: " + str(mean_train_accuracy))
    mean_test_accuracy = bagging.score(scaled_test_X, test_Y)
    print ("mean_test_accuracy is: " + str(mean_test_accuracy))
    print (bagging)
if __name__ == "__main__":
    main()