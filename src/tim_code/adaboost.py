import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from format import get_data
from timeit import default_timer as timer
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

def main():
    #getting and splitting data
    print ("parsing data")
    patients, egm_matrix, cancer_onehot = get_data()
    train_X, test_X, train_Y, test_Y = train_test_split(egm_matrix,cancer_onehot, test_size=0.20, random_state=0)

    #make y sets +/- 1
    # print ("train_Y is: " + str(train_Y))
    # for ind in range(train_Y):
    # 	if train_Y[ind] == 0:
    # 		train_Y[ind] = -1
    # for ind2 in range(test_Y):
    # 	if test_Y[ind] == 0:
    # 		test_Y[ind] = -1

    #form scaled data
    scaler = preprocessing.StandardScaler().fit(train_X)
    print (scaler)
    scaled_train_X = scaler.transform(train_X)
    scaled_test_X = scaler.transform(test_X)

    ada_boost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
                          n_estimators=100, learning_rate=1.0, random_state=0)
    ada_boost.fit(scaled_train_X, train_Y)

    pred_Y = ada_boost.predict(scaled_test_X)

    f1_score_adda = f1_score(test_Y, pred_Y, average=None)
    print ("f1 score is: " + str(f1_score_adda)) #f1 score is: [0.63703704 0.36363636] for learning rate 1.0

    mean_train_accuracy = ada_boost.score(scaled_train_X, train_Y)
    print ("mean_train_accuracy is: " + str(mean_train_accuracy))
    mean_test_accuracy = ada_boost.score(scaled_test_X, test_Y)
    print ("mean_test_accuracy is: " + str(mean_test_accuracy))
    print (ada_boost)



if __name__ == "__main__":
    main()