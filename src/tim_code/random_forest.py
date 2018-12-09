from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from format import get_data
from timeit import default_timer as timer
from sklearn import preprocessing
from sklearn.metrics import f1_score

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

    rand_forest = RandomForestClassifier(bootstrap=True, criterion='gini', n_estimators=80, max_depth=3, 
    									max_features='auto', oob_score=True, random_state=0)
    rand_forest.fit(scaled_train_X, train_Y)

    pred_Y = rand_forest.predict(scaled_test_X)
    rf_f1_score = f1_score(test_Y, pred_Y, average=None)
    print ("rf_f1_score is: " + str(rf_f1_score))

    print ("oob score is: " + str(rand_forest.oob_score_))

    rf_mean_train_accuracy = rand_forest.score(scaled_train_X, train_Y)
    print ("mean rf train accuracy is: " + str(rf_mean_train_accuracy))

    rf_mean_test_accuracy = rand_forest.score(scaled_test_X, test_Y)
    print ("mean rf test accuracy is: " + str(rf_mean_test_accuracy))

    print (rand_forest)
    print (len(rand_forest.feature_importances_))
    feat_importance = rand_forest.feature_importances_
    considered_feats = 0
    total_feats = 0
    total_weight = 0
    max_weight = 0
    for feat in feat_importance:
    	if feat > 0:
    		considered_feats += 1
    		total_weight += feat
    		max_weight = max(max_weight, feat)
    	total_feats += 1
    print ("considered_feats is: " + str(considered_feats))
    print ("total_weight is: " + str(total_weight))
    print ("max weight is: " + str(max_weight))
if __name__ == "__main__":
    main()