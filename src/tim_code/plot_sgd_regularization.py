import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from format import get_data
from timeit import default_timer as timer
from sklearn import preprocessing
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

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

    max_allowed_iters = 1500
    print ("SGD fitting")
    #test sgd with l1, l2, or elasticnet
    
    alpha_values = [0.0001, 0.005, 0.02, 0.03,0.04, 0.05, 0.07,0.08,0.09,0.1] #0.07 is best
    f1_l1_scores = []
    f1_elasticnet_scores = []
    f1_l2_scores = []
    for alpha_val in alpha_values:
    	sgd_elasticnet = linear_model.SGDClassifier(max_iter=1000, tol=1e-3, 
    	                                penalty='elasticnet', alpha=alpha_val, shuffle=False)
    	sgd_elasticnet.fit(scaled_train_X, train_Y)
    	pred_elasticnet_Y = sgd_elasticnet.predict(scaled_test_X)

    	score_elasticnet_f1 = f1_score(test_Y, pred_elasticnet_Y)
    	print ("f1_score is: " + str(score_elasticnet_f1) + "for alpha: " + str(alpha_val))

    	f1_elasticnet_scores.append(score_elasticnet_f1)


    	sgd_l1 = linear_model.SGDClassifier(max_iter=1000, tol=1e-3, 
    	                                penalty='l1', alpha=alpha_val, shuffle=False)
    	sgd_l1.fit(scaled_train_X, train_Y)
    	pred_l1_Y = sgd_l1.predict(scaled_test_X)

    	score_l1_f1 = f1_score(test_Y, pred_l1_Y)
    	f1_l1_scores.append(score_l1_f1)

    	sgd_l2 = sgd_l1 = linear_model.SGDClassifier(max_iter=1000, tol=1e-3, 
    	                                penalty='l2', alpha=alpha_val, shuffle=False)
    	sgd_l2.fit(scaled_train_X, train_Y)
    	pred_l2_f1 = sgd_l2.predict(scaled_test_X)
    	score_l2_f1 = f1_score(test_Y, pred_l2_f1)
    	f1_l2_scores.append(score_l2_f1)

    plt.plot(alpha_values, f1_elasticnet_scores, 'ro-', alpha_values, f1_l1_scores, 'g^-',
    		alpha_values, f1_l2_scores, 'b2-')
    plt.ylabel('f1 score')
    plt.xlabel("regularization alpha values")
    plt.title("F1 scores for SGD Classifier with Regularization")
    plt.show()
    # savefig("sgd_f1_scores.png")

if __name__ == "__main__":
    main()