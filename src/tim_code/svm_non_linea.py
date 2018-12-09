import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn import linear_model
from sklearn import svm
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

    max_allowed_iters = 1500
    print ("SVM fitting")
    start = timer()
    #test sgd with l1, l2, or elasticnet
    # sgd = linear_model.SGDClassifier(loss='squared_hinge', max_iter=1000, tol=1e-3, 
    #                                 penalty='l1', shuffle=True)
    # clf_svm = svm.SVC(C=0.5, gamma=0.2, kernel='rbf', shrinking=False)
    clf_svm = svm.SVC(kernel='linear', shrinking=False)
    clf_svm.fit(scaled_train_X, train_Y)
    pred_Y = clf_svm.predict(scaled_test_X)
    end = timer()
    print ("sgd took: " + str(end - start) + " seconds")

    # num_iters = clf_svm.n_iter_
    # print ("took " + str(num_iters) + " iterations")

    print ("test accuracy is: " + str(clf_svm.score(scaled_test_X, test_Y)))
    print ("train accuracy is: " + str(clf_svm.score(scaled_train_X, train_Y)))
    print ("pred_Y is: " + str(pred_Y))
    score_f1 = f1_score(test_Y, pred_Y, average=None)
    print ("f1_score is: " + str(score_f1))
    print (clf_svm)

if __name__ == "__main__":
    main()