import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from format import get_data
from timeit import default_timer as timer
from sklearn.metrics import f1_score

def main():
    #getting and splitting data
    print ("parsing data")
    patients, egm_matrix, cancer_onehot = get_data()
    train_X, test_X, train_Y, test_Y = train_test_split(egm_matrix,cancer_onehot, test_size=0.20, random_state=0)

    # total_train_ex = 0
    # total_train_pos = 0
    # for lbl in train_Y:
    #     total_train_ex += 1
    #     if lbl == 1:
    #         total_train_pos += 1
    # print ("total_train_ex: " + str(total_train_ex) + " total_train_pos: " + str(total_train_pos))

    # total_test_ex = 0
    # total_test_pos = 0
    # for lbl in test_Y:
    #     total_test_ex += 1
    #     if lbl == 1:
    #         total_test_pos += 1
    # print ("total_test_ex: " + str(total_test_ex) + " total_test_pos: " + str(total_test_pos))
    best_f1_score = f1_score(test_Y, test_Y, average=None)
    print ("best_f1_score: " + str(best_f1_score))

    print ("LR fitting")
    start = timer()
    max_allowed_iters = 1500
    # lr = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=max_allowed_iters)
    # lr = LogisticRegression(solver='lbfgs', max_iter=400)
    lr = LogisticRegression(solver='lbfgs', max_iter=max_allowed_iters)
    lr.fit(train_X, train_Y)
    # print ("LR predicting")
    pred_Y = lr.predict(test_X)
    end = timer()
    print ("logistic regression took: " + str(end - start) + " seconds")

    num_iters = lr.n_iter_
    print ("took " + str(num_iters[0]) + " iterations")

    print ("test accuracy is: " + str(lr.score(test_X, test_Y)))
    print ("train accuracy is: " + str(lr.score(train_X, train_Y)))

    score_f1 = f1_score(test_Y, pred_Y, average=None)
    print ("f1_score is: " + str(score_f1))

if __name__ == "__main__":
    main()