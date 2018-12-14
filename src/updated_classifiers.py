#standard packages
import numpy as np
from timeit import default_timer as timer
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
#sklearn utilities
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
#sklearn classifiers
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
#sklearn evaluation metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

#custom written utilities
# from format import load_BioAge1HO
from format import get_data

'''constants'''
K_CV_SPLITS = 5
MAX_ITERS = 1500

######################### MODELS #########################
#based on example from Towards Data Science (https://towardsdatascience.com)
#article "PCA using Python (scikit-learn)"
#parameters: PCA_param, which can be a float for variance retention
#or a number of components to retain
def PCA_classifier(PCA_param):
    steps = [
        ('scl', StandardScaler()),
        ('pca', PCA(n_components=PCA_param)),
        ('clf', LogisticRegression(solver='lbfgs', max_iter=MAX_ITERS))
    ]
    return Pipeline(steps)

def LR_classifier():
    steps = [
        ('scl', StandardScaler()),
        ('clf', LogisticRegression(solver='lbfgs', max_iter=MAX_ITERS))
    ]
    return Pipeline(steps)

#parameters: alpha (0, 0.0001, 0.1), penalty ('elasticnet','l1','l2')
#tol? used 1e-13 w/ default alpha = 0.0001 and 1e-3 here
def sgd_classifier(alpha_val, penalty_type):
    steps = [
        ('scl', StandardScaler()),
        ('sgd', SGDClassifier(max_iter=MAX_ITERS, tol=1e-3, 
            penalty=penalty_type, alpha = alpha_val, shuffle=False))
    ]
    return Pipeline(steps)

def svm_classifier(gamma_value):
    steps = [
        ('scl', StandardScaler()),
        ('svc', SVC(C=0.2, kernel='rbf', gamma=gamma_value, shrinking=False))
    ]
    return Pipeline(steps)

def tree_classifier(depth):
    steps = [
        ('scl', StandardScaler()),
        ('dt', DecisionTreeClassifier(max_depth=depth, max_features=None))
    ]
    return Pipeline(steps)

def adaBoost_classifier(learn_rate):
    steps = [
        ('scl', StandardScaler()),
        ('ada', AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),
            n_estimators=100, learning_rate=learn_rate, random_state=0))
    ]
    return Pipeline(steps)

def random_forest_classifier(num_estimators):
    steps = [
        ('scl', StandardScaler()),
        ('rfc', RandomForestClassifier(bootstrap=True, criterion='gini',
            n_estimators=num_estimators, max_depth=3, max_features = 'auto',
            oob_score=True, random_state=0))
    ]
    return Pipeline(steps)

def bagging_classifier(num_max_features):
    steps = [
        ('scl', StandardScaler()),
        ('bag', BaggingClassifier(DecisionTreeClassifier(max_depth=3), 
            max_features=num_max_features, max_samples=1.0, n_estimators=11, random_state=2))
    ]
    return Pipeline(steps)

##########################################################

def test_model(model, X, y): #modify for rf
    #initializing evaluation scores
    avg_train_acc = 0.0
    avg_test_acc = 0.0
    avg_prec = 0.0
    avg_rec = 0.0
    avg_roc_auc = 0.0
    avg_f1 = 0.0
    #cross validation
    cv = KFold(n_splits=K_CV_SPLITS)
    for train, test in cv.split(X, y):
        #learn and make predictions
        model.fit(X[train], y[train])
        y_train_pred = model.predict(X[train])
        y_pred = model.predict(X[test])
        #evaluate performance
        avg_train_acc += accuracy_score(y[train], y_train_pred)
        avg_test_acc += accuracy_score(y[test], y_pred)
        avg_prec += precision_score(y[test], y_pred)
        avg_rec += recall_score(y[test], y_pred)
        avg_f1 += f1_score(y[test], y_pred)

    #average performance scores
    avg_train_acc /= K_CV_SPLITS
    avg_test_acc /= K_CV_SPLITS
    avg_prec /= K_CV_SPLITS
    avg_rec /= K_CV_SPLITS
    avg_roc_auc /= K_CV_SPLITS
    avg_f1 /= K_CV_SPLITS

    #return [avg_acc, avg_prec, avg_rec, avg_f1, avg_roc_auc]
    return [avg_train_acc, avg_test_acc, avg_prec, avg_rec, avg_f1]

def evaluate_hyperparams(hyperparams_arr, classifier, X, y, 
                        clf_name, param_name):
    scores = []
    for param in hyperparams_arr:
        print (param)
        curr_model = classifier(param)
        # curr_model = classifier(param, 'l2') #only for sgd
        scores.append(test_model(curr_model, X, y))

    results = np.array(scores)
    train_accs = results[:,0]
    test_accs = results[:,1]
    precs = results[:,2]
    recs = results[:,3]
    f1s = results[:,4]
    print ("printing results for: " + clf_name)
    print ("train_accs: " + str(train_accs))
    print ("test_accs: " + str(test_accs))
    print ("precs: " + str(precs))
    print ("recs: " + str(recs))
    print ("f1s: " + str(f1s))

    plt.plot(hyperparams_arr, recs, 'b2-')
    plt.ylabel('Recall score')
    plt.xlabel(param_name)
    plt.title("Recall scores for " + clf_name + " Classifier")
    plt.show()

def run_on_test_data(test_model, train_X, train_Y, test_X, test_Y, test_name):
    print (test_name)
    test_model.fit(train_X, train_Y)
    test_pred = test_model.predict(test_X)
    train_pred = test_model.predict(train_X)

    train_acc = accuracy_score(train_Y, train_pred)
    test_acc = accuracy_score(test_Y, test_pred)
    prec_score = precision_score(test_Y, test_pred)
    rec_score = recall_score(test_Y, test_pred)
    test_f1_score = f1_score(test_Y, test_pred)
    print ("1's in train_pred: " + str(train_pred.tolist().count(1)))
    print ("1's in test_pred: " + str(test_pred.tolist().count(1)))
    print ("test_acc: " + str(test_acc))
    print ("train_acc: " + str(train_acc))
    print ("prec_score: " + str(prec_score))
    print ("rec_score: " + str(rec_score))
    print ("test_f1_score: " + str(test_f1_score))
    conf_matrix = confusion_matrix(test_Y, test_pred)
    print ("confustion matrix is: " + str(conf_matrix))


def main():
    #load_BioAge1HO()
    
    #getting and splitting data
    print("parsing data...")
    patients, X, y = get_data()
    print("done.")

    #form test and train set
    num_ex, num_features = X.shape
    train_X_arr = []
    train_Y_arr = []
    test_X_arr = []
    test_Y_arr = []
    num_neg_test_ex = 64
    num_pos_test_ex = 42
    for ind in range(num_ex):
        lbl = y[ind]
        if num_pos_test_ex > 0 and lbl == 1:
            test_Y_arr.append(1)
            test_X_arr.append(X[ind])
            num_pos_test_ex -= 1
        elif num_neg_test_ex > 0 and lbl == 0:
            test_Y_arr.append(0)
            test_X_arr.append(X[ind])
            num_neg_test_ex -= 1
        else:
            train_Y_arr.append(lbl)
            train_X_arr.append(X[ind])

    train_X = np.array(train_X_arr)
    train_Y = np.array(train_Y_arr)
    test_X = np.array(test_X_arr)
    test_Y = np.array(test_Y_arr)


    #random forests
    # n_estimators_arr = [50, 75, 90, 95, 100, 105, 110,115]
    # evaluate_hyperparams(n_estimators_arr, random_forest_classifier,
                            # train_X, train_Y, "Random Forests", "Number of trees per forest")
    # rf_clf = random_forest_classifier(100)
    # run_on_test_data(rf_clf, train_X, train_Y, test_X, test_Y, 'random forest test')
    #sgd w/ l2 regularization
    alphas = [0.0001, 0.001, 0.01, 0.1, 0.7, 1, 5]
    # evaluate_hyperparams(alphas, sgd_classifier, train_X, train_Y 
                        # "SGD w/ L2 regularization", "alpha values")



    #bagging
    max_features_arr = [17000, 18500, 20000, 20500, 21000, 21100, 21200, 22000]
    # evaluate_hyperparams(max_features_arr, bagging_classifier, train_X, train_Y,
                        # "Bagging", "Maximum Features for Estimator")
    # bagging_clf = bagging_classifier(21000)
    # run_on_test_data(bagging_clf, train_X, train_Y, test_X, test_Y, 'bagging test')

    #adaboost
    learn_rates = [0.2, 0.4, 0.5, 0.7, 0.9, 1.0]
    # evaluate_hyperparams(learn_rates, adaBoost_classifier, train_X, train_Y, 
    					# "AdaBoost", "Learning Rate")
    
    #sigmoid
    # sigmoid_penalty_terms = [0.2, 0.4, 0.5, 0.7, 0.75, 0.85, 1.0]
    # evaluate_hyperparams(sigmoid_penalty_terms, svm_classifier, train_X, train_Y,
    					# "Sigmoid Kernel", "Penalty term")
    sigmoid_coef0_arr = [0.0, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7]
    # evaluate_hyperparams(sigmoid_coef0_arr, svm_classifier, train_X, train_Y,
    					# "Sigmoid Kernel", "Constant value")
    # sigmoid_clf = svm_classifier(0.0)
    # run_on_test_data(sigmoid_clf, train_X, train_Y, test_X, test_Y, 'sigmoid kernel test')

    #polynomial
    degrees = [1,2,3,4]
    # evaluate_hyperparams(degrees, svm_classifier, train_X, train_Y,
                        # "Polynomial Kernel", "Degree of Polynomial")
    # poly_svm = svm_classifier(1)
    # run_on_test_data(poly_svm, train_X, train_Y, test_X, test_Y, 'poly test')

    #rbf svm
    rbf_gamma_arr = [0.1, 0.2, 0.4, 0.55, 0.6, 0.75, 0.8, 0.85, 0.9]
    # rbf_penalty_terms = [0.2, 0.4, 0.5, 0.7, 0.75, 0.85]
    # evaluate_hyperparams(rbf_gamma_arr, svm_classifier, train_X, train_Y,
    					# "RBF kernel", 'Gamma values')
    # rbf_clf = svm_classifier()
    # run_on_test_data(rbf_clf, train_X, )

    #PCA
    # retained_variance = [0.5, 0.6, 0.75, 0.85, 0.95]
    num_components = [75, 80, 85, 90, 95, 100, 110, 115, 120, 125, 130, 135, 140]
    # evaluate_hyperparams(num_components, PCA_classifier, train_X, train_Y,
                        # "PCA", "Number of Components Retained")
    # pca_clf = PCA_classifier(110)
    # run_on_test_data(pca_clf, train_X, train_Y, test_X, test_Y, 'pca test')

    #decision trees
    max_depths = [1,2,3,4,5,6]
    # max_features = ['auto', 'sqrt', None]
    # evaluate_hyperparams(max_depths, tree_classifier, train_X, train_Y,
                        # "Decision Tree", "Maximum Depth")
    # dt_clf = tree_classifier(4)
    # run_on_test_data(dt_clf, train_X, train_Y, test_X, test_Y, "decision tree test")


    # lr_model = LR_classifier()
    # lr_results = test_model(lr_model, train_X, train_Y)
    # print ("lr results:")
    # print ("avg_train_acc: " + str(lr_results[0]))
    # print ("avg_test_acc: " + str(lr_results[1]))
    # print ("avg_prec: " + str(lr_results[2]))
    # print ("avg_recall: " + str(lr_results[3]))
    # print ("f1_score: " + str(lr_results[4]))

    #lr train/test
    # test_model = LogisticRegression(solver='lbfgs', max_iter=MAX_ITERS)
    test_model = LR_classifier()
    test_model.fit(train_X, train_Y)
    test_pred = test_model.predict(test_X)
    train_pred = test_model.predict(train_X)
    train_acc = accuracy_score(train_Y, train_pred)
    test_acc = accuracy_score(test_Y, test_pred)
    prec_score = precision_score(test_Y, test_pred)
    rec_score = recall_score(test_Y, test_pred)
    lr_f1_score = f1_score(test_Y, test_pred)
    print ("1's in train_pred: " + str(train_pred.tolist().count(1)))
    print ("1's in test_pred: " + str(test_pred.tolist().count(1)))
    print ("test_acc: " + str(test_acc))
    print ("train_acc: " + str(train_acc))
    print ("prec_score: " + str(prec_score))
    print ("rec_score: " + str(rec_score))
    print ("f1_score: " + str(lr_f1_score))
    lr_conf_matrix = confusion_matrix(test_Y, test_pred)
    print ("lr confusion matrix is: " + str(lr_conf_matrix))

    #SGD
    # l1_eval_scores = []
    # l2_eval_scores = []
    # elasticnet_eval_scores = []
    # alphas = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.13, 0.14]
    # for alpha in alphas:
    #     l1_model = sgd_classifier(alpha, 'l1')
    #     l1_eval_scores.append(test_model(l1_model, train_X, train_Y))

    #     l2_model = sgd_classifier(alpha, 'l2')
    #     l2_eval_scores.append(test_model(l2_model, train_X, train_Y))

    #     elasticnet_model = sgd_classifier(alpha, 'elasticnet')
    #     elasticnet_eval_scores.append(test_model(elasticnet_model, train_X, train_Y))
    # l1_results = np.array(l1_eval_scores)
    # l1_train_accs = l1_results[:,0]
    # l1_test_accs = l1_results[:,1]
    # l1_precs = l1_results[:,2]
    # l1_recs = l1_results[:,3]
    # l1_f1s = l1_results[:,4]
    # print ("l1_train_accs: " + str(l1_train_accs))
    # print ("l1_test_accs: " + str(l1_test_accs))
    # print ("l1_precs: " + str(l1_precs))
    # print ("l1_recs: " + str(l1_recs))
    # print ("l1_f1s: " + str(l1_f1s))

    # l2_results = np.array(l2_eval_scores)
    # l2_train_accs = l2_results[:,0]
    # l2_test_accs = l2_results[:,1]
    # l2_precs = l2_results[:,2]
    # l2_recs = l2_results[:,3]
    # l2_f1s = l2_results[:,4]

    # print ("l2_train_accs: " + str(l2_train_accs))
    # print ("l2_test_accs: " + str(l2_test_accs))
    # print ("l2_precs: " + str(l2_precs))
    # print ("l2_recs: " + str(l2_recs))
    # print ("l2_f1s: " + str(l2_f1s))

    # elasticnet_results = np.array(elasticnet_eval_scores)
    # elasticnet_train_accs = elasticnet_results[:,0]
    # elasticnet_test_accs = elasticnet_results[:,1]
    # elasticnet_precs = elasticnet_results[:,2]
    # elasticnet_recs = elasticnet_results[:,3]
    # elasticnet_f1s = elasticnet_results[:,4]

    # print ("elasticnet_train_accs: " + str(elasticnet_train_accs))
    # print ("elasticnet_test_accs: " + str(elasticnet_test_accs))
    # print ("elasticnet_precs: " + str(elasticnet_precs))
    # print ("elasticnet_recs: " + str(elasticnet_recs))
    # print ("elasticnet_f1s: " + str(elasticnet_f1s))

    # plt.plot(alphas, l1_recs, 'g^-', alphas, l2_recs, 'ro-', alphas, elasticnet_recs, 'b2-')
    # plt.ylabel('recall score')
    # plt.xlabel("regularization alpha values")
    # plt.title("Recall scores for SGD Classifier with Regularization")
    # plt.show()

    #l2 scores
    # l2_sgd = sgd_classifier(0.06, 'l2')
    # run_on_test_data(l2_sgd, train_X, train_Y, test_X, test_Y, 'l2 sgd')

    # l1_sgd = sgd_classifier(0.05, 'l1')
    # run_on_test_data(l1_sgd, train_X, train_Y, test_X, test_Y, 'l1 sgd')

    # elasticnet_sgd = sgd_classifier(0.07, 'elasticnet')
    # run_on_test_data(elasticnet_sgd, train_X, train_Y, test_X, test_Y, 'elasticnet')

   
if __name__ == "__main__":
    main()