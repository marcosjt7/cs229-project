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
    return  LogisticRegression(solver='lbfgs', max_iter=MAX_ITERS)

#parameters: alpha (0, 0.0001, 0.1), penalty ('elasticnet','l1','l2')
#tol? used 1e-13 w/ default alpha = 0.0001 and 1e-3 here
def sgd_classifier(alpha_val, penalty_type):
    steps = [
        ('scl', StandardScaler()),
        ('sgd', SGDClassifier(max_iter=MAX_ITERS, tol=1e-3, 
            penalty=penalty_type, alpha = alpha_val, shuffle=False))
    ]
    return Pipeline(steps)

def svm_classifier():
    steps = [
        ('scl', StandardScaler()),
        ('svc', SVC(kernel='linear', shrinking=False))
    ]
    return Pipeline(steps)

def tree_classifier(depth):
    steps = [
        ('scl', StandardScaler()),
        ('dt', DecisionTreeClassifier(max_depth=depth, max_features=None))
    ]
    return Pipeline(steps)

def adaBoost_classifier():
    steps = [
        ('scl', StandardScaler()),
        ('ada', AdaboostClassifier(DecisionTreeClassifier(max_depth=2),
            n_estimators=100, learning_rate=1.0, random_state=0))
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
    avg_num_features = 0
    #cross validation
    cv = KFold(n_splits=K_CV_SPLITS)
    for train, test in cv.split(X, y):
        #learn and make predictions
        model.fit(X[train], y[train])
        y_train_pred = model.predict(X[train])
        y_pred = model.predict(X[test])
        #y_pos_probs = model.predict_proba(X[test])[:,1]
        #evaluate performance
        avg_train_acc += accuracy_score(y[train], y_train_pred)
        avg_test_acc += accuracy_score(y[test], y_pred)
        avg_prec += precision_score(y[test], y_pred)
        avg_rec += recall_score(y[test], y_pred)
        #avg_roc_auc += roc_auc_score(y[test], y_pos_probs)
        avg_f1 += f1_score(y[test], y_pred)

        # rf_feature_weights = model.feature_importances_
        # curr_num_features_used = 0
        # for weight in  rf_feature_weights:
        #     if weight >= 0:
        #         curr_num_features_used += 1
        # avg_num_features += curr_num_features_used
    #average performance scores
    avg_train_acc /= K_CV_SPLITS
    avg_test_acc /= K_CV_SPLITS
    avg_prec /= K_CV_SPLITS
    avg_rec /= K_CV_SPLITS
    avg_roc_auc /= K_CV_SPLITS
    avg_f1 /= K_CV_SPLITS
    avg_num_features /= K_CV_SPLITS

    #return [avg_acc, avg_prec, avg_rec, avg_f1, avg_roc_auc]
    return [avg_train_acc, avg_test_acc, avg_prec, avg_rec, avg_f1, avg_num_features]

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

    plt.plot(hyperparams_arr, f1s, 'b2-')
    plt.ylabel('F1 score')
    plt.xlabel(param_name)
    plt.title("F1 scores for " + clf_name + " Classifier")
    plt.show()

def main():
    #load_BioAge1HO()
    
    #getting and splitting data
    print("parsing data...")
    patients, X, y = get_data()
    print("done.")
    #training models with different parameters

    #random forests
    n_estimators_arr = [35, 40, 50, 75, 90, 100, 115]
    evaluate_hyperparams(n_estimators_arr, random_forest_classifier,
                            X, y, "Random Forests", "Number of trees per forest")

    #sgd w/ l2 regularization
    alphas = alphas = [0.0001, 0.001, 0.01, 0.1, 0.7, 1, 5]
    # evaluate_hyperparams(alphas, sgd_classifier, X, y, 
                        # "SGD w/ L2 regularization", "alpha values")

    #bagging
    max_features_arr = [17000, 18500, 20000, 20500, 21000, 21100, 21200, 22000]
    # evaluate_hyperparams(max_features_arr, bagging_classifier, X, y,
                        # "Bagging", "Maximum Features for Estimator")
    

    #PCA
    retained_variance = [0.5, 0.6, 0.75, 0.85, 0.95]
    # evaluate_hyperparams(retained_variance, PCA_classifier, X, y,
                        # "PCA", "Variance Retained")

    #decision trees
    max_depths = [1,2,3,4,5,6]
    # max_features = ['auto', 'sqrt', None]
    # evaluate_hyperparams(max_depths, tree_classifier, X, y,
                        # "Decision Tree", "Maximum Depth")

    # lr_model = LR_classifier()
    # lr_results = test_model(lr_model, X, y)
    # print ("lr results:")
    # print ("avg_train_acc: " + str(lr_results[0]))
    # print ("avg_test_acc: " + str(lr_results[1]))
    # print ("f1_score: " + str(lr_results[4]))

    #SGD
    # l1_eval_scores = []
    # l2_eval_scores = []
    # elasticnet_eval_scores = []
    # alphas = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.13, 0.14]
    # for alpha in alphas:
    #     l1_model = sgd_classifier(alpha, 'l1')
    #     l1_eval_scores.append(test_model(l1_model, X, y))

    #     l2_model = sgd_classifier(alpha, 'l2')
    #     l2_eval_scores.append(test_model(l2_model, X, y))

    #     elasticnet_model = sgd_classifier(alpha, 'elasticnet')
    #     elasticnet_eval_scores.append(test_model(elasticnet_model, X, y))
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

    # plt.plot(alphas, l1_f1s, 'g^-', alphas, l2_f1s, 'ro-', alphas, elasticnet_f1s, 'b2-')
    # plt.ylabel('f1 score')
    # plt.xlabel("regularization alpha values")
    # plt.title("F1 scores for SGD Classifier with Regularization")
    # plt.show()

   
if __name__ == "__main__":
    main()