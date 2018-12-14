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
#custom written utilities
from format import load_BioAge1HO
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

def tree_classifier():
    return DecisionTreeClassifier()

def adaBoost_classifier():
    steps = [
        ('scl', StandardScaler()),
        ('ada', AdaboostClassifier(DecisionTreeClassifier(max_depth=2),
            n_estimators=100, learning_rate=1.0, random_state=0))
    ]
    return Pipeline(steps)

def random_forest_classifier():
    steps = [
        ('scl', StandardScaler()),
        ('rfc', RandomForestClassifier(bootstrap=True, criterion='gini',
            n_estimators=80, max_depth=3, max_features = 'aut0',
            oob_score=True, random_state=0))
    ]
    return Pipeline(steps)

def bagging_classifier():
    steps = [
        ('scl', StandardScaler()),
        ('bag', BaggingClassifier(DecisionTreeClassifier(max_depth=3), 
            max_features=20000, max_samples=1.0, n_estimators=11, random_state=2))
    ]
    return Pipeline(steps)

##########################################################

def test_model(model, X, y):
    #initializing evaluation scores
    avg_acc = 0.0
    avg_prec = 0.0
    avg_rec = 0.0
    avg_roc_auc = 0.0
    avg_f1 = 0.0
    #cross validation
    cv = KFold(n_splits=K_CV_SPLITS)
    for train, test in cv.split(X, y):
        #learn and make predictions
        model.fit(X[train], y[train])
        y_pred = model.predict(X[test])
        #y_pos_probs = model.predict_proba(X[test])[:,1]
        #evaluate performance
        avg_acc += accuracy_score(y[test], y_pred)
        avg_prec += precision_score(y[test], y_pred)
        avg_rec += recall_score(y[test], y_pred)
        #avg_roc_auc += roc_auc_score(y[test], y_pos_probs)
        avg_f1 += f1_score(y[test], y_pred)
    #average performance scores
    avg_acc /= K_CV_SPLITS
    avg_prec /= K_CV_SPLITS
    avg_rec /= K_CV_SPLITS
    avg_roc_auc /= K_CV_SPLITS
    avg_f1 /= K_CV_SPLITS

    #return [avg_acc, avg_prec, avg_rec, avg_f1, avg_roc_auc]
    return [avg_acc, avg_prec, avg_rec, avg_f1]

def main():
    #load_BioAge1HO()
    
    #getting and splitting data
    print("parsing data...")
    patients, X, y = get_data()
    print("done.")
    #training models with different parameters
    #SGD
    eval_scores = []
    alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    for alpha in alphas:
        start_time = timer()
        model = sgd_classifier(alpha, 'l1')
        eval_scores.append(test_model(model, X, y))
        end_time = timer()
        print(end_time - start_time)
    results = np.array(eval_scores)
    accs = results[:,0]
    precs = results[:,1]
    recs = results[:,2]
    f1s = results[:,3]
    #rocs = results[:,4]
    plt.plot(alphas, accs)
    plt.show()

   
if __name__ == "__main__":
    main()
