#standard packages
import numpy as np
from timeit import default_timer as timer
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
def PCA_classifier(PCA_param):
    #PCR pipeline
    print("PCR:")
    steps = [
        ('scl', StandardScaler()),
        ('pca', PCA(n_components=PCA_param)),
        ('clf', LogisticRegression(solver='lbfgs', max_iter=MAX_ITERS))
    ]
    pcr_pipe = Pipeline(steps)
    return pcr_pipe

def LR_classifier():
    #simple logistic regression pipeline
    print("LR:")
    lr = LogisticRegression(solver='lbfgs', max_iter=MAX_ITERS)
    return lr

def sgd_classifier():
    steps = [
        ('scl', StandardScaler()),
        ('clf', SGDClassifier(max_iter=MAX_ITERS, tol=1e-13, penalty='l1',shuffle=False)
    ]
    sgd_pipe = Pipeline(steps)
    return sgd_pipe

def svm_classifier():
    steps = [
        ('scl', StandardScaler()),
        ('clf', SVC(kernel='linear', shrinking=False)
    ]
    svm_pipe = Pipeline(steps)
    return svm_pipe

def tree_classifier():
    print("Tree:")
    steps = [('clf', tree.DecisionTreeClassifier())]
    tree_pipe = Pipeline(steps)
    return tree_pipe
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
        y_pos_probs = model.predict_proba(X[test])[:,1]
        #evaluate performance
        avg_acc += accuracy_score(y[test], y_pred)
        avg_prec += precision_score(y[test], y_pred)
        avg_rec += recall_score(y[test], y_pred)
        avg_roc_auc += roc_auc_score(y[test], y_pos_probs)
        avg_f1 += f1_score(y[test], y_pred)
    #average performance scores
    avg_acc /= K_CV_SPLITS
    avg_prec /= K_CV_SPLITS
    avg_rec /= K_CV_SPLITS
    avg_roc_auc /= K_CV_SPLITS
    avg_f1 /= K_CV_SPLITS

    return avg_acc, avg_prec, avg_rec, avg_roc_auc, avg_f1

def main():
    #load_BioAge1HO()
    
    #getting and splitting data
    print("parsing data")
    patients, X, y = get_data()
    #training models with different parameters

    start_time = timer()
    model = LR_classifier()
    test_model(model, X, y)
    end_time = timer()
    runtime = start_time - end_time
   
if __name__ == "__main__":
    main()
