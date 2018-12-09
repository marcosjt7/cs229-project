import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from format import load_BioAge1HO
from format import get_data

'''constants'''
K_CV_SPLITS = 5

######################### MODELS #########################
def PCA_classifier(num_components):
    #PCR pipeline
    print "PCR:"
    steps = [
        ('scl', StandardScaler()),
        ('pca', PCA(n_components=num_components)),
        ('clf',LogisticRegression(solver='lbfgs', max_iter=1000, random_state=1))
    ]
    pcr_pipe = Pipeline(steps)
    return pcr_pipe

def LR_classifier():
    #simple logistic regression pipeline
    print "LR:"
    steps = [('clf', LogisticRegression(solver='lbfgs', max_iter=1000, random_state=1))]
    lr_pipe = Pipeline(steps)
    return lr_pipe

def tree_classifier():
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

    print avg_acc, avg_prec, avg_rec, avg_roc_auc, avg_f1

''''''''''''''''''''''''

def main():
    '''constants'''
    
    #load_BioAge1HO()
    
    #getting and splitting data
    print "parsing data"
    patients, egm_matrix, cancer_onehot = get_data()
    model = LR_classifier()
    test_model(model, egm_matrix, cancer_onehot)
    

   
if __name__ == "__main__":
    main()
