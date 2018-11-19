import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from format import get_data

def main():
    #getting and splitting data
    print "parsing data"
    patients, egm_matrix, cancer_onehot = get_data()
    train_X, test_X, train_Y, test_Y = train_test_split(egm_matrix,cancer_onehot, test_size=0.20, random_state=0)

    #logistic regression on original data
    print "LR predicting"
    lr_O = LogisticRegression(solver='lbfgs', max_iter=1000)
    lr_O.fit(train_X, train_Y)
    print lr_O.score(test_X, test_Y)
    
    #PCA transform
    print "PCA scaling"
    scaler = StandardScaler()
    scaler.fit(train_X)
    print "PCA fitting"
    pca = PCA()
    pca.fit(train_X)
    print "PCA transforming"
    print "num components %f" % pca.n_components_
    train_X = pca.transform(train_X)
    test_X = pca.transform(test_X)
    #logistic regression on PCA (PCR)
    print "LR fitting"
    lr = LogisticRegression(solver='lbfgs', max_iter=1000)
    lr.fit(train_X, train_Y)
    print "LR predicting"
    preds = lr.predict(test_X)


    print lr.score(test_X, test_Y)


if __name__ == "__main__":
    main()
