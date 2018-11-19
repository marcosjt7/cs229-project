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
    print egm_matrix
    print cancer_onehot
    train_X, test_X, train_Y, test_Y = train_test_split(egm_matrix,cancer_onehot, test_size=0.20, random_state=0)
    #PCA transform
    print "PCA scaling"
    scaler = StandardScaler()
    scaler.fit(train_X)
    print "PCA fitting"
    pca = PCA(0.95)
    pca.fit(train_X)
    print "PCA transforming"
    print "num components for .95: %f" % pca.n_components_
    train_X = pca.transform(train_X)
    test_X = pca.transform(test_X)
    #logistic regression
    print "LR fitting"
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(train_X, train_Y)
    print "LR predicting"
    preds = lr.predict(test_X)
    print preds



    
    '''
    pca_class = PCA(egm_matrix)
    pca_class.calculate_mu()
    print ("calc mu")
    pca_class.subtract_mean_from_examples()
    print ("subtracted mean")
    pca_class.calc_variance()
    print("calculated variance")
    pca_class.rescale_values_with_variance()
    print ("rescale values")
    pca_class.form_eigen_subparts()
    print("formed eigen")
    reduced_vectors = pca_class.form_new_vectors()
    np.savetxt("pca_vectors.txt", reduced_vectors)
    '''

if __name__ == "__main__":
    main()
