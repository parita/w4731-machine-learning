from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from random import shuffle
import pickle
import numpy as np
import math

def get_data_labels(filename):
    data = np.loadtxt(filename, skiprows = 1, delimiter = ',', usecols = [1], dtype = 'S')
    labels = np.loadtxt(filename, skiprows = 1, delimiter = ',', usecols = [0])
    print np.shape(data)
    print np.shape(labels)
    return data, labels

def load_data():
    data = np.load('data.npy')
    labels = np.load('labels.npy')
    return data, labels

def load_test_data():
    testdata = np.load('testdata.npy')
    testlabels = np.load('testlabels.npy')
    return testdata, testlabels

def unigram_tf(data, vectorizer):
    tf = vectorizer.fit_transform(data)
    tokens = vectorizer.get_feature_names()
    return tf, tokens

def unigram_tf_transform(data, vectorizer):
    tf = vectorizer.transform(data)
    return tf

def cross_validate(data, labels, kfolds, n):
    # Size of Test Data
    sample = n/kfolds;
    # Size of Training Data
    trsize = n * (kfolds - 1) / kfolds
    error_rate = 0
    
    cv = KFold(len(data), n_folds=kfolds, indices=True)
    ifold = 0
    for traincv, testcv in cv:
        err_fold = 0
        vectorizer = CountVectorizer()
        # Compute Features
        print "Computing Features ..."
        tf, tokens = unigram_tf(data[traincv], vectorizer)

        print "Fold", ifold
        rng = np.random.RandomState(1)
        regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 10), n_estimators=100, random_state=1)

        tf_split = 30
        tf_size = np.shape(tf)[0]/tf_split
        for idx_tf in range(tf_split):
            print "From", np.shape(tf)[0], "processing range:", idx_tf*tf_size, ", ", (idx_tf + 1)*tf_size
            tf_part = tf[idx_tf*tf_size:(idx_tf + 1)*tf_size]
            print np.shape(tf_part)
            labels_part = labels[traincv];
            labels_part = labels[idx_tf*tf_size:(idx_tf + 1)*tf_size]
            regr.fit(tf_part.toarray(), labels_part);

        # Get error rate on test data
        tf_test = unigram_tf_transform(data[testcv], vectorizer)
        err_fold = regr_test(tf_test.toarray(), labels[testcv], regr, sample) 
        print "Fold", ifold, "error rate:", err_fold, "%" 
        error_rate = error_rate + err_fold
        ifold = ifold + 1

    error_rate = np.float(error_rate) / np.float(kfolds)
    print "Error:", error_rate, "%"

def train_and_test(data, labels, testdata, testlabels, n):
    error_rate = 0
    
    vectorizer = CountVectorizer()
    # Compute Features
    print "Training complete dataset ..."
    tf, tokens = unigram_tf(data, vectorizer)

    rng = np.random.RandomState(1)
    regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 10), n_estimators=100, random_state=rng)
    tf_split = 30
    tf_size = np.shape(tf)[0]/tf_split
    for idx_tf in range(tf_split):
        print "From", np.shape(tf)[0], "processing range:", idx_tf*tf_size, ", ", (idx_tf + 1)*tf_size
        tf_part = tf[idx_tf*tf_size:(idx_tf + 1)*tf_size]
        labels_part = labels[idx_tf*tf_size:(idx_tf + 1)*tf_size]
        regr.fit(tf_part.toarray(), labels_part);

    print "Computing error rate ..."
    # Get error rate on test data
    tf_test = unigram_tf_transform(testdata, vectorizer)
    error_rate = regr_test(tf_test, testlabels, regr, n) 
    print "Error:", error_rate, "%"

def regr_test(testfeatures, testlabels, regr, n):
    err = 0.0
    print "Testing ..."
    err = 0.0
    testfeatures = csr_matrix(testfeatures)
    for i in range(n):
        if (i % 5000 == 0):
            print "Testing", i
        x = testfeatures.getrow(i).toarray()
        y = testlabels[i]
        predictlabel = regr.predict(x)
        err = err + np.sum(predictlabel != y)
    return (err * 100.0 / n)

if __name__ == "__main__":
    filename = 'reviews_tr.csv'
    print "Loading data ..."
    data, labels = load_data()
    testdata, testlabels = load_test_data()
    data = data
    labels = labels
    testdata = testdata
    testlabels = testlabels
    ndata = data.shape[0]
    print "Cross-validating ..."
    cross_validate(data, labels, 5, ndata)
    ntestdata = testdata.shape[0]
    train_and_test(data, labels, testdata, testlabels, ntestdata);
