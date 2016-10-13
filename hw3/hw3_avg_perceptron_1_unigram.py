from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
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

def tokenize(s):
    return s.split()

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
        vectorizer = CountVectorizer(tokenizer = tokenize)
        # Compute Features
        print "Computing Features ..."
        tf, tokens = unigram_tf(data[traincv], vectorizer)

        print "Fold", ifold
        # Train for Averaged Online Perceptron
        W = online_perceptron_train(tf, labels[traincv], trsize, len(tokens))
        
        # Get error rate on test data
        tf_test = unigram_tf_transform(data[testcv], vectorizer)
        err_fold = online_perceptron_test(tf_test, labels[testcv], W, sample) 
        print "Fold", ifold, "error rate:", err_fold, "%" 
        error_rate = error_rate + err_fold
        ifold = ifold + 1

    error_rate = np.float(error_rate) / np.float(kfolds)
    print "Error:", error_rate, "%"

def online_perceptron_train(features, y, n, nfeatures):
    wt = np.zeros([nfeatures + 1,])
    features = csr_matrix(features)
    range_n = range(n)
    # First Pass
    print "First Pass"
    shuffle(range_n)
    count = 0
    for i in range_n:
        if (count % 5000 == 0):
            print "Processing", count
        count = count + 1
        x = features.getrow(i).toarray()
        x = np.append(x, 1)
        if (wt.dot(x)*y[i] <= 0):
            wt = wt + y[i]*x
    # Second Pass
    W = np.zeros([nfeatures + 1,])
    print "Second Pass"
    shuffle(range_n)
    count = 0
    for i in range_n:
        if (count % 5000 == 0):
            print "Processing", count
        count = count + 1
        x = features.getrow(i).toarray()
        x = np.append(x, 1)
        if (wt.dot(x)*y[i] <= 0):
            wt = wt + y[i]*x
        W = W + wt
    W = W / (n + 1)
    return W

def online_perceptron_test(testfeatures, testlabels, W, n):
    err = 0.0
    testfeatures = csr_matrix(testfeatures)
    for i in range(n):
        if (i % 5000 == 0):
            print "Testing", i
        x = testfeatures.getrow(i).toarray()
        x = np.append(x, 1)
        y = testlabels[i]
        if (np.dot(W, x)*y <= 0):
            err = err + 1
    return (err * 100.0 / n)

if __name__ == "__main__":
    filename = 'reviews_tr.csv'
    print "Loading data ..."
    data, labels = load_data()
    ndata = data.shape[0]
    print "Cross-validating ..."
    cross_validate(data, labels, 5, ndata)
