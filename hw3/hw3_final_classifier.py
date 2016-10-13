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
    data = data[0:200000]
    labels = labels[0:200000]
    labels[labels == -1] = labels
    print np.shape(data)
    print np.shape(labels)
    return data, labels, len(data)

def save_data(data, labels, testdata, testlabels):
    np.save('data.npy', data)
    np.save('labels.npy', labels)
    np.save('testdata.npy', testdata)
    np.save('testlabels.npy', testlabels)

def load_data():
    data = np.load('data.npy')
    labels = np.load('labels.npy')
    testdata = np.load('testdata.npy')
    testlabels = np.load('testlabels.npy')
    return data, labels, testdata, testlabels

def get_test_data_labels(filename):
    data = np.loadtxt(filename, skiprows = 1, delimiter = ',', usecols = [1], dtype = 'S')
    labels = np.loadtxt(filename, skiprows = 1, delimiter = ',', usecols = [0])
    labels[labels == -1] = labels
    print np.shape(data)
    print np.shape(labels)
    return data, labels, len(data)

def tokenize(s):
    return s.split()

def unigram_tf(data, vectorizer):
    tf = vectorizer.fit_transform(data)
    tokens = vectorizer.get_feature_names()
    return tf, tokens

def unigram_tf_transform(data, vectorizer):
    tf = vectorizer.transform(data)
    return tf

def train_and_test_error_rate(data, labels, testdata, testlabels, ndata, ntestdata):
    vectorizer = CountVectorizer(tokenizer = tokenize)
    # Compute Features
    print "Computing Features ..."
    tf, tokens = unigram_tf(data, vectorizer)

    # Train for Averaged Online Perceptron
    W = online_perceptron_train(tf, labels, ndata, len(tokens))
    
    # Get error rate on training data
    tr_err = online_perceptron_test(tf, labels, W, ndata)
    print "Training Error Rate:", tr_err, "%"

    # Get error rate on test data
    tf_test = unigram_tf_transform(testdata, vectorizer)
    te_err = online_perceptron_test(tf_test, testlabels, W, ntestdata) 

    print "Test Error Rate:", te_err, "%"

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
    tfilename = 'reviews_te.csv'
    print "Loading data ..."
    data, labels, ndata = get_data_labels(filename)
    testdata, testlabels, ntestdata = get_test_data_labels(tfilename)
    train_and_test_error_rate(data, labels, testdata, testlabels, ndata, ntestdata)
