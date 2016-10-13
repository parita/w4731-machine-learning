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
    labels[labels == 0] = -1
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

def unigram_logidf(tf):
    tfidf_transformer = TfidfTransformer(smooth_idf = False)
    X = tfidf_transformer.fit_transform(tf);
    idf = tfidf_transformer.idf_
    log_idf = (idf - 1)/math.log(10);
    return log_idf

def cross_validate(data, labels, kfolds, n):
    # Size of Test Data
    sample = n/kfolds;
    # Size of Training Data
    trsize = n * (kfolds - 1) / kfolds
    error_rate = 0
    
    cv = KFold(len(data), n_folds=kfolds, indices=True)
    ifold = 0
    for traincv, testcv in cv:
        print "Fold", ifold
        err_fold = 0
        
        # Compute Features
        vectorizer = CountVectorizer(tokenizer = tokenize)
        print "Computing Features ..."
        tf, tokens = unigram_tf(data[traincv], vectorizer)
        logidf = unigram_logidf(tf)

        # Train for Averaged Online Perceptron
        W = online_perceptron_train(tf, logidf, labels[traincv], trsize, len(tokens))
        
        # Get error rate on test data
        tf_test = unigram_tf_transform(data[testcv], vectorizer)
        logidf_test = unigram_logidf(tf_test)
        logidf_test[logidf_test == np.inf] = 0
        err_fold = online_perceptron_test(tf_test, logidf_test, labels[testcv], W, sample) 
        print "Fold", ifold, "error rate:", err_fold, "%" 
        error_rate = error_rate + err_fold
        ifold = ifold + 1

    error_rate = np.float(error_rate) / np.float(kfolds)
    print "Error:", error_rate, "%"

def online_perceptron_train(tf, logidf, y, n, d):
    wt = np.zeros([d + 1,])
    tf = csr_matrix(tf)
    range_n = range(n)
    print "First Pass"
    shuffle(range_n)
    count = 0
    for i in range_n:
        if (count % 5000 == 0):
            print "Processing", count
        count = count + 1
        xtf = tf.getrow(i).toarray()
        x = xtf.flatten() * logidf # tf-idf = tf x log10(idf)
        x = np.append(x, 1)
        if (wt.dot(x)*y[i] <= 0):
            wt = wt + y[i]*x
    # Second Pass
    W = np.zeros([d + 1,])
    print "Second Pass"
    shuffle(range_n)
    count = 0
    for i in range_n:
        if (count % 5000 == 0):
            print "Processing", count
        count = count + 1
        xtf = tf.getrow(i).toarray()
        x = xtf.flatten() * logidf # tf-idf = tf x log10(idf)
        x = np.append(x, 1)
        if (wt.dot(x)*y[i] <= 0):
            wt = wt + y[i]*x
        W = W + wt
    W = W / (n + 1)
    return W

def online_perceptron_test(test_tf, test_logidf, testlabels, W, n):
    err = 0.0
    test_tf = csr_matrix(test_tf)
    for i in range(n):
        if (i % 5000 == 0):
            print "Testing", i
        xtf = test_tf.getrow(i).toarray()
        x = xtf.flatten() * test_logidf # tf-idf = tf x log10(idf)
        x = np.append(x, 1)
        y = testlabels[i]
        if (np.dot(W, x)*y <= 0):
            err = err + 1
    return (err * 100.0 / n)

if __name__ == "__main__":
    filename = 'reviews_tr.csv'
    print "Loading data ..."
    data, labels = load_data()
    print "Cross-validating ..."
    cross_validate(data, labels, 5, 200000)
