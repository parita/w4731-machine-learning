from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from random import shuffle
import numpy as np

def get_data_labels(filename):
    data = np.loadtxt(filename, skiprows = 1, delimiter = ',', usecols = [1], dtype = 'S')
    labels = np.loadtxt(filename, skiprows = 1, delimiter = ',', usecols = [0])
    print np.shape(data)
    print np.shape(labels)
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
        print "Fold", ifold
        vectorizer = CountVectorizer(tokenizer = tokenize)
        # Compute Features
        print "Computing Features ..."
        tf, tokens = unigram_tf(data[traincv], vectorizer)
        test_tf = unigram_tf_transform(data[testcv], vectorizer)

        print "Training Classifier ..."
        nb = MultinomialNB(alpha = 0)
        nb.fit(tf, labels[traincv])
        preds = nb.predict(test_tf)
        correct = preds * labels[testcv]
        err = (correct <= 0).sum() / np.float(sample) * 100.0
        print "Fold", ifold, "error rate:", err, "%"
        error_rate += err
        ifold = ifold + 1
    error_rate = np.float(error_rate) / np.float(kfolds)
    print "Error:", error_rate, "%"

if __name__ == "__main__":
    filename = 'reviews_tr.csv'
    #data, labels = get_data_labels(filename)
    #data = data[1:200000]
    #labels = labels[1:200000]
    print "Loading data ..."
    data = np.load('data.npy')
    labels = np.load('labels.npy')
    labels[labels == 0] = -1
    # tf, tokens = unigram_tf(data[1:100])
    print "Cross-validating ..."
    # cross_validate(data[0:10], labels[0:10], 5, 10)
    cross_validate(data, labels, 5, 200000)
