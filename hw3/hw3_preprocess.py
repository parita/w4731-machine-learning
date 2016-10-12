from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from scipy.sparse import csr_matrix
import pickle
import numpy as np
import math

def get_data_labels(filename):
    data = np.loadtxt(filename, skiprows = 1, delimiter = ',', usecols = [1], dtype = 'S')
    labels = np.loadtxt(filename, skiprows = 1, delimiter = ',', usecols = [0])
    print np.shape(data)
    print np.shape(labels)
    return data, labels

def tokenize(s):
    return s.split()

def unigram_tf(data):
    vectorizer = CountVectorizer(tokenizer = tokenize)
    tf = vectorizer.fit_transform(data)
    tokens = vectorizer.get_feature_names()
    return tf, tokens

def unigram_tfidf(tf):
    tfidf_transformer = TfidfTransformer(smooth_idf = False)
    X = tfidf_transformer.fit_transform(tf);
    idf = tfidf_transformer.idf_
    log_idf = (idf - 1)/math.log(10);
    return log_idf

def bigram_tf(data):
    vectorizer = CountVectorizer(tokenizer = tokenize, ngram_range = (2, 2))
    tf = vectorizer.fit_transform(data)
    tokens = vectorizer.get_feature_names()
    return tf, tokens

def cross_validate(data, labels, kfolds, n):
    sample = n/kfolds;
    for ifold in range(kfolds):
        print "Folding ", ifold
        ileft = (ifold+1)*sample - sample;
        trbegin = ifold*sample
        trend = min(n, (ifold + kfolds - 1)*sample)
        trend2 = max(0, (ifold - 1)*sample)
        tebegin = max(0, (ifold - 1)*sample)
        teend = ifold*sample
        training_data = np.append(data[trbegin:trend], data[0:trend2])
        training_labels = np.append(data[trbegin:trend], data[0:trend2])
        test_data = np.append(data[trend:n], data[tebegin:teend])
        test_labels = np.append(data[trend:n], data[tebegin:teend])
        tf, tokens = unigram_tf(training_data)
        # tf = np.concatenate((tf, np.ones([np.shape(tf)[0], 1])), axis = 1)
        with open('tf_uni_' + str(ifold) + '.pkl', 'w') as f:
            pickle.dump(tf, f)
        trsize = n * (kfolds - 1) / kfolds
        print len(tokens)
        W = online_perceptron_train(tf, labels, trsize, len(tokens))
        np.save('Classifier', W)

def online_perceptron_train(features, y, n, nfeatures):
    W = np.zeros([1, nfeatures + 1])
    wt = W[0, :]
    features = csr_matrix(features)
    for i in range(n):
        if (i % 5000 == 0):
            print "Processing", i
        x = features.getrow(i).toarray()
        x = np.append(x, 1)
        if (np.dot(wt, x)*y[i] <= 0):
            c = y[i]
            if (y[i] == 0): 
                c = -1
            wt = wt + c*x
        W = W + wt
    W = W / (2*n + 1)
    return W

def online_perceptron_test(tokens, testfeatures, testlabels, n):
    pass

if __name__ == "__main__":
    filename = 'reviews_tr.csv'
    #data, labels = get_data_labels(filename)
    #data = data[1:200000]
    #labels = labels[1:200000]
    print "Loading data ..."
    data = np.load('data.npy')
    labels = np.load('labels.npy')
    # tf, tokens = unigram_tf(data[1:100])
    print "Cross-validating ..."
    cross_validate(data, labels, 5, 200000)

    #np.save('tf', tf.toarray())
    #np.save('features', tokens)
    #log_idf = unigram_tfidf(tf)
    #np.save('log_idf', log_idf)
    #bitf, bitokens = bigram_tf(data[1:200000])
    #np.save('bitf', bitf.toarray())
    #np.save('bigram_features', bitokens)
