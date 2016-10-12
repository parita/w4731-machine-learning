from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
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
    return tf.toarray(), tokens

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
        np.save('tf_uni_' + str(ifold), tf)

if __name__ == "__main__":
    filename = 'reviews_tr.csv'
    #data, labels = get_data_labels(filename)
    #data = data[1:200000]
    #labels = labels[1:200000]
    data = load('data.npy')
    labels = load('labels.npy')
    tf, tokens = unigram_tf(data)
    np.save('tf', tf.toarray())
    np.save('features', tokens)
    log_idf = unigram_tfidf(tf)
    np.save('log_idf', log_idf)
    bitf, bitokens = bigram_tf(data[1:200000])
    np.save('bitf', bitf)
    np.save('bigram_features', bitokens)
