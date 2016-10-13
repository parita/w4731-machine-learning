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
    labels[labels == 0] = -1
    print np.shape(data)
    print np.shape(labels)
    return data, labels, data.shape[0]

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
    labels[labels == 0] = -1
    print np.shape(data)
    print np.shape(labels)
    return data, labels, data.shape[0]

if __name__ == "__main__":
    filename = 'reviews_tr.csv'
    tfilename = 'reviews_te.csv'
    print "Loading training data ..."
    data, labels, ndata = get_data_labels(filename)
    print "Loading test data ..."
    testdata, testlabels, ntestdata = get_test_data_labels(tfilename)
    print "Saving data ..."
    save_data(data, labels, testdata, testlabels)
