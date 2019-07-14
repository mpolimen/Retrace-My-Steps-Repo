import struct, os, gzip
import numpy as np
import time
from sklearn.neural_network import MLPClassifier
import pickle

# Code organization traces to 15-112 lecture on machine learning...
# https://github.com/tusharc97/15112_ML_LectureF17/blob/master/MNIST/
#MNISTsample.py

# MNIST database taken from http://yann.lecun.com/exdb/mnist.

DATAFILE = 'mnist' # Name of folder the .gz files are in.

# FILE READING

''' Adapted from:
https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40
Takes an MNIST .gz file and returns the information in nparray format.'''
def unzip(filename):
    with gzip.open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

''' Takes the folder that contains the MNIST .gz files and returns nparrays
corresponding to each file.'''
def readMNIST(datafile):
    trainData = unzip(os.path.join(datafile,'train-images-idx3-ubyte.gz'))
    trainLabels = unzip(os.path.join(datafile,'train-labels-idx1-ubyte.gz'))
    testData = unzip(os.path.join(datafile,'t10k-images-idx3-ubyte.gz'))
    testLabels = unzip(os.path.join(datafile,'t10k-labels-idx1-ubyte.gz'))
    return trainData,trainLabels,testData,testLabels

'''Takes an nparray representing MNIST images and returns the nparray containing
each flattened image (row-major). The resulant nparray is 2D.
Works for the .gz files.'''
def flatten(data):
    return np.reshape(data,(data.shape[0],data.shape[1]*data.shape[2]))

'''Takes an nparray representing MNIST labels and returns the nparray containing
each unrolled label. The ith element of the returned nparray is an nparray
containing 10 elements - one 1 at the index corresponding to the value of the
ith element of the input array and 0's everywhere else.'''
def unpack(data):
    newdata = np.zeros((data.shape[0],10))
    for i in range(len(data)):
        newdata[i][data[i]]=1
    return newdata

# Returns the neural network setup.
def nn():
    network=MLPClassifier(hidden_layer_sizes=(392,196,98,49),activation='logistic')
    return network

'''Runs the network on the training and test sets, and returns the network,
training and test accuracies after fitting.'''
def mnist():
    # Get data from .gz files we downloaded and train the network.
    trainData,trainLabels,testData,testLabels = readMNIST(DATAFILE)
    trainData,testData = flatten(trainData),flatten(testData)
    trainLabels,testLabels = unpack(trainLabels),unpack(testLabels)
    network = nn()
    network.fit(trainData,trainLabels)
    return network

if __name__=="__main__":
    network = mnist()
    print("Done")
    # Pickling code from...
    # https://machinelearningmastery.com/save-load-machine-learning-models-
    #python-scikit-learn/
    pickle.dump(network, open("finalized_model.sav", 'wb'))