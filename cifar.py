# EXPECTS THE CIFAR-100 dataset to be in same directory 
# as 'train' and 'test' pickled files 

import cPickle
import numpy 
import numpy as np




def unpickle(filename):
    with open(filename, 'rb') as f:
        d = cPickle.load(f)
    return d

def load_data(train='cifar-100/train', test='cifar-100/test', labels = 'fine_labels', npixels = 32):

  # meta = unpickle('meta')
  train = unpickle(train)


  xtrain = train['data']
  ytrain = np.array(train[labels]).astype('int32')
  print "Loaded training set, dims =", xtrain.shape
  ntrain = xtrain.shape[0]
  xtrain_reshape = np.reshape(xtrain, (ntrain, 3, npixels,npixels), order='C').astype('float32') / 256

  test = unpickle(test)
  xtest = test['data']
  ytest = np.array(test[labels]).astype('int32')
  ntest = xtest.shape[0]
  xtest_reshape = np.reshape(xtest, (ntest, 3, npixels,npixels), order='C').astype('float32') / 256
  print "Loaded test set, dims =", xtest.shape
  return xtrain_reshape, ytrain, xtest_reshape, ytest

