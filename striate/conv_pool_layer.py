"""
Stolen whole cloth from the Theano LeNet5 tutorial.
"""

import numpy as np 

import theano 
import theano.tensor as T 
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.misc.pycuda_utils import to_cudandarray, to_gpuarray 
from theano.sandbox.cuda import CudaNdarray

import pycuda
import pycuda.autoinit 

class ConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), activation='relu'):
        """
        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        float_t = getattr(np, theano.config.floatX)
        W_init = to_cudandarray(pycuda.curandom.rand(shape=filter_shape, dtype=float_t))
        W_init *= (2*W_bound)
        W_init -= W_bound 
        self.W = theano.shared(W_init, borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = to_cudandarray(pycuda.gpuarray.zeros( shape = (filter_shape[0],), dtype=float_t))
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        if activation == 'tanh':
          self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        else:
          assert activation == 'relu'
          self.output = T.maximum(0, pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')) 

        # store parameters of this layer
        self.params = [self.W, self.b]
