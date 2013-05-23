import numpy as np

import theano
import theano.tensor as T 
from theano.misc.pycuda_utils import to_cudandarray, to_gpuarray 
from theano.sandbox.cuda import CudaNdarray

import pycuda 
import pycuda.autoinit
import pycuda.curandom 
from pycuda.gpuarray import GPUArray 

from mlp import HiddenLayer
from logistic_sgd import LogisticRegression 
from conv_pool_layer import ConvPoolLayer 
from params_list import ParamsList 


rng = np.random.RandomState(23455)


class ConvNet(object): 
  def __init__(self,  
                     mini_batch_size, 
                     learning_rate, 
                     momentum = 0.0, 
                     n_out = 10, 
                     input_height = 32, 
                     input_width = 32, 
                     n_colors = 3, 
                     filter_size = (5,5),
                     pool_size = (2,2), 
                     conv_activation = 'relu',
                     n_filters = [64, 64], 
                     n_hidden = [200, 100, 50, 25]): 
    
    self.mini_batch_size = mini_batch_size  
    self.momentum = momentum 
    self.learning_rate = learning_rate
    # allocate symbolic variables for the data
    x = T.tensor4('x') # matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels


    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '  >> Building model: mini_batch_size = %d, learning_rate = %s, momentum = %s, n_filters = %s' % (mini_batch_size, learning_rate, momentum, n_filters)

    # Reshape matrix of rasterized images 
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    conv0_input = x.reshape((mini_batch_size, n_colors, input_height, input_width))
    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
    # maxpooling reduces this further to (24/2,24/2) = (12,12)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
    pool_size = (2,2)
    filter_size = (5,5) 
    conv0 = ConvPoolLayer(rng, input=conv0_input,
            image_shape=(mini_batch_size, n_colors, input_height, input_width),
            filter_shape=(n_filters[0], n_colors, filter_size[0], filter_size[1]), 
            poolsize=pool_size, activation = conv_activation)
    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
    # maxpooling reduces this further to (8/2,8/2) = (4,4)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
    conv1 = ConvPoolLayer(rng, input=conv0.output,
            image_shape=(mini_batch_size, n_filters[0], 14, 14),
            filter_shape=(n_filters[1], n_filters[0], filter_size[0], filter_size[1]), 
            poolsize=pool_size, activation = conv_activation)
    # the TanhLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    
    if isinstance(n_hidden, int):
      n_hidden = [n_hidden]
    
    hidden_layers = []
    last_output = conv1.output.flatten(2)
    last_output_size = n_filters[1] * filter_size[0] * filter_size[1]
    for n in n_hidden:
      # construct a fully-connected sigmoidal layer
      layer = HiddenLayer(rng, 
                  input=last_output, 
                  n_in=last_output_size,
                  n_out=n, activation=T.tanh)
      last_output = layer.output
      last_output_size = n
      hidden_layers.append(layer)

    # classify the values of the fully-connected sigmoidal layer
    output_layer = LogisticRegression(
                     input=last_output, 
                     n_in=last_output_size, 
                    n_out=n_out)

    # the cost we minimize during training is the NLL of the model
    self.cost = output_layer.negative_log_likelihood(y)
    # create a function to compute the mistakes that are made by the model
    self.test_model = theano.function([x,y], output_layer.errors(y)) 
    self.params = output_layer.params + conv1.params + conv0.params
    for layer in hidden_layers:
      self.params.extend(layer.params)

    # create a list of gradients for all model parameters
    self.grads = T.grad(self.cost, self.params)
    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i],grads[i]) pairs.
    updates = []
    for param_i, grad_i in zip(self.params, self.grads):
        updates.append((param_i, param_i - learning_rate * grad_i))
    # WARNING: We are going to overwrite the gradients!
    borrowed_grads = [theano.Out(g, borrow=True) for g in self.grads ]
    self.bprop_return_grads = theano.function([x, y], borrowed_grads)
    self.bprop_update_return_grads = theano.function([x, y], borrowed_grads, updates = updates) 
    self.bprop_update_return_cost = theano.function([x, y], self.cost, updates = updates) 
    self.return_cost = theano.function([x, y], self.cost)

  def get_weights_list(self):
    return [p.get_value(borrow=True) for p in self.params]

  def get_weights(self):
    return ParamsList(self.get_weights_list()).flatten()
 
  def add_list_to_weights(self, dxs):
    """
    Given a ragged list of arrays, add each to network params
    """
    for p, dx in zip(self.params, dxs):
      old_value = p.get_value(borrow = True)
      old_value += dx 
      p.set_value(old_value, borrow=True)


  def set_weights(self, new_w):
    curr_idx = 0
    assert isinstance(new_w, (CudaNdarray, GPUArray))
    for p in self.params:
      w = p.get_value(borrow=True, return_internal_type=True)
      nelts = 1 if np.isscalar(w) else w.size 
      if isinstance(w, (CudaNdarray, GPUArray)):
        new_reshaped = new_w[curr_idx:curr_idx+nelts].reshape(w.shape)
        nbytes = new_reshaped.nbytes if hasattr(new_reshaped, 'nbytes') else 4 * nelts 
        pycuda.driver.memcpy_dtod(w.gpudata, new_reshaped.gpudata, nbytes)
        # p.set_value(pycuda.gpuarray.to_gpu(new_reshaped), borrow=True)
      else:
        assert np.isscalar(w), "Expected scalar, got %s" % type(w) 
        p.set_value(np.array(new_w[curr_idx:curr_idx+1])[0])
      curr_idx += nelts 
    assert curr_idx == len(new_w)
  

  def get_gradients_list(self, xslice, yslice):
    return [to_gpuarray(g_elt, copyif=True)
            for g_elt in self.bprop_return_grads(xslice, yslice)]

  def get_gradients(self, xslice, yslice):
    return ParamsList(self.get_gradients_list(xslice, yslice), copy_first=False).flatten()

  def average_gradients(self, x, y):
    """
    Get the average gradient across multiple mini-batches
    """
    n_batches = x.shape[0] / self.mini_batch_size
    if n_batches == 1:
      return self.get_gradients(x,y)
    combined = ParamsList(copy_first=False)
    for batch_idx in xrange(n_batches):
      start = batch_idx*self.mini_batch_size
      stop = start + self.mini_batch_size
      xslice = x[start:stop]
      yslice = y[start:stop]
      gs = self.get_gradient_list(xslice, yslice)
      combined.iadd(gs)
    g = combined.flatten() 
    g *= (1.0 / combined.n_updates)
    return g 
 
  def get_state(self, x, y):
    return self.get_weights(), self.average_gradients(x,y)
  
 
  def for_each_slice(self, x, y, fn, mini_batch_size = None):
    if mini_batch_size is None:
      mini_batch_size = self.mini_batch_size
    results = []
    for mini_batch_idx in xrange(x.shape[0] / mini_batch_size):
      start = mini_batch_idx * mini_batch_size 
      stop = start + mini_batch_size 
      xslice = x[start:stop]
      yslice = y[start:stop]
      result = fn(xslice, yslice)
      if result is not None:
        results.append(result)
    if len(results) > 0:
      return results 

  def average_cost(self, x, y):
    costs = self.for_each_slice(x,y,self.return_cost)  
    return np.mean(costs)
  def average_error(self, x, y):
    errs = self.for_each_slice(x,y,self.test_model)
    return np.mean(errs)
    
  def update_batches(self, x, y, average=False):
    """
    Returns list containing most recent gradients
    """
    g_sum = ParamsList(copy_first=False)
    def fn(xslice, yslice):
      if average:
        g_list = self.bprop_update_return_grads(xslice, yslice)
        g_sum.iadd(g_list)
        del g_list 
      else:
        self.bprop_update_return_cost(xslice, yslice)
    self.for_each_slice(x, y, fn)

    if average:
      g = g_sum.flatten() 
      g *= (1.0 / g_sum.n_updates)
      return g 
