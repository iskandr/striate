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
                     batch_size,
                     learning_rate,
                     momentum=0.0,
                     input_size=(32, 32),
                     n_colors=3,
                     n_out=10,
                     n_filters=(64, 64),
                     filter_size=(5, 5),
                     pool_size=(2, 2),
                     conv_activation='relu',
                     n_hidden=(200, 100, 50, 25)): 
    
    self.batch_size = batch_size  
    self.momentum = momentum 
    self.learning_rate = learning_rate
    # allocate symbolic variables for the data
    x = T.tensor4('x')  # matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels


    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '  >> Building model: batch_size = %d, learning_rate = %s, momentum = %s, n_filters = %s' % (batch_size, learning_rate, momentum, n_filters)

    input_height, input_width = input_size 
    pool_height, pool_width = pool_size
    filter_height, filter_width = filter_size 
    last_output_shape = (batch_size, n_colors, input_height, input_width)
    
    # Reshape matrix of rasterized images 
    # to a 4D tensor, compatible with our LConvPoolLayer
    last_output = x.reshape(last_output_shape)
    
    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
    # maxpooling reduces this further to (24/2,24/2) = (12,12)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
    
    conv_layers = []
    last_n_filters = n_colors
    for curr_n_filters in n_filters:
      
      conv_layer = ConvPoolLayer(rng, input=last_output,
                    image_shape=last_output_shape,
                    filter_shape=(curr_n_filters, last_n_filters, filter_height, filter_width),
                    poolsize=pool_size, activation=conv_activation)
      last_output = conv_layer.output
      
      out_height = (last_output_shape[2] - filter_height + 1) / pool_height

      out_width = (last_output_shape[3] - filter_width + 1) / pool_width
      last_output_shape = (batch_size, curr_n_filters, out_height, out_width)
      last_output = conv_layer.output 
      conv_layers.append(conv_layer) 
      last_n_filters = curr_n_filters
    
    # the TanhLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    
    if isinstance(n_hidden, int):
      n_hidden = [n_hidden]
    
    hidden_layers = []
    last_output = last_output.flatten(2)
    last_output_size = last_output_shape[1] * last_output_shape[2] * last_output_shape[3]
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
    self.test_model = theano.function([x, y], output_layer.errors(y)) 
    self.params = output_layer.params
    for layer in hidden_layers:
      self.params.extend(layer.params)
    for layer in conv_layers:
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
    self.bprop_no_update = theano.function([x, y], [self.cost] + borrowed_grads)
    self.bprop_update = theano.function([x, y], [self.cost] + borrowed_grads, updates=updates) 
    # self.bprop_update_return_cost = theano.function([x, y], self.cost, updates = updates) 
    # self.return_cost = theano.function([x, y], self.cost)

 
  def get_weights_list(self):
    return [p.get_value(borrow=True) for p in self.params]

  def get_weights(self):
    return ParamsList(self.get_weights_list()).flatten()
 
  def add_list_to_weights(self, dxs):
    """
    Given a ragged list of arrays, add each to network params
    """
    for p, dx in zip(self.params, dxs):
      old_value = p.get_value(borrow=True)
      old_value += dx 
      p.set_value(old_value, borrow=True)


  def set_weights(self, new_w):
    curr_idx = 0
    assert isinstance(new_w, (CudaNdarray, GPUArray))
    for p in self.params:
      w = p.get_value(borrow=True, return_internal_type=True)
      nelts = 1 if np.isscalar(w) else w.size 
      if isinstance(w, (CudaNdarray, GPUArray)):
        new_reshaped = new_w[curr_idx:curr_idx + nelts].reshape(w.shape)
        nbytes = new_reshaped.nbytes if hasattr(new_reshaped, 'nbytes') else 4 * nelts 
        pycuda.driver.memcpy_dtod(w.gpudata, new_reshaped.gpudata, nbytes)
        # p.set_value(pycuda.gpuarray.to_gpu(new_reshaped), borrow=True)
      else:
        assert np.isscalar(w), "Expected scalar, got %s" % type(w) 
        p.set_value(np.array(new_w[curr_idx:curr_idx + 1])[0])
      curr_idx += nelts 
    assert curr_idx == len(new_w)
  

  
  def get_gradients_list(self, xslice, yslice):
    grads_list = self.batch_grads(xslice, yslice)
    return [to_gpuarray(g_elt, copyif=True) if not np.isscalar(g_elt) else g_elt
            for g_elt in grads_list]
    
  def get_gradients(self, xslice, yslice):
    return ParamsList(self.get_gradients_list(xslice, yslice), copy_first=False).flatten()

  def average_gradients(self, x, y):
    """
    Get the average gradient across multiple mini-batches
    """
    n_batches = x.shape[0] / self.batch_size
    if n_batches == 1:
      return self.get_gradients(x, y)
    combined = ParamsList(copy_first=False)
    for batch_idx in xrange(n_batches):
      start = batch_idx * self.batch_size
      stop = start + self.batch_size
      xslice = x[start:stop]
      yslice = y[start:stop]
      gs = self.get_gradient_list(xslice, yslice)
      combined.iadd(gs)
    g = combined.flatten() 
    g *= (1.0 / combined.n_updates)
    return g 
 
  def get_state(self, x, y):
    return self.get_weights(), self.average_gradients(x, y)
  
 
  def for_each_slice(self, x, y, fn, batch_size=None):
    if batch_size is None:
      batch_size = self.batch_size
    results = []
    for batch_idx in xrange(x.shape[0] / batch_size):
      start = batch_idx * batch_size 
      stop = start + batch_size 
      xslice = x[start:stop]
      if y is None:
        result = fn(xslice)
      else:
        yslice = y[start:stop]
        result = fn(xslice, yslice)
      if result is not None:
        results.append(result)
    if len(results) > 0:
      return results 

  def batch_cost(self, xslice, yslice):
    outputs = self.bprop_no_update(xslice, yslice)
    return outputs[0]
  
  def batch_grads(self, xslice, yslice):
    outputs = self.bprop_no_update(xslice, yslice)
    return outputs[1:]
  
  def average_cost(self, x, y):
    costs = self.for_each_slice(x, y, self.batch_cost)  
    return np.mean(costs)
  def average_error(self, x, y):
    errs = self.for_each_slice(x, y, self.test_model)
    return np.mean(errs)
    
  def predict(self, x):
    assert False, "Not yet implemented" 
    batch_outputs = self.for_each_slice(x, None, self.fprop)
    return np.array(batch_outputs)

  def fit(self, x, y,
          n_epochs=1,
          shuffle=False,
          return_average_gradient=False):
    
    if return_average_gradient:
      g_sum = ParamsList(copy_first=False)
      def fn(xslice, yslice):
        outputs = self.bprop_update(xslice, yslice)
        g_sum.iadd(outputs[1:])
        del outputs 
    else:
      costs = []
      def fn(xslice, yslice):
       outputs = self.bprop_update(xslice, yslice)
       costs.append(outputs[0])
    for _ in xrange(n_epochs):
      if shuffle:
        shuffle_indices = np.arange(len(y))
        np.random.shuffle(shuffle_indices)
        x = x.take(shuffle_indices, axis=0)
        y = y.take(shuffle_indices)
      self.for_each_slice(x, y, fn)
    
    if return_average_gradient:
      g = g_sum.flatten() 
      g *= (1.0 / g_sum.n_updates)
      return g 
    else:
      return np.mean(costs)
    
