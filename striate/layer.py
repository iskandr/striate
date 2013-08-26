from cuda_kernel import *
from pycuda import cumath, gpuarray, driver as cuda
from striate.util import *
import cudaconv2
import numpy as np
import sys

from striate import util


PFout = False
PBout = False
TEST = 0
TRAIN = 1

class Layer(object):
  def __init__(self, name, type):
    self.name = name
    self.type = type
    self.disableBprop = False

  def fprop(self, input, output, train=TRAIN):
    assert False, "No implementation for fprop"

  def bprop(self, grad, input, output, outGrad):
    assert False, "No implementation for bprop"

  def disableBprop(self):
    self.disableBprop = True

  def get_output_shape(self):
    assert False, 'No implementation for getoutputshape'

  def change_batch_size(self, batch_size):
    self.batchSize = batch_size

  def dump(self):
    d = {}
    attr = [att for att in dir(self) if not att.startswith('__')]
    for att in attr:
      if type(getattr(self, att)) != type(self.__init__) and type(getattr(self, att)) != type(lambda:1):
        d[att] = getattr(self, att)
    return d

def randn(shape, dtype):
  np.random.seed(0)
  return np.require(np.random.randn(*shape), dtype=dtype, requirements='C')
  #return np.random.randn(*shape).astype(dtype)

class WeightedLayer(Layer):
  def __init__(self, name, type, epsW, epsB, initW, initB, momW, momB, wc, weight, bias,
      weightIncr , biasIncr, weightShape, biasShape):
    Layer.__init__(self, name, type)

    self.epsW = F(epsW)
    self.epsB = F(epsB)
    self.initW = initW
    self.initB = initB
    self.momW = F(momW)
    self.momB = F(momB)
    self.wc = F(wc)

    if weight is None:
      self.weight = gpuarray.to_gpu(randn(weightShape, np.float32) * self.initW)
    else:
      print >> sys.stderr,  'init weight from disk'
      self.weight = gpuarray.to_gpu(weight)#.astype(np.float32)

    if bias is None:
      if self.initB > 0.0:
        self.bias = gpuarray.to_gpu((np.ones(biasShape, dtype=np.float32) * self.initB))
      else:
        self.bias = gpuarray.zeros(biasShape, dtype=np.float32)
    else:
      print >> sys.stderr,  'init bias from disk'
      self.bias = gpuarray.to_gpu(bias).astype(np.float32)

    self.weightGrad = gpuarray.zeros_like(self.weight)
    self.biasGrad = gpuarray.zeros_like(self.bias)
    if self.momW > 0.0:
      if weightIncr is None:
        self.weightIncr = gpuarray.zeros_like(self.weight)
      else:
        print >> sys.stderr,  'init weightIncr from disk'
        #weightIncr = np.require(weightIncr, dtype = np.float, requirements = 'C')
        self.weightIncr = gpuarray.to_gpu(weightIncr)
    if self.momW > 0.0:
      if biasIncr is None:
        self.biasIncr = gpuarray.zeros_like(self.bias)
      else:
        print >> sys.stderr,  'init biasIncr from disk'
        #biasIncr = np.require(biasIncr, dtype = np.float, requirements = 'C')
        self.biasIncr = gpuarray.to_gpu(biasIncr)


  def clear_weight_incr(self):
    self.weightIncr.fill(0)

  def clear_bias_incr(self):
    self.biasIncr.fill(0)

  def clear_incr(self):
    self.clear_weight_incr()
    self.clear_bias_incr()

  def update(self, weightGrad = None, biasGrad = None):

    if weightGrad is not None:
      self.weightGrad = weightGrad
    if biasGrad is not None:
      self.biasGrad = biasGrad

    if self.momW > 0.0:
      matrix_add(self.weightIncr, self.weightGrad, alpha=self.momW, beta=self.epsW / F(self.batchSize))
      matrix_add(self.weightIncr, self.weight, alpha=1, beta= F(-self.wc * self.epsW))
      matrix_add(self.weight, self.weightIncr)
    else:
      #self.weight += self.weightGrad * self.epsW / self.batchSize
      matrix_add(self.weight, self.weightGrad, alpha = 1, beta = self.epsW / F(self.batchSize))

    if self.momB > 0.0:
      matrix_add(self.biasIncr, self.biasGrad, alpha=self.momB, beta=self.epsB / F(self.batchSize))
      matrix_add(self.bias, self.biasIncr)
    else:
      #self.bias += self.biasGrad * self.epsB / self.batchSize
      matrix_add(self.bias, self.biasGrad, alpha = 1, beta = self.epsB / F(self.batchSize))


  def scaleLearningRate(self, l):
    self.epsW *= l
    self.epsB *= l

  def get_summary(self, type = 'mean'):
    w = self.weight.get()
    w = np.mean(np.abs(w))
    wi = 0.0

    b = self.bias.get()
    b = np.mean(np.abs(b))
    bi = 0.0
    return self.name, (w, wi, b, bi)


  def dump(self):
    d = Layer.dump(self)
    d['weight'] = self.weight.get()
    d['bias'] = self.bias.get()
    if 'weightIncr' in d:
      d['weightIncr'] = self.weightIncr.get()
    if 'biasIncr' in d:
      d['biasIncr'] = self.biasIncr.get()
    del d['weightGrad'], d['biasGrad']
    return d

class ConvLayer(WeightedLayer):
  def __init__(self , name, filter_shape, image_shape, padding=2, stride=1, initW=0.01, initB=
      0.0, partialSum = 0, sharedBiases = 0, epsW=0.001, epsB=0.002, momW=0.0, momB=0.0, wc=0.0, bias=None, weight=None, weightIncr = None, biasIncr = None):

    self.filterSize = filter_shape[2]
    self.numFilter = filter_shape[0]
    self.imgShape = image_shape

    self.numColor, self.imgSize, _, self.batchSize = image_shape
    self.padding = padding
    self.stride = stride

    self.partialSum = partialSum
    self.sharedBiases = sharedBiases

    self.outputSize = 1 + divup(2 * self.padding + self.imgSize - self.filterSize, self.stride)
    self.modules = self.outputSize ** 2

    self.weightShape = (self.filterSize * self.filterSize * self.numColor, self.numFilter)
    self.biasShape = (self.numFilter, 1)
    WeightedLayer.__init__(self, name, 'conv', epsW, epsB, initW, initB, momW, momB, wc, weight,
        bias, weightIncr, biasIncr, self.weightShape, self.biasShape)


  def dump(self):
    d = WeightedLayer.dump(self)
    if 'tmp' in d:
      del d['tmp']
    return d

  def get_cross_width(self):return self.filterSize - 1

  def get_single_img_size(self):
    return self.modules * self.numFilter

  def get_output_shape(self):
    self.outputShape = (self.numFilter, self.outputSize, self.outputSize, self.batchSize)
    return self.outputShape


  def fprop(self, input, output, train=TRAIN):
    cudaconv2.convFilterActs(input, self.weight, output, self.imgSize, self.outputSize,
        self.outputSize, -self.padding, self.stride, self.numColor, 1)
    self.tmp = gpuarray.empty((self.numFilter,
                               self.get_single_img_size() * self.batchSize / self.numFilter),
                              dtype=np.float32)
    gpu_copy_to(output, self.tmp)
    add_vec_to_rows(self.tmp, self.bias)
    gpu_copy_to(self.tmp, output)

    if PFout:
      print_matrix(output, self.name)

  def bprop(self, grad, input, output, outGrad):
    cudaconv2.convImgActs(grad, self.weight, outGrad, self.imgSize, self.imgSize,
        self.outputSize, -self.padding, self.stride, self.numColor, 1, 0.0, 1.0)
    # bprop weight
    self.weightGrad.fill(0)
    cudaconv2.convWeightActs(input, grad, self.weightGrad, self.imgSize, self.outputSize,
        self.outputSize, self.filterSize, -self.padding, self.stride, self.numColor, 1, 0, 0, 1)
    # bprop bias
    self.biasGrad.fill(0)
    gpu_copy_to(grad, self.tmp)
    add_row_sum_to_vec(self.biasGrad, self.tmp)


class MaxPoolLayer(Layer):
  def __init__(self, name, image_shape, poolSize=2, stride=2, start=0):
    Layer.__init__(self, name, 'pool')
    self.pool = 'max'
    self.poolSize = poolSize
    self.stride = stride
    self.start = start
    self.imgShape = image_shape

    self.numColor, self.imgSize, _, self.batchSize= image_shape

    self.outputSize = divup(self.imgSize - self.poolSize - self.start, self.stride) + 1

  def get_output_shape(self):
    self.outputShape = (self.numColor, self.outputSize, self.outputSize, self.batchSize)
    return self.outputShape


  def get_cross_width(self): return self.poolSize - 1

  def fprop(self, input, output, train=TRAIN):
    cudaconv2.convLocalMaxPool(input, output, self.numColor, self.poolSize, self.start, self.stride,
        self.outputSize)
    if PFout:
      print_matrix(output, self.name)

  def bprop(self, grad, input, output, outGrad):
    cudaconv2.convLocalMaxUndo(input, grad, output, outGrad, self.poolSize,
        self.start, self.stride, self.outputSize, 0.0, 1.0)

class AvgPoolLayer(Layer):
  def __init__(self, name, image_shape, poolSize=2, stride=2, start=0):
    Layer.__init__(self, name, 'pool')
    self.pool = 'avg'
    self.poolSize = poolSize
    self.stride = stride
    self.start = start
    self.imgShape = image_shape

    self.numColor, self.imgSize, _, self.batchSize= image_shape

    self.outputSize = divup(self.imgSize - self.poolSize - self.start, self.stride) + 1

  def get_output_shape(self):
    self.outputShape = (self.numColor, self.outputSize, self.outputSize, self.batchSize)
    return self.outputShape

  def get_cross_width(self): return self.poolSize - 1

  def fprop(self, input, output, train=TRAIN):
    cudaconv2.convLocalAvgPool(input, output, self.numColor, self.poolSize, self.start, self.stride,
        self.outputSize)
    if PFout:
      print_matrix(output, self.name)

  def bprop(self, grad, input, output, outGrad):
    cudaconv2.convLocalAvgUndo(grad, outGrad, self.poolSize,
        self.start, self.stride, self.outputSize, self.imgSize, 0.0, 1.0)

class ResponseNormLayer(Layer):
  def __init__(self, name, image_shape, pow=0.75, size=9, scale=0.001):
    Layer.__init__(self, name, 'rnorm')
    self.numColor, self.imgSize, _, self.batchSize= image_shape
    self.imgShape = image_shape

    self.pow = pow
    self.size = size
    self.scale = scale
    self.scaler = self.scale / self.size ** 2
    self.denom = None

  def get_output_shape(self):
    self.outputShape = (self.numColor, self.imgSize, self.imgSize, self.batchSize)
    return self.outputShape

  def fprop(self, input, output, train=TRAIN):
    self.denom = gpuarray.zeros_like(input)
    cudaconv2.convResponseNorm(input, self.denom, output, self.numColor, self.size, self.scaler,
        self.pow)
    if PFout:
      print_matrix(output, self.name)

  def get_cross_width(self): return self.size - 1

  def bprop(self, grad, input, output, outGrad):
    cudaconv2.convResponseNormUndo(grad, self.denom, input, output, outGrad, self.numColor,
        self.size, self.scaler, self.pow, 0.0, 1.0)

  def dump(self):
    d = Layer.dump(self)
    if 'denom' in d:
      del d['denom']
    return d


class CrossMapResponseNormLayer(ResponseNormLayer):
  def __init__(self, name, image_shape, pow=0.75, size=9, scale=0.001, blocked=False):
    ResponseNormLayer.__init__(self, name, image_shape, pow, size, scale)
    self.type = 'cmrnorm'
    self.scaler = self.scale / self.size
    self.blocked = blocked

  def get_cross_width(self): return self.size - 1

  def fprop(self, input, output, train=TRAIN):
    self.denom = gpuarray.zeros_like(input)
    cudaconv2.convResponseNormCrossMap(input, self.denom, output, self.numColor, self.size, self.scaler, self.pow, self.blocked)
    if PFout:
      print_matrix(output, self.name)

  def bprop(self, grad, input, output, outGrad):
    cudaconv2.convResponseNormCrossMapUndo(grad, self.denom, input, output, outGrad, self.numColor,
        self.size, self.scaler, self.pow, self.blocked, 0.0, 1.0)

  def dump(self):
    d = Layer.dump(self)
    if 'denom' in d:
      del d['denom']
    d['blocked'] = self.blocked
    return d

class FCLayer(WeightedLayer):
  def __init__(self, name, input_shape, n_out, epsW=0.001, epsB=0.002, initW=0.01, initB=0.0,
      momW=0.0, momB=0.0, wc=0.0, dropRate=0.0, weight=None, bias=None, weightIncr = None, biasIncr
      = None):
    self.inputShape = input_shape
    self.inputSize, self.batchSize = input_shape

    self.outputSize = n_out
    self.dropRate = dropRate

    self.weightShape = (self.outputSize, self.inputSize)
    self.biasShape = (self.outputSize, 1)
    WeightedLayer.__init__(self, name, 'fc', epsW, epsB, initW, initB, momW, momB, wc, weight,
        bias, weightIncr, biasIncr, self.weightShape, self.biasShape)
    util.log('%s dropRate: %s', self.name, self.dropRate)


  def dump(self):
    d = WeightedLayer.dump(self)
    '''
    weight = self.weight.get()
    if weight.shape[1] > 96 * 26 * 26:
      print 'weight of fc layer is too larget, split.....'
      weights = np.split(weight, 4)
    else:
      weights = weight
    d['weight'] = weights
    '''
    if 'dropMask' in d:
      del d['dropMask']
    return d

  def get_output_shape(self):
    self.outputShape = (self.outputSize, 1, 1, self.batchSize)
    return self.outputShape

  def fprop(self, input, output, train=TRAIN):
    gpu_copy_to(dot(self.weight, input), output)
    add_vec_to_rows(output, self.bias)

    if train == TEST:
      if self.dropRate > 0.0:
        output *= (1.0 - self.dropRate)
    else:
      if self.dropRate > 0.0:
        self.dropMask = gpuarray.to_gpu(np.random.uniform(0, 1, output.size).astype(np.float32).reshape(output.shape))
        bigger_than_scaler(self.dropMask, self.dropRate)
        gpu_copy_to(output * self.dropMask, output)

    if PFout:
      print_matrix(output, self.name)

  def bprop(self, grad, input, output, outGrad):
    if self.dropRate > 0.0:
      gpu_copy_to(grad * self.dropMask, grad)
    #gpu_copy_to(dot(transpose(self.weight), grad), outGrad)
    gpu_copy_to(transpose( dot( transpose(grad), self.weight ) ), outGrad )
    self.weightGrad = dot(grad, transpose(input))
    add_row_sum_to_vec(self.biasGrad, grad, alpha=0.0)



class SoftmaxLayer(Layer):
  def __init__(self, name, input_shape):
    Layer.__init__(self, name, "softmax")
    self.inputShape = input_shape
    self.inputSize, self.batchSize = input_shape
    self.outputSize = self.inputSize
    self.cost = gpuarray.zeros((self.batchSize, 1), dtype=np.float32)
    self.batchCorrect = 0

  def get_output_shape(self):
    self.outputShape = (self.outputSize, 1, 1, self.batchSize)
    return self.outputShape

  def fprop(self, input, output, train=TRAIN):
    max = gpuarray.zeros((1, self.batchSize), dtype=np.float32)
    col_max_reduce(max, input)
    add_vec_to_cols(input, max, output, alpha= -1)
    eltwise_exp(output)
    sum = gpuarray.zeros(max.shape, dtype=np.float32)
    add_col_sum_to_vec(sum, output, alpha=0)
    div_vec_to_cols(output, sum)
    if PFout:
      print_matrix(output, self.name)

  def logreg_cost(self, label, output):
    if self.cost.shape[0] !=  self.batchSize:
      self.cost = gpuarray.zeros((self.batchSize, 1), dtype=np.float32)
    maxid = gpuarray.zeros((self.batchSize, 1), dtype=np.float32)
    find_col_max_id(maxid, output)
    self.batchCorrect = same_reduce(label , maxid)
    logreg_cost_col_reduce(output, label, self.cost)

  def bprop(self, label, input, output, outGrad):
    softmax_bprop(output, label, outGrad)


  def get_correct(self):
    return  1.0 * self.batchCorrect / self.batchSize

  def dump(self):
    d = Layer.dump(self)
    del d['cost']
    return d


class Neuron:
  def __init__(self, type):
    self.type = type

  def activate(self, input, output):
    assert False, 'No Implementation of Activation'

  def computeGrad(self, grad, output, inputGrad):
    assert False, 'No Implementation of Gradient'

  def dump(self):
    return {'neuron': self.type}

class ReluNeuron(Neuron):
  def __init__(self, e):
    Neuron.__init__(self, 'relu')
    self.e = e;

  def activate(self, input, output):
    relu_activate(input, output, self.e)

  def computeGrad(self, grad, output, outGrad):
    relu_compute_grad(grad, output, outGrad, self.e)

  def dump(self):
    d = Neuron.dump(self)
    d['e'] = self.e
    return d

class TanhNeuron(Neuron):
  def __init__(self, a, b):
    Neuron.__init__(self, 'tanh')
    self.a, self.b = a, b

  def activate(self, input, output):
    tanh_activate(input, output, self.a , self.b)

  def computeGrad(self, grad, output, outGrad):
    tanh_compute_grad(grad, output, outGrad, self.a, self.b)

  def dump(self):
    d = Neuron.dump(self)
    d['a'] = self.a
    d['b'] = self.b
    return d

class NeuronLayer(Layer):
  def __init__(self, name, image_shape, type='relu', a=1.0, b=1.0, e=0.0):
    Layer.__init__(self, name, 'neuron')
    self.imgShape = image_shape
    if type == 'relu':
      self.neuron = ReluNeuron(e)
    elif type == 'tanh':
      self.neuron = TanhNeuron(a, b)
    self.numColor, self.imgSize, _, self.batchSize= image_shape

  def get_cross_width(self): return 0

  def get_output_shape(self):
    self.outputShape = (self.numColor, self.imgSize, self.imgSize, self.batchSize)
    return self.outputShape

  def fprop(self, input, output, train=TRAIN):
    self.neuron.activate(input, output)
    if PFout:
      print_matrix(output, self.name)

  def bprop(self, grad, input, output, outGrad):
    self.neuron.computeGrad(grad, output, outGrad)

  def dump(self):
    d = Layer.dump(self)
    for k, v in self.neuron.dump().items():
      d[k] = v
    return d
