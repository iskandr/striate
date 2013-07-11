from pycuda import gpuarray, driver as cuda, autoinit
import numpy as np
import cudaconv2
from pycuda import cumath
from util import *
from cuda_kernel import *

import sys

PFout = False
PBout = False
TEST = 0
TRAIN = 1

class Layer(object):
  def __init__(self, name, type):
    self.name = name
    self.type = type
    self.diableBprop = False

  def fprop(self, input, output):
    assert False, "No implementation for fprop"

  def bprop(self, grad, input, output, outGrad):
    assert False, "No implementation for bprop"

  def disableBprop(self):
    self.diableBprop = True

  def get_output_shape(self):
    assert False, 'No implementation for getoutputshape'

  def change_batch_size(self, batch_size):
    self.batchSize = batch_size

  def dump(self):
    d = {}
    attr = [att for att in dir(self) if not att.startswith('__')]
    for att in attr:
      if type(getattr(self, att)) != type(self.__init__) and type(getattr(self, att))!= type(lambda:1):
        d[att] = getattr(self, att)
    return d



class WeightedLayer(Layer):
  def __init__(self, name, type, epsW, epsB, initW, initB, weight, bias, weightShape, biasShape):
    Layer.__init__(self, name, type)

    self.epsW = epsW
    self.epsB = epsB
    self.initW = initW
    self.initB = initB

    if weight is None:
      self.weight = gpuarray.to_gpu(np.random.randn(*weightShape) *
          self.initW).astype(np.float32)
    else:
      self.weight = gpuarray.to_gpu(weight).astype(np.float32)

    if bias is None:
      self.bias = gpuarray.to_gpu(np.random.randn(*biasShape) *
          self.initB).astype(np.float32)
    else:
      self.bias = gpuarray.to_gpu(bias).astype(np.float32)
    self.weightGrad = gpuarray.zeros_like(self.weight)
    self.biasGrad = gpuarray.zeros_like(self.bias)


  def update(self):
    matrix_add(self.weight, self.weightGrad, beta = self.epsW/ self.batchSize)
    matrix_add(self.bias, self.biasGrad, beta = self.epsB /self.batchSize)

  def scaleLearningRate(self, l):
    self.epsW *= l
    self.epsB *= l

class ConvLayer(WeightedLayer):
  def __init__(self , name, filter_shape, image_shape,  padding = 2, stride = 1, initW = 0.01, initB =
      0.0, epsW = 0.001, epsB = 0.002, bias = None, weight = None):

    self.filterSize = filter_shape[2]
    self.numFilter = filter_shape[0]
    self.imgShape = image_shape

    self.batchSize, self.numColor, self.imgSize, _ = image_shape
    self.padding = padding
    self.stride = stride

    self.outputSize = 1 + ceil(2 * self.padding + self.imgSize - self.filterSize, self.stride)
    self.modules = self.outputSize ** 2

    self.weightShape = (self.filterSize * self.filterSize * self.numColor, self.numFilter)
    self.biasShape = (self.numFilter, 1)
    WeightedLayer.__init__(self, name, 'conv', epsW, epsB, initW, initB, weight, bias,
        self.weightShape, self.biasShape)

  @staticmethod
  def parseFromFASTNET(ld):
    numFilter = ld['numFilter']
    filterSize = ld['filterSize']
    numColor = ld['numColor']
    padding = ld['padding']
    stride = ld['stride']
    initW = ld['initW'] if 'initW' in ld else 0.01
    initB = ld['initB'] if 'initB' in ld else 0.00
    epsW = ld['epsW'] if 'epsW' in ld else 0.001
    epsB = ld['epsB'] if 'epsB' in ld else 0.002
    bias  = ld['bias'] if 'bias' in ld else None
    weight = ld['weight'] if 'weight' in ld else None
    name = ld['name']
    filter_shape = (numFilter, numColor, filterSize, filterSize)
    img_shape = ld['imgShape']
    return ConvLayer(name, filter_shape, img_shape, padding, stride, initW, initB, epsW, epsB,
        bias, weight)

  @staticmethod
  def parseFromCUDACONVNET(ld):
    numFilter = ld['filters']
    filterSize = ld['filterSize'][0]
    numColor = ld['channels'][0]
    padding = -ld['padding'][0]
    stride = ld['stride'][0]
    initW = ld['initW'][0]
    initB = ld['initB']
    name = ld['name']
    epsW = ld['epsW'][0]
    epsB = ld['epsB']

    imgSize = ld['imgSize']

    bias = ld['biases']
    weight = ld['weights'][0]

    filter_shape = (numFilter, numColor, filterSize, filterSize)
    img_shape = self.imgShapes[-1]
    return ConvLayer(name, filter_shape, img_shape, padding, stride, initW, initB, epsW, epsB, bias,
        weight)

  def dump(self):
    d = WeightedLayer.dump(self)
    del d['weightGrad'], d['biasGrad'] , d['tmp']
    d['weight'] = self.weight.get()
    d['bias'] = self.bias.get()
    return d


  def get_single_img_size(self):
    return self.modules * self.numFilter

  def get_output_shape(self):
    self.outputShape = (self.batchSize, self.numFilter, self.outputSize, self.outputSize)
    return self.outputShape


  def fprop(self, input, output):
    cudaconv2.convFilterActs(input, self.weight, output, self.imgSize, self.outputSize,
        self.outputSize, -self.padding, self.stride, self.numColor, 1)

    self.tmp = gpuarray.empty((self.numFilter, self.get_single_img_size() * self.batchSize/self.numFilter), dtype=np.float32)
    gpu_copy_to(output, self.tmp)
    add_vec_to_rows(self.tmp, self.bias)
    gpu_copy_to(self.tmp, output)

  def bprop(self, grad, input, output, outGrad):
    cudaconv2.convImgActs(grad, self.weight, outGrad, self.imgSize, self.imgSize,
        self.outputSize, -self.padding, self.stride, self.numColor, 1, 0.0, 1.0)
    #bprop weight
    self.weightGrad.fill(0)
    cudaconv2.convWeightActs(input, grad, self.weightGrad, self.imgSize, self.outputSize,
        self.outputSize, self.filterSize, -self.padding, self.stride, self.numColor, 1, 0, 1, 1)
    #bprop bias
    self.biasGrad.fill(0)
    gpu_copy_to(grad,self.tmp)
    add_row_sum_to_vec(self.biasGrad, self.tmp)


class MaxPoolLayer(Layer):
  def __init__(self,  name, image_shape,  poolSize = 2, stride = 2, start = 0):
    Layer.__init__(self, name, 'pool')
    self.poolSize = poolSize
    self.stride = stride
    self.start = start
    self.imgShape = image_shape

    self.batchSize, self.numColor, self.imgSize, _  = image_shape

    self.outputSize = ceil(self.imgSize - self.poolSize -self.start, self.stride) + 1

  @staticmethod
  def parseFromFASTNET(ld):
    stride = ld['stride']
    start = ld['start']
    poolSize = ld['poolSize']
    img_shape = ld['imgShape']
    name = ld['name']
    return MaxPoolLayer(name, img_shape, poolSize, stride, start)

  @staticmethod
  def parseFromCUDACONVNET(ld):
    stride = ld['stride']
    start = ld['start']
    poolSize = ld['sizeX']
    img_shape = ld['imgShape']
    name = ld['name']
    return MaxPoolLayer(name, img_shape, poolSize, stride, start)


  def get_output_shape(self):
    self.outputShape = (self.batchSize, self.numColor, self.outputSize, self.outputSize)
    return self.outputShape

  def fprop(self, input, output):
    cudaconv2.convLocalMaxPool(input, output, self.numColor, self.poolSize, self.start, self.stride,
        self.outputSize)

  def bprop(self, grad, input, output, outGrad):
    cudaconv2.convLocalMaxUndo(input, grad, output, outGrad, self.poolSize,
        self.start, self.stride, self.outputSize, 0.0, 1.0)

class ResponseNormLayer(Layer):
  def __init__(self, name, image_shape, pow = 0.75, size = 9, scale = 0.001):
    Layer.__init__(self, name, 'rnorm')
    self.batchSize,self.numColor, self.imgSize, _ = image_shape
    self.imgShape = image_shape

    self.pow = pow
    self.size = size
    self.scale = scale
    self.denom = None

  @staticmethod
  def parseFromFASTNET(ld):
    name = ld['name']
    pow = ld['pow']
    size = ld['size']
    scale = ld['scale']
    image_shape = ld['imgShape']
    return ResponseNormLayer(name, image_shape, pow, size, scale)

  @staticmethod
  def parseFromCUDACONVNET(ld):
    return ResponseNormLayer.parseFromFASTNET(ld)


  def get_output_shape(self):
    self.outputShape = (self.batchSize, self.numColor, self.imgSize, self.imgSize)
    return self.outputShape

  def fprop(self, input, output):
    self.denom = gpuarray.zeros_like(input)
    cudaconv2.convResponseNorm(input, self.denom, output, self.numColor, self.size, self.scale,
        self.pow)


  def bprop(self, grad,input, output, outGrad):
    cudaconv2.convResponseNormUndo(grad, self.denom, input, output, outGrad, self.numColor,
        self.size, self.scale, self.pow, 0.0, 1.0)

  def dump(self):
    d = Layer.dump(self)
    del d['denom']
    return d

class FCLayer(WeightedLayer):
  def __init__(self, name, input_shape, n_out, epsW=0.001, epsB=0.002, initW = 0.01, initB = 0.0, weight =
      None, bias = None):
    self.inputShape = input_shape
    self.inputSize, self.batchSize = input_shape

    self.outputSize = n_out

    self.weightShape = (self.outputSize, self.inputSize)
    self.biasShape = (self.outputSize, 1)
    WeightedLayer.__init__(self, name, 'fc', epsW, epsB, initW, initB, weight, bias, self.weightShape,
        self.biasShape)


  @staticmethod
  def parseFromFASTNET(ld):
    epsB = ld['epsB'] if 'epsB' in ld else 0.002
    epsW = ld['epsW'] if 'epsW' in ld else 0.001
    initB = ld['initB'] if 'initB' in ld else 0.00
    initW = ld['initW'] if 'initW' in ld else 0.01

    n_out = ld['outputSize']
    bias = ld['bias'] if 'bias' in ld else None
    weight = ld['weight'] if 'weight' in ld else None
    name = ld['name']
    input_shape = ld['inputShape']
    return FCLayer(name, input_shape, n_out, epsW, epsB, initW, initB, weight, bias)

  @staticmethod
  def parseFromCUDACONVNET(ld):
    epsB = ld['epsB']
    epsW = ld['epsW'][0]
    initB = ld['initB']
    initW = ld['initW'][0]

    n_out = ld['outputs']
    bias = ld['biases']
    weight = ld['weights'][0].transpose()
    name = ld['name']
    input_shape = ld['inputShape']
    return FCLayer(name, input_shape, n_out, epsW, epsB, initW, initB, weight, bias)


  def dump(self):
    d = WeightedLayer.dump(self)
    del d['weightGrad'], d['biasGrad']
    d['weight'] = self.weight.get()
    d['bias'] = self.bias.get()
    return d

  def get_output_shape(self):
    self.outputShape = (self.batchSize, self.outputSize, 1, 1)
    return self.outputShape

  def fprop(self, input, output ):
    gpu_copy_to(dot(self.weight, input), output)
    add_vec_to_rows(output, self.bias)

  def bprop(self, grad, input, output, outGrad):
    gpu_copy_to(dot(transpose(self.weight), grad), outGrad)
    self.weightGrad = dot(grad, transpose(input))
    add_row_sum_to_vec(self.biasGrad, grad, alpha = 0.0)


class SoftmaxLayer(Layer):
  def __init__(self, name, input_shape):
    Layer.__init__(self, name, "softmax")
    self.inputShape = input_shape
    self.inputSize, self.batchSize = input_shape
    self.outputSize = self.inputSize
    self.cost = gpuarray.zeros((self.batchSize, 1), dtype = np.float32)
    self.batchCorrect = 0

  @staticmethod
  def parseFromFASTNET(ld):
    name = ld['name']
    input_shape = ld['inputShape']
    return SoftmaxLayer(name, input_shape)

  @staticmethod
  def parseFromCUDACONVNET(ld):
    return SoftmaxLayer.parseFromFASTNET(ld)

  def get_output_shape(self):
    self.outputShape = (self.batchSize, self.outputSize, 1, 1)
    return self.outputShape

  def fprop(self, input, output):
    max = gpuarray.zeros((1, self.batchSize), dtype = np.float32)
    col_max_reduce(max, input)
    add_vec_to_cols(input, max, output, alpha = -1)
    gpu_copy_to(cumath.exp(output), output)
    sum = gpuarray.zeros(max.shape, dtype = np.float32)
    add_col_sum_to_vec(sum, output, alpha = 0)
    div_vec_to_cols(output, sum)


  def logreg_cost(self, label, output):
    maxid = gpuarray.zeros((self.batchSize, 1), dtype = np.float32)
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
  def __init__(self, name, image_shape,  type = 'relu', a = 1.0, b = 1.0, e = 0.0):
    Layer.__init__(self, name, 'neuron')
    self.imgShape = image_shape
    if type == 'relu':
      self.neuron = ReluNeuron(e)
    elif type == 'tanh':
      self.neuron = TanhNeuron(a, b)
    self.batchSize, self.numColor, self.imgSize, _= image_shape

  @staticmethod
  def parseFromFASTNET(ld):
    name = ld['name']
    img_shape = ld['imgShape']
    if ld['neuron'] == 'relu':
      e = ld['e']
      return NeuronLayer(name, img_shape, type = 'relu', e = e)

    if ld['neuron'] == 'tanh':
      a = ld['a']
      b = ld['b']
      return NeuronLayer(name, imag_shape, type = 'tanh', a = a, b= b)

    assert False, 'No implementation for the neuron type' + ld['neuron']['type']


  @staticmethod
  def parseFromCUDACONVNET(ld):
    if ld['neuron']['type'] == 'relu':
      img_shape = ld['imgShape']
      name = ld['name']
      e = ld['neuron']['e']
      return NeuronLayer(name, img_shape, type = 'relu', e = e)
    if ld['neuron']['type'] == 'tanh':
      name = ld['name']
      img_shape = ld['imgShape']
      a = ld['neuron']['a']
      b = ld['neuron']['b']
      return NeuronLayer(name, img_shape, 'tanh', a = a, b = b)

    assert False, 'No implementation for the neuron type' + ld['neuron']['type']


  def get_output_shape(self):
    self.outputShape = (self.batchSize, self.numColor, self.imgSize, self.imgSize)
    return self.outputShape

  def fprop(self, input, output):
    self.neuron.activate(input, output)

  def bprop(self, grad, input, output, outGrad):
    self.neuron.computeGrad(grad, output, outGrad)

  def dump(self):
    d = Layer.dump(self)
    for k,v in self.neuron.dump().items():
      d[k] = v
    return d
