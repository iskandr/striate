from pycuda import gpuarray, driver as cuda, autoinit
import numpy as np
import cudaconv2
from pycuda import cumath
from util import *

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

  def update(self):
    pass

  def scaleLearningRate(self, l):
    pass

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
      if type(getattr(self, att)) != type(self.__init__):
        d[att] = getattr(self, att)
    return d


class ConvLayer(Layer):
  def __init__(self , name, filter_shape, image_shape,  padding = 2, stride = 1, initW = 0.01, initB =
      0.0, epsW = 0.001, epsB = 0.002, bias = None, weight = None):
    Layer.__init__(self, name, 'conv')

    self.filterSize = filter_shape[2]
    self.numFilter = filter_shape[0]

    self.batchSize, self.numColor, self.imgSize, _ = image_shape
    self.padding = padding
    self.stride = stride
    self.initW = initW
    self.initB = initB
    self.epsW = epsW
    self.epsB = epsB

    self.outputSize = 1 + int(((2 * self.padding + self.imgSize - self.filterSize) / float(self.stride)))
    self.modules = self.outputSize ** 2

    if weight is None:
      self.filter = gpuarray.to_gpu(np.random.randn(self.filterSize * self.filterSize *
        self.numColor, self.numFilter) * self.initW).astype(np.float32)
    else:
      self.filter = gpuarray.to_gpu(weight).astype(np.float32)

    if bias is None:
      self.bias = gpuarray.to_gpu(np.random.randn(self.numFilter, 1) * initB).astype(np.float32)
    else:
      self.bias = gpuarray.to_gpu(bias).astype(np.float32)

    self.filterGrad = gpuarray.zeros_like(self.filter)
    self.biasGrad = gpuarray.zeros_like(self.bias)

  def dump(self):
    d = Layer.dump(self)
    del d['filterGrad'], d['biasGrad'] , d['tmp']
    d['filter'] = self.filter.get()
    d['bias'] = self.bias.get()
    return d


  def get_single_img_size(self):
    return self.modules * self.numFilter

  def get_output_shape(self):
    self.outputShape = (self.batchSize, self.numFilter, self.outputSize, self.outputSize)
    return self.outputShape


  def fprop(self, input, output):
    cudaconv2.convFilterActs(input, self.filter, output, self.imgSize, self.outputSize,
        self.outputSize, -self.padding, self.stride, self.numColor, 1)

    self.tmp = gpuarray.empty((self.numFilter, self.get_single_img_size() * self.batchSize/self.numFilter), dtype=np.float32)
    gpu_copy_to(output, self.tmp)
    add_vec_to_rows(self.tmp, self.bias)
    gpu_copy_to(self.tmp, output)

    if PFout:
      printMatrix(output, self.name)

  def bprop(self, grad, input, output, outGrad):
    cudaconv2.convImgActs(grad, self.filter, outGrad, self.imgSize, self.imgSize,
        self.outputSize, -self.padding, self.stride, self.numColor, 1, 0.0, 1.0)
    #bprop weight
    self.filterGrad.fill(0)
    cudaconv2.convWeightActs(input, grad, self.filterGrad, self.imgSize, self.outputSize,
        self.outputSize, self.filterSize, -self.padding, self.stride, self.numColor, 1, 0, 1, 1)
    #bprop bias
    self.biasGrad.fill(0)
    gpu_copy_to(grad,self.tmp)
    add_row_sum_to_vec(self.biasGrad, self.tmp)

    if PBout:
      printMatrix(outGrad, self.name)

  def update(self):
    self.filter = self.filter.mul_add(1, self.filterGrad, self.epsW / self.batchSize)
    self.bias = self.bias.mul_add(1, self.biasGrad, self.epsB /self.batchSize)

  def scaleLearningRate(self, lr):
    self.epsW *= lr
    self.epsB *= lr

class MaxPoolLayer(Layer):
  def __init__(self,  name, image_shape,  poolSize = 2, stride = 2, start = 0):
    Layer.__init__(self, name, 'pool')
    self.poolSize = poolSize
    self.stride = stride
    self.start = start

    self.batchSize, self.numColor, self.imgSize, _  = image_shape

    self.outputSize = ceil(self.imgSize - self.poolSize -self.start, self.stride) + 1

  def get_output_shape(self):
    self.outputShape = (self.batchSize, self.numColor, self.outputSize, self.outputSize)
    return self.outputShape

  def fprop(self, input, output):
    cudaconv2.convLocalMaxPool(input, output, self.numColor, self.poolSize, self.start, self.stride,
        self.outputSize)

    if PFout:
      printMatrix(output, self.name)

  def bprop(self, grad, input, output, outGrad):
    cudaconv2.convLocalMaxUndo(input, grad, output, outGrad, self.poolSize,
        self.start, self.stride, self.outputSize, 0.0, 1.0)

    if PBout:
      printMatrix(outGrad, self.name)

class ResponseNormLayer(Layer):
  def __init__(self, name, image_shape, pow = 0.75, size = 9, scale = 0.001):
    Layer.__init__(self, name, 'rnorm')
    self.batchSize,self.numColor, self.imgSize, _ = image_shape

    self.pow = pow
    self.size = size
    self.scale = scale
    self.denom = None

  def get_output_shape(self):
    self.outputShape = (self.batchSize, self.numColor, self.imgSize, self.imgSize)
    return self.outputShape

  def fprop(self, input, output):
    self.denom = gpuarray.zeros_like(input)
    cudaconv2.convResponseNorm(input, self.denom, output, self.numColor, self.size, self.scale,
        self.pow)

    if PFout:
      printMatrix(output, self.name)


  def bprop(self, grad,input, output, outGrad):
    cudaconv2.convResponseNormUndo(grad, self.denom, input, output, outGrad, self.numColor,
        self.size, self.scale, self.pow, 0.0, 1.0)

    if PBout:
      printMatrix(outGrad, self.name)

  def dump(self):
    d = Layer.dump(self)
    del d['denom']
    return d

class FCLayer(Layer):
  def __init__(self, name, input_shape, n_out, epsW=0.001, epsB=0.002, initW = 0.01, initB = 0.0, weight =
      None, bias = None):
    Layer.__init__(self, name, 'fc')
    self.epsW = epsW
    self.epsB = epsB
    self.initW = initW
    self.initB = initB

    self.inputSize, self.batchSize = input_shape
    self.outputSize = n_out

    self.weightShape = (self.outputSize, self.inputSize)
    if weight is None:
      self.weight = gpuarray.to_gpu(np.random.randn(*self.weightShape) *
          self.initW).astype(np.float32)
    else:
      self.weight = gpuarray.to_gpu(weight).astype(np.float32)

    if bias is None:
      self.bias = gpuarray.to_gpu(np.random.randn(self.outputSize, 1) *
          self.initB).astype(np.float32)
    else:
      self.bias = gpuarray.to_gpu(bias).astype(np.float32)
    self.weightGrad = gpuarray.zeros_like(self.weight)
    self.biasGrad = gpuarray.zeros_like(self.bias)


  def dump(self):
    d = Layer.dump(self)
    del d['weightGrad'], d['biasGrad']
    d['weight'] = self.weight.get()
    d['bias'] = self.bias.get()
    return d

  def get_output_shape(self):
    self.outputShape = (self.batchSize, self.outputSize, 1, 1)
    return self.outputShape

  def fprop(self, input, output ):
    gpu_copy_to( output.mul_add(0, dot(self.weight, input), 1), output)
    add_vec_to_rows(output, self.bias)

    if PFout:
      printMatrix(output, self.name)

  def bprop(self, grad, input, output, outGrad):
    gpu_copy_to(outGrad.mul_add(0, dot(transpose(self.weight), grad), 1), outGrad)
    self.weightGrad = self.weightGrad.mul_add(0, dot(grad, transpose(input)), 1)
    add_row_sum_to_vec(self.biasGrad, grad, alpha = 0.0)

    if PBout:
      printMatrix(outGrad, self.name)

  def update(self):
    self.weight = self.weight.mul_add(1, self.weightGrad, self.epsW / self.batchSize)
    self.bias = self.bias.mul_add(1, self.biasGrad, self.epsB / self.batchSize)

  def scaleLearningRate(self, l):
    self.epsW *= l
    self.epsB *= l


class SoftmaxLayer(Layer):
  def __init__(self, name, input_shape):
    Layer.__init__(self, name, "softmax")
    self.inputSize, self.batchSize = input_shape
    self.outputSize = self.inputSize
    self.cost = gpuarray.zeros((self.batchSize, 1), dtype = np.float32)
    self.batchCorrect = 0

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

    if PFout:
      printMatrix(output, self.name)

  def logreg_cost(self, label, output):
    maxid = gpuarray.zeros((self.batchSize, 1), dtype = np.float32)
    find_col_max_id(maxid, output)
    self.batchCorrect = same_reduce(label , maxid)

    logreg_cost_col_reduce(output, label, self.cost)

  def bprop(self, label, input, output, outGrad):
    softmax_bprop(output, label, outGrad)

    if PBout:
      printMatrix(outGrad, self.name)

  def get_correct(self):
    return  1.0 * self.batchCorrect / self.batchSize

  def dump(self):
    d = Layer.dump(self)
    del d['cost']
    return d


class Neuron:
  def __init__(self): pass

  def activate(self, input, output):
    assert('No Implementation of Activation')

  def computeGrad(self, grad, output, inputGrad):
    assert('No Implementation of Gradient')

class ReluNeuron(Neuron):
  def __init__(self):
    Neuron.__init__(self)

  def activate(self, input, output):
    relu_activate(input, output)

  def computeGrad(self, grad, output, outGrad):
    relu_compute_grad(grad, output, outGrad)


neuronDict = {'relu': lambda : ReluNeuron(), }


class NeuronLayer(Layer):
  def __init__(self, name, image_shape,  type = 'relu'):
    Layer.__init__(self, name, type)
    self.neuron = neuronDict[type]()
    self.batchSize, self.numColor, self.imgSize, _= image_shape


  def get_output_shape(self):
    self.outputShape = (self.batchSize, self.numColor, self.imgSize, self.imgSize)
    return self.outputShape

  def fprop(self, input, output):
    self.neuron.activate(input, output)

    if PFout:
      printMatrix(output, self.name)

  def bprop(self, grad, input, output, outGrad):
    self.neuron.computeGrad(grad, output, outGrad)
    if PBout:
      printMatrix(outGrad, self.name)

  def dump(self):
    d = Layer.dump(self)
    del d['neuron']
    return d
