'''
Created on Jun 11, 2013

@author: justin
'''

from layer import *
from pycuda import cumath, gpuarray, driver as cuda
from striate import util
from striate.util import *
import cudaconv2
import numpy as np
import sys
# from cuda_kernel import *

class FastNet(object):
  def __init__(self, learningRate, imgShape, numOutput, initModel=None, autoAdd=True):
    self.learningRate = learningRate
    self.batchSize, self.numColor, self.imgSize, _ = imgShape
    self.imgShapes = [imgShape]
    self.inputShapes = [(self.numColor * (self.imgSize ** 2), self.batchSize)]
    self.numOutput = numOutput
    self.layers = []
    self.outputs = []
    self.grads = []
    self.output = None

    self.numCase = self.cost = self.correct = 0.0

    self.numConv = 0
    if initModel:
      if 'model_state' in initModel:
        self.append_layers_from_dict(initModel['model_state']['layers'])
      else:
        self.append_layers_from_dict(initModel)  # param of layers
      return

    if autoAdd:  # for imagenet, use param file
      self.autoAddLayerForCifar10(numOutput)

  def makeLayerFromFASTNET(self, ld):
    if ld['type'] == 'conv':
      ld['imgShape'] = self.imgShapes[-1]
      return ConvLayer.parseFromFASTNET(ld)

    if ld['type'] == 'pool':
      ld['imgShape'] = self.imgShapes[-1]
      return MaxPoolLayer.parseFromFASTNET(ld)

    if ld['type'] == 'neuron':
      ld['imgShape'] = self.imgShapes[-1]
      return NeuronLayer.parseFromFASTNET(ld)

    if ld['type'] == 'fc':
      ld['inputShape'] = self.inputShapes[-1]
      return FCLayer.parseFromFASTNET(ld)

    if ld['type'] == 'softmax':
      ld['inputShape'] = self.inputShapes[-1]
      return SoftmaxLayer.parseFromFASTNET(ld)

    if ld['type'] == 'rnorm':
      ld['imgShape'] = self.imgShapes[-1]
      return ResponseNormLayer.parseFromFASTNET(ld)

    if ld['type'] == 'cmrnorm':
      ld['imgShape'] = self.imgShapes[-1]
      return CrossMapResponseNormLayer.parseFromFASTNET(ld)

  def makeLayerFromCUDACONVNET(self, ld):
    if ld['type'] == 'conv':
      ld['imgShape'] = self.imgShapes[-1]
      return ConvLayer.parseFromCUDACONVNET(ld)

    if ld['type'] == 'pool':
      return MaxPoolLayer.parseFromCUDACONVNET(ld)

    if ld['type'] == 'neuron':
      ld['imgShape'] = self.imgShapes[-1]
      return NeuronLayer.parseFromCUDACONVNET(ld)

    if ld['type'] == 'fc':
      ld['inputShape'] = self.inputShapes[-1]
      return FCLayer.parseFromCUDACONVNET(ld)

    if ld['type'] == 'softmax':
      ld['inputShape'] = self.inputShapes[-1]
      return SoftmaxLayer.parseFromCUDACONVNET(ld)

    if ld['type'] == 'rnorm':
      return ResponseNormLayer.parseFromCUDACONVNET(ld)

  def append_layers_from_dict(self, layers):
    for l in layers:
      layer = self.makeLayerFromFASTNET(l)
      if layer:
        self.append_layer(layer)

  def autoAddLayerForCifar10(self, n_out):
    conv1 = ConvLayer('conv1', filter_shape=(64, 3, 5, 5), image_shape=self.imgShapes[-1],
        padding=2, stride=1, initW=0.0001, epsW=0.001, epsB=0.002)
    conv1.scaleLearningRate(self.learningRate)
    self.append_layer(conv1)

    conv1_relu = NeuronLayer('conv1_neuron', self.imgShapes[-1], type='relu', e=0.0)
    self.append_layer(conv1_relu)

    pool1 = MaxPoolLayer('pool1', self.imgShapes[-1], poolSize=3, stride=2, start=0)
    self.append_layer(pool1)

    rnorm1 = ResponseNormLayer('rnorm1', self.imgShapes[-1], pow=0.75, scale=0.001, size=9)
    self.append_layer(rnorm1)

    conv2 = ConvLayer('conv2', filter_shape=(64, 64, 5, 5) , image_shape=self.imgShapes[-1],
        padding=2, stride=1, initW=0.01, epsW=0.001, epsB=0.002)
    conv2.scaleLearningRate(self.learningRate)
    self.append_layer(conv2)

    conv2_relu = NeuronLayer('conv2_neuron', self.imgShapes[-1], type='relu', e=0.0)
    self.append_layer(conv2_relu)

    rnorm2 = ResponseNormLayer('rnorm2', self.imgShapes[-1], pow=0.75, scale=0.001, size=9)
    self.append_layer(rnorm2)

    pool2 = MaxPoolLayer('pool2', self.imgShapes[-1], poolSize=3, start=0, stride=2)
    self.append_layer(pool2)

    fc1 = FCLayer('fc', self.inputShapes[-1], n_out)
    fc1.scaleLearningRate(self.learningRate)
    self.append_layer(fc1)

    softmax1 = SoftmaxLayer('softmax', self.inputShapes[-1])
    self.append_layer(softmax1)

  def add_parameterized_layers(self, n_filters=None, size_filters=None, fc_nout=[10]):
    if n_filters is None or n_filters == []:
      self.autoAddLayerForCifar10(fc_nout[-1])
    else:
      for i in range(len(n_filters)):
        prev = n_filters[i - 1] if i > 0 else self.imgShapes[-1][1]
        filter_shape = (n_filters[i], prev, size_filters[i], size_filters[i])
        conv = ConvLayer('conv' + str(self.numConv), filter_shape, self.imgShapes[-1])
        self.append_layer(conv)
        conv.scaleLearningRate(self.learningRate)

        neuron = NeuronLayer('neuron' + str(self.numConv), self.imgShapes[-1], type='tanh')
        self.append_layer(neuron)

        pool = MaxPoolLayer('pool' + str(self.numConv), self.imgShapes[-1])
        self.append_layer(pool)

        rnorm = ResponseNormLayer('rnorm' + str(self.numConv), self.imgShapes[-1])
        self.append_layer(rnorm)

      for i in range(len(fc_nout)):
        fc = FCLayer('fc' + str(i + 1), self.inputShapes[-1], fc_nout[-1])
        self.append_layer(fc)

      self.append_layer(SoftmaxLayer('softmax', self.inputShapes[-1]))

  def append_layer(self, layer):
    self.layers.append(layer)
    if layer.type == 'conv':
      self.numConv += 1

    outputShape = layer.get_output_shape()
    row = outputShape[1] * outputShape[2] * outputShape[3]
    col = outputShape[0]
    self.inputShapes.append((row, col))
    self.imgShapes.append(outputShape)

    self.outputs.append(gpuarray.zeros((row, col), dtype=np.float32))
    util.log('Allocating %s bytes', np.prod(self.inputShapes[-2]) * 4)
    self.grads.append(gpuarray.zeros(self.inputShapes[-2], dtype=np.float32))
    print 'append layer', layer.name, 'to network'
    print 'the output of the layer is', outputShape

  def del_layer(self):
    name = self.layers[-1]
    del self.layers[-1], self.inputShapes[-1], self.imgShapes[-1], self.outputs[-1], self.grads[-1]
    print 'delete layer', name
    print 'the last layer would be', self.layers[-1].name

  @staticmethod
  def split_conv_to_stack(conv_params):
    stack = []
    s = []
    for ld in conv_params:
      if ld['type'] in ['fc', 'softmax']:
        break
      elif ld['type'] == 'conv':
        if s != []:
          stack.append(s)
        s = [ld]
      else:
        s.append(ld)
    stack.append(s)
    return stack

  @staticmethod
  def split_fc_to_stack(fc_params):
    stack = []
    s = []
    for ld in fc_params:
      if ld['type'] == 'softmax':
        break
      elif ld['type'] == 'fc':
        if s != []:
          stack.append(s)
        s = [ld]
      else:
        s.append(ld)
    stack.append(s)
    return stack

  def fprop(self, data, probs, train=TRAIN):
    input = data
    for i in range(len(self.layers)):
      l = self.layers[i]
      l.fprop(input, self.outputs[i], train)
      input = self.outputs[i]

    # probs.shape = self.outputs[-1].shape
    gpu_copy_to(self.outputs[-1], probs)

  def bprop(self, data, label, prob, train=TRAIN):
    grad = label
    for i in range(1, len(self.layers) + 1):

      l = self.layers[-i]
      if l.diableBprop:
        return
      if i == len(self.layers):
        input = data
      else:
        input = self.outputs[-(i + 1)]
      output = self.outputs[-i]
      outGrad = self.grads[-i]
      l.bprop(grad, input, output, outGrad)
      grad = outGrad

  def update(self):
    for l in self.layers:
      if l.diableBprop or not isinstance(l, WeightedLayer):
        continue
      l.update()

  def adjust_learning_rate(self, factor=1.0):
    for layer in self.layers:
      if isinstance(layer, WeightedLayer):
        l.scaleLearningRate(factor)

  def get_cost(self, label, output):
    outputLayer = self.layers[-1]
    outputLayer.logreg_cost(label, output)
    return outputLayer.cost.get().sum(), outputLayer.batchCorrect

  def get_batch_information(self):
    cost = self.cost
    numCase = self.numCase
    correct = self.correct
    self.cost = self.numCase = self.correct = 0.0
    return cost / numCase , correct / numCase, int(numCase)

  def get_correct(self):
    outputLayer = self.layers[-1]
    return outputLayer.get_correct()

  def prepare_for_train(self, data, label):
    timer.start()
    input = data
    ########
    # The last minibatch of data_batch file may not be 1024
    ########
    if input.shape[1] != self.batchSize:
      self.batchSize = input.shape[1]
      for l in self.layers:
        l.change_batch_size(self.batchSize)
      self.inputShapes = None
      self.imgShapes = None
      self.outputs = []
      self.grads = []

      self.imgShapes = [(self.batchSize, self.numColor, self.imgSize, self.imgSize)]
      self.inputShapes = [(self.numColor * (self.imgSize ** 2), self.batchSize)]
      for layer in self.layers:
        outputShape = layer.get_output_shape()
        row = outputShape[1] * outputShape[2] * outputShape[3]
        col = outputShape[0]
        self.inputShapes.append((row, col))
        self.imgShapes.append(outputShape)

        self.outputs.append(gpuarray.zeros((row, col), dtype=np.float32))
        self.grads.append(gpuarray.zeros(self.inputShapes[-2], dtype=np.float32))

    timer.end('prepare transform')

    if not isinstance(data, GPUArray):
      self.data = gpuarray.to_gpu(data).astype(np.float32)
    else:
      timer.start()
      self.data = data
      timer.end('assignment')

    if not isinstance(label, GPUArray):
      self.label = gpuarray.to_gpu(label).astype(np.float32)
    else:
      self.label = label

    timer.end('prepare assignment')

    self.label.shape = (label.size, 1)
    self.numCase += input.shape[1]
    outputShape = self.inputShapes[-1]
    if self.output is None or self.output.shape != outputShape:
      self.output = gpuarray.zeros(outputShape, dtype=np.float32)

    timer.end('prepare end')

  def train_batch(self, data, label, train=TRAIN):
    self.prepare_for_train(data, label)
    # printMatrix(data, 'data')
    self.fprop(self.data, self.output, train)
    cost, correct = self.get_cost(self.label, self.output)
    self.cost += cost
    self.correct += correct
    if train == TRAIN:
      self.bprop(self.data, self.label, self.output)
      self.update()

  def get_dumped_layers(self):
    layers = []
    for l in self.layers:
      layers.append(l.dump())

    return layers

  def disable_bprop(self):
    for l in self.layers:
      l.disableBprop()

  def get_report(self):
    pass



class AdaptiveFastNet(FastNet):
  def __init__(self, learningRate, imgShape, numOutput, train, test, initModel=None, autoAdd=True):
    FastNet.__init__(self, learningRate, imgShape, numOutput, initModel, autoAdd)
    self.train_data, self.train_label = train
    self.test_data, self.test_label = test
    self.adjust_info = [(self.learningRate, 0, 0)]

  def adjust_learning_rate(self, factor):
    factors = factor
    train_data = self.train_data
    test_data = self.test_data
    train_label = self.train_label
    test_label = self.test_label

    weights = []
    biases = []
    epsW = []
    epsB = []

    print 'store the weight, bias and learning rate'
    for layer in self.layers:
      if isinstance(layer, WeightedLayer):
        weight = gpuarray.empty_like(layer.weight)
        gpu_copy_to(layer.weight, weight)
        weights.append(weight)
        epsW.append(layer.epsW)

        bias = gpuarray.empty_like(layer.bias)
        gpu_copy_to(layer.bias, bias)
        biases.append(bias)
        epsB.append(layer.epsB)

    print 'find the best learning rate'
    print 'the factor list is ', factors

    self.prepare_for_train(train_data, train_label)
    self.fprop(self.data, self.output)
    self.bprop(self.data, self.label, self.output)

    self.get_batch_information()
    self.update()

    self.train_batch(test_data, test_label, TEST)
    cost, correct, numCase = self.get_batch_information()
    best = (correct , 1.0)
    origin = (correct, 1.0)
    print 'The normal update produce the correct', correct, 'number of case is', numCase

    for factor in factors:
      print 'Try the factor', factor
      i = 0
      for layer in self.layers:
        if isinstance(layer, WeightedLayer):
          gpu_copy_to(weights[i], layer.weight)
          gpu_copy_to(biases[i], layer.bias)
          layer.epsW = epsW[i] * factor
          layer.epsB = epsB[i] * factor
          i += 1

      self.update()
      '''
      for layer in self.layers:
        if isinstance(layer, WeightedLayer):
          print 'epsW', layer.epsW, 'epsB', layer.epsB
          printMatrix(layer.weight, layer.name + 'weight')
          printMatrix(layer.bias, layer.name + 'bias')
      '''
      self.train_batch(test_data, test_label, TEST)
      cost, correct, numCase = self.get_batch_information()
      print 'Applying factor', factor, ', The correct is', correct, 'number of case is', numCase
      if correct > best[0]:
        best = (correct, factor)

    self.adjust_info.append((best[1], best[0], origin[0]))
    if best[0] / origin[0] < 1.025:
      best = origin
    factor = best[1]
    i = 0
    for layer in self.layers:
      if isinstance(layer, WeightedLayer):
        gpu_copy_to(weights[i], layer.weight)
        gpu_copy_to(biases[i], layer.bias)
        layer.epsW = epsW[i] * factor
        layer.epsB = epsB[i] * factor
        print 'Layer', layer.name
        print 'epsW is', layer.epsW, 'epsB is', layer.epsB
        i += 1
    # self.update()

  def get_report(self):
    return self.adjust_info
