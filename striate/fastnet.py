from mpi4py import MPI
from pycuda import cumath, gpuarray, driver as cuda
from pycuda.gpuarray import GPUArray
from striate import util
from striate.cuda_kernel import gpu_copy_to, transpose
from striate.layer import ConvLayer, NeuronLayer, MaxPoolLayer, \
  ResponseNormLayer, FCLayer, SoftmaxLayer, TRAIN, WeightedLayer, TEST
from striate.parser import is_cudaconvnet_config, add_layers, FastNetBuilder, \
  CudaconvNetBuilder
from striate.util import timer
from virtual import virtual_array, Area
import numpy as np
import sys
  

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class FastNet(object):
  def __init__(self, learningRate, imgShape, numOutput, init_model = None):
    self.learningRate = learningRate
    self.numColor, self.imgSize, _ , self.batchSize = imgShape
    self.imgShapes = [imgShape]
    self.inputShapes = [(self.numColor * (self.imgSize ** 2), self.batchSize)]
    print self.inputShapes
    self.numOutput = numOutput
    self.layers = []
    self.outputs = []
    self.grads = []
    self.output = None
    self.save_layers = None
    self.save_output = []

    self.numCase = self.cost = self.correct = 0.0

    self.numConv = 0

    if init_model is None:
      util.log('initial model not provided, network doesn\'t have any layer')
      return

    if 'model_state' in init_model:
      # Loading from a checkpoint
      add_layers(FastNetBuilder(), self, init_model['model_state']['layers'])
    else:
      if is_cudaconvnet_config(init_model):
        # AlexK config file
        add_layers(CudaconvNetBuilder(), self, init_model)
      else:
        # FastNet config file
        add_layers(FastNetBuilder(), self, init_model)
      self.adjust_learning_rate(self.learningRate)

    util.log('Learning rates:')
    for l in self.layers:
      util.log('%s: %s %s', l.name, getattr(l, 'epsW', 0), getattr(l, 'epsB', 0))

  def save_layerouput(self, layers):
    self.save_layers = layers

  def append_layer(self, layer):
    self.layers.append(layer)
    if layer.type == 'conv':
      self.numConv += 1

    outputShape = layer.get_output_shape()
    row = outputShape[0] * outputShape[1] * outputShape[2]
    col = outputShape[3]
    self.inputShapes.append((row, col))
    self.imgShapes.append(outputShape)

    self.outputs.append(gpuarray.zeros((row, col), dtype=np.float32))
    self.grads.append(gpuarray.zeros(self.inputShapes[-2], dtype=np.float32))
    util.log('%s', self.outputs[-1].shape)
    util.log('%s', self.grads[-1].shape)
    print >> sys.stderr,  'append a', layer.type, 'layer', layer.name, 'to network'
    print >> sys.stderr,  'the output of the layer is', outputShape

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
      if l.disableBprop:
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
      if l.disableBprop or not isinstance(l, WeightedLayer):
        continue
      l.update()

  def adjust_learning_rate(self, factor=1.0):
    for layer in self.layers:
      if isinstance(layer, WeightedLayer):
        layer.scaleLearningRate(factor)


  def clear_weight_incr(self):
    for l in self.layers:
      if isinstance(l, WeightedLayer):
        l.clear_incr()

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

      self.imgShapes = [(self.numColor, self.imgSize, self.imgSize, self.batchSize)]
      self.inputShapes = [(self.numColor * (self.imgSize ** 2), self.batchSize)]
      for layer in self.layers:
        # layer.update_shape(...)
        outputShape = layer.get_output_shape()
        row = outputShape[0] * outputShape[1] * outputShape[2]
        col = outputShape[3]
        self.inputShapes.append((row, col))
        self.imgShapes.append(outputShape)

        self.outputs.append(gpuarray.zeros((row, col), dtype=np.float32))
        self.grads.append(gpuarray.zeros(self.inputShapes[-2], dtype=np.float32))

    if not isinstance(data, GPUArray):
      self.data = gpuarray.to_gpu(data).astype(np.float32)
    else:
      self.data = data

    if not isinstance(label, GPUArray):
      self.label = gpuarray.to_gpu(label).astype(np.float32)
    else:
      self.label = label

    self.label = self.label.reshape((label.size, 1))
    self.numCase += input.shape[1]
    outputShape = self.inputShapes[-1]
    if self.output is None or self.output.shape != outputShape:
      self.output = gpuarray.zeros(outputShape, dtype=np.float32)

  def train_batch(self, data, label, train=TRAIN):
    self.prepare_for_train(data, label)
    self.fprop(self.data, self.output, train)
    cost, correct = self.get_cost(self.label, self.output)
    self.cost += cost
    self.correct += correct

    if self.save_layers is not None:
      it = [(i, self.layers[i].name) for i in range(len(self.layers)) if self.layers[i].name in self.save_layers]
      outputs = [transpose(o).get() for o in self.outputs]
      label = self.label.get()
      self.save_output.extend([(label[i, 0], dict([(name, outputs[j][i,:]) for j, name in it])) for i in range(self.batchSize)])

    if train == TRAIN:
      self.bprop(self.data, self.label, self.output)
      self.update()

  def get_dumped_layers(self):
    layers = []
    for l in self.layers:
      layers.append(l.dump())

    return layers


  def get_save_output(self):
    if self.save_layers is None:
      assert False, 'Not specify any save layer name'
    else:
      tmp = self.save_output
      self.save_output = []
      return tmp

  def disable_bprop(self):
    for l in self.layers:
      l.disableBprop()

  def enable_bprop(self):
    for l in self.layers:
      l.enableBprop()

  def get_report(self):
    pass


  def get_summary(self):
    sum = []
    for l in self.layers:
      if isinstance(l, WeightedLayer):
        sum.append(l.get_summary())
    return sum


class AdaptiveFastNet(FastNet):
  def __init__(self, learningRate, imgShape, numOutput, train, test, init_model):
    FastNet.__init__(self, learningRate, imgShape, numOutput, init_model)
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

class DistFastNet(FastNet):
  def __init__(self, learning_rate, image_shape, num_ouput, init_model):
    if 'model_state' in init_model:
      model = init_model['model_state']['layers']
    else:
      model = init_model

    for layer in model:
      if layer.type == 'fc':
        layer['outputSize'] = layer['outputSize'] / comm.Get_size()
        size = comm.Get_size()
        if 'weight' in layer:
          layer['weight'] = np.vsplit(layer['weight'], size)[rank]
          layer['bias'] = np.vsplit(layer['bias'], size)[rank]
        if 'weightIncr' in layer:
          layer['weightIncr'] = np.vsplit(layer['weightIncr'], size)[rank]
          layer['biasIncr'] = np.vsplit(layers['biasIncr'], size)[rank]

    FastNet.__init__(self, learning_rate, image_shape, num_ouput, init_model)

  def _log(self, fmt, *args):
    util.log('%s :: %s', rank, fmt % args)


  def fprop(self, data, probs, train = TRAIN):
    fc = False
    input = data
    local_missing = True if not self.local_grads else False
    for i in range(len(self.layers)):
      l = self.layers[i]

      if l.type == 'fc' or l.type == 'softmax':
        fc = True
        input.gather()
        shape = input.get_global_shape()
        if len(shape) == 4:
          c, h, w, b = shape
          input.reshape(c * h * w, b)
        input.distribute(axis = -1)

      if fc != True:
        padding = l.get_cross_width()
        if padding != 0:
          d, area = input.get_cross(padding)
        else:
          d = input.get_local()
          area = input.get_local_area()
        d = d.reshape((d.shape[0] * d.shape[1] * d.shape[2], d.shape[3]))
      else:
        d = input.get_local()
        area = input.get_local_area()

      if local_missing:
        self.local_grads.append(gpuarray.zeros(d.shape, dtype = np.float32))
        self.grads.append(virtual_array(area = area))

      if l.type == 'softmax':
        output = self.output
      else:
        output = self.local_outputs[i]
        input = self.outputs[i]

      l.fprop(d, output, train)
      if not fc:
        shape = self.outputs[i].get_local_shape()
        self.outputs[i].store(output.reshape(shape), self.get_local_area())
      else:
        self.outputs[i].store(output, self.get_local_area())


  def bprop(self, data, label, prob, train = TRAIN):
    grad = label
    fc = True
    for i in range(1, len(self.layers) + 1):
      l = self.layers[-i]
      if l.disableBprop:
        return
      if i == len(self.layers):
        input = data
      else:
        input = self.outputs[-(i+1)]

      if l.type in ['pool', 'rnorm', 'cmrnorm', 'conv']:
        fc = False

      if fc != True:
        padding = l.get_cross_width()
        if padding != 0:
          d, area = input.get_cross(padding)
        else:
          d = input.get_local()
        local_input = d.reshape((d.shape[0] * d.shape[1] * d.shape[2], d.shape[3]))
      else:
        local_input = input.get_local()

      grad.reduce_add()
      if l.type == 'neuron' and fc:
        grad.distribute(axis = 0)
        grad = grad.get_local()
      elif not fc:
        grad.distribute_squre()
        grad = grad.get_local()
        grad = grad.reshape((grad.shape[0] * grad.shape[1] * grad.shape[2] , grad.shape[3]))


      if l.type == 'softmax':
        area = make_plain_area(self.output.shape)
        output = virtual_array(local = self.output, area = area)
        output.gather()
        output.distribute(axis = 0)
        local_output = output.get_local()
      else:
        local_output = self.local_outputs[-i];

      local_outGrad = self.local_grads[-i]
      l.bprop(grad, local_input, local_output, local_outGrad)

      grad = self.grads[-i]
      grad.store(local_outGrad, grad.get_local_area())


  def update(self):
    for layer in self.layers:
      if layer.disableBprop or not isinstance(layer, WeightedLayer):
        continue
      if layer.type == 'fc':
        layer.update()
      else:
        weightGrad, biasGrad = self.weightGrad, self.biasGrad
        area = make_plain_area(weightGrad)
        weightGrad = virtual_array(local = weightGrad, area = area)
        weightGrad.add_reduce()
        weightGrad = wegihGrad.get_local()

        area = make_plain_area(biasGrad)
        biasGrad = virtual_array(local = biasGrad, area = area)
        biasGrad.add_reduce()
        biasGrad = biasGrad.get_local()

        layer.update(weightGrad, biasGrad)

  def prepare_for_train(data, label):
    assert len(data.shape) == 4
    if data.shape[3] != self.batchSize:
      self.batchSize = data.shape[3]
      for l in self.layers:
        l.change_batch_size(self.batchSize)
      self.inputShapes = None
      self.imgShapes = None
      self.outputs = []
      self.grads = []
      self.local_outputs = []
      self.local_grads = []


      self.imgShapes = [(self.numColor, self.imgSize / 2, self.imgSize / 2, self.batchSize)]
      self.inputShapes = [(self.numColr * (self.imgSize ** 2) / 4, self.batchSize)]

      fc = False
      for layer in self.layers:
        outputShape = layer.get_output_shape()

        row = outputShape[0] * outputShape[1] * outputShape[2]
        col = outputShape[3]

        if layer.type == 'softmax':
          row *= comm.Get_size()
          outputShape = (outputShape[0] * comm.Get_size(), 1, 1, outputShape[3])

        self.inputShapes.append((row, col))
        self.imgShapes.append(outputShape)

        area = make_area(outputShape)
        self.outputs.append(virtual_array(rank, area = area))
        self.local_outputs.append(gpuarray.zeros((row, col), dtype =np.float32))

        inputShape = self.inputShapes[-2]
        #if layer.type == 'fc':
        #  inputShape = (inputShape[0] * comm.Get_size(), inputShape[1])
        #  self.local_grads.append(gpuarray.zeors(inputShape, dtype = np.float32))
        #  area = make_plain_area(inputShape)
        #else:
        #  self.local_grads.append(gpuarray.zeros(inputShape, dtype= np.float32))
        #  area = make_area(self.imgShapes[-2])
        #self.grads.append(virtual_array(rank, area = area))

      area = make_area((self.numColor, self.imgSize / 2, self.imgSize / 2, self.batchSize))
      self.data = virtual_array(rank, local = gpuarray.to_gpu(data.__getitem__(area.to_slice())),
          area = area)

      if not isinstance(label, GPUArray):
        self.label = gpuarray.to_gpu(label).astype(np.float32)
      else:
        self.label = label

      self.label = self.label.reshape((label.size, 1))
      self.numCase += data.shape[1]
      outputShape = self.inputShapes[-1]

      if self.output is None or self.output.shape != outputShape:
        self.output = gpuarray.zeros(outputShape, dtype = np.float32)



  def train_batch(self, data, label, train = TRAIN):
    self.prepare_for_train(data, label)
    self.fprop(self.data, self.output, train)
    cost, correct = self.get_cost(self.label, self.output)
    self.cost += cost
    self.correct += correct

    if train == TRAIN:
      self.bprop(self.data, self.label, self.output)
      self.update()

  def get_dumped_layers(self):

    def concatenate(dic, key):
      if key in dic:
        tmp = comm.gather(dic[key])
        comm.barrier()
        dic[key] = np.concatenate(tmp, axis = 0)

    layer_params = []
    for layer in self.layers:
      dic = layer.dump()
      if layer.type == 'fc':
        for key in ['weight', 'bias', 'weightIncr', 'biasIncr']:
          concatenate(dic, key)
      layer_params.append(dic)
      return layer_params




def make_area(shape):
  assert len(shape) == 4
  channel, height, width, batch_size = shape
  if height == 1:
    #input for fc layer
    size = comm.Get_size()
    row_from, row_to = rank * channel / size, (rank + 1) * channel / size - 1
    f = Point(row_from, 0)
    t = Point(row_to, batch_size - 1)
    area = Area(f, t)
  else:
    row_from, row_to, col_from, col_to = rank / 2 * height, rank / 2 * height + height -1, \
            rank % 2 * width, rank % 2 * width + width - 1
    f = Point(0, row_from, col_from, 0)
    t = Point(channel - 1, row_to, col_to, batch_size - 1)
    area = Area(f, t)
  return area

def make_plain_area(shape):
  assert len(shape) == 2
  height, width = shape
  return Area(Point(0, 0), Point(height-1, width-1))

