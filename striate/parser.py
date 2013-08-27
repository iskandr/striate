from striate.layer import ConvLayer, MaxPoolLayer, AvgPoolLayer, \
  CrossMapResponseNormLayer, SoftmaxLayer, NeuronLayer, ResponseNormLayer, FCLayer
from striate.util import isfloat
import numpy as np
import os

def parse_config_file(parsing_file):
  rst = []
  with open(parsing_file) as f:
    for line in f:
      line = line.strip()
      if line.startswith('#'):
        continue
      elif line.startswith('['):
        name = line[1:line.find(']')]
        rst.append({'name':name})
      elif len(line) == 0:
        continue
      else:
        key = line[0:line.find('=')]
        value = line[line.find('=')+1: len(line)]

        if value.isdigit():
          value = int(value)
        elif isfloat(value):
          value = float(value)

        rst[-1][key] = value
  return rst


class Builder(object):
  valid_dic = {}
  @staticmethod
  def set_val(ld, name, default = None):
    val  = ld.get(name, default)
    Builder.valid_dic[name] = 1
    return val

  @staticmethod
  def check_opts(ld):
    for k in Builder.valid_dic:
      if k not in ld:
        raise Exception, 'Unknown key %s' % k
    else:
      Builder.valid_dic = {}

  def make_layer(self, net, ld):
    ld['imgShape'] = net.imgShapes[-1]
    ld['inputShape'] = net.inputShapes[-1]

    if ld['type'] == 'conv': return self.conv_layer(ld)
    elif ld['type'] == 'pool': return self.pool_layer(ld)
    elif ld['type'] == 'neuron': return self.neuron_layer(ld)
    elif ld['type'] == 'fc': return self.fc_layer(ld)
    elif ld['type'] == 'softmax': return self.softmax_layer(ld)
    elif ld['type'] == 'rnorm': return self.rnorm_layer(ld)
    elif ld['type'] == 'cmrnorm': return self.crm_layer(ld)
    else:
      return None
      #raise Exception, 'Unknown layer %s' % ld['type']


class FastNetBuilder(Builder):
  def conv_layer(self, ld):
    numFilter = Builder.set_val(ld, 'numFilter')
    filterSize = Builder.set_val(ld, 'filterSize')
    numColor = Builder.set_val(ld, 'numColor')
    padding = Builder.set_val(ld, 'padding')
    stride = Builder.set_val(ld, 'stride')
    initW = Builder.set_val(ld, 'initW', 0.01)
    initB = Builder.set_val(ld, 'initB', 0.00)
    epsW = Builder.set_val(ld, 'epsW', 0.001)
    epsB = Builder.set_val(ld, 'epsB', 0.002)
    momW = Builder.set_val(ld, 'momW', 0.0)
    momB = Builder.set_val(ld, 'momB', 0.0)
    sharedBiases = Builder.set_val(ld, 'sharedBiases', default = 1)
    partialSum = Builder.set_val(ld, 'partialSum', default = 0)
    wc = Builder.set_val(ld, 'wc', 0.0)
    bias = Builder.set_val(ld, 'bias')
    weight = Builder.set_val(ld, 'weight')
    weightIncr = Builder.set_val(ld, 'weightIncr')
    biasIncr = Builder.set_val(ld, 'biasIncr')
    name = Builder.set_val(ld, 'name')
    img_shape = Builder.set_val(ld, 'imgShape')
    filter_shape = (numFilter, numColor, filterSize, filterSize)
    disableBprop = Builder.set_val(ld, 'disableBprop', default = False)
    cv = ConvLayer(name, filter_shape, img_shape, padding, stride, initW, initB,
        partialSum,sharedBiases, epsW, epsB, momW, momB, wc, bias, weight,
        weightIncr = weightIncr, biasIncr = biasIncr, disableBprop = disableBprop)
    return cv

  def pool_layer(self, ld):
    stride = Builder.set_val(ld, 'stride')
    start = Builder.set_val(ld, 'start')
    poolSize = Builder.set_val(ld, 'poolSize')
    img_shape = Builder.set_val(ld, 'imgShape')
    name = Builder.set_val(ld, 'name')
    pool = Builder.set_val(ld, 'pool', default = 'max')
    disableBprop = Builder.set_val(ld, 'disableBprop', default = False)
    if pool == 'max':
      return MaxPoolLayer(name, img_shape, poolSize, stride, start, disableBprop = disableBprop)
    elif pool == 'avg':
      return AvgPoolLayer(name, img_shape, poolSize, stride, start, disableBprop = disableBprop)

  def crm_layer(self, ld):
    name = Builder.set_val(ld, 'name')
    pow = Builder.set_val(ld, 'pow')
    size = Builder.set_val(ld, 'size')
    scale = Builder.set_val(ld, 'scale')
    image_shape = Builder.set_val(ld, 'imgShape')
    blocked = bool(Builder.set_val(ld, 'blocked', default = 0))
    disableBprop = Builder.set_val(ld, 'disableBprop', default = False)
    return CrossMapResponseNormLayer(name, image_shape, pow, size, scale, blocked, disableBprop =
        disableBprop)

  def softmax_layer(self, ld):
    name = Builder.set_val(ld, 'name')
    input_shape = Builder.set_val(ld, 'inputShape')
    disableBprop = Builder.set_val(ld, 'disableBprop', default = False)
    return SoftmaxLayer(name, input_shape, disableBprop = disableBprop)

  def neuron_layer(self, ld):
    name = Builder.set_val(ld, 'name')
    img_shape = Builder.set_val(ld, 'imgShape')
    disableBprop = Builder.set_val(ld, 'disableBprop', default = False)
    if ld['neuron'] == 'relu':
      e = Builder.set_val(ld, 'e')
      return NeuronLayer(name, img_shape, type='relu', e=e, disableBprop = disableBprop)

    if ld['neuron'] == 'tanh':
      a = Builder.set_val(ld, 'a')
      b = Builder.set_val(ld, 'b')
      return NeuronLayer(name, img_shape, type='tanh', a=a, b=b, disableBprop = disableBprop)

    assert False, 'No implementation for the neuron type' + ld['neuron']['type']

  def rnorm_layer(self, ld):
    name = Builder.set_val(ld, 'name')
    pow = Builder.set_val(ld,'pow')
    size = Builder.set_val(ld, 'size')
    scale = Builder.set_val(ld, 'scale')
    image_shape = Builder.set_val(ld, 'imgShape')
    disableBprop = Builder.set_val(ld, 'disableBprop', default = False)
    return ResponseNormLayer(name, image_shape, pow, size, scale, disableBprop = disableBprop)


  def fc_layer(self, ld):
    epsB = Builder.set_val(ld, 'epsB', 0.002)
    epsW = Builder.set_val(ld ,'epsW', 0.001)
    initB = Builder.set_val(ld, 'initB', 0.00)
    initW = Builder.set_val(ld, 'initW', 0.01)
    momB = Builder.set_val(ld, 'momB', 0.0)
    momW = Builder.set_val(ld, 'momW', 0.0)
    wc = Builder.set_val(ld, 'wc', 0.0)
    dropRate = Builder.set_val(ld, 'dropRate', 0.0)

    n_out = Builder.set_val(ld , 'outputSize')
    bias = Builder.set_val(ld, 'bias')
    weight = Builder.set_val(ld, 'weight')
    #if isinstance(weight, list):
    #  weight = np.concatenate(weight)

    weightIncr = Builder.set_val(ld, 'weightIncr')
    biasIncr = Builder.set_val(ld, 'biasIncr')
    name = Builder.set_val(ld, 'name')
    input_shape = Builder.set_val(ld, 'inputShape')
    disableBprop = Builder.set_val(ld, 'disableBprop', default = False)
    return FCLayer(name, input_shape, n_out, epsW, epsB, initW, initB, momW, momB, wc, dropRate,
        weight, bias, weightIncr = weightIncr, biasIncr = biasIncr, disableBprop = disableBprop)




class CudaconvNetBuilder(FastNetBuilder):
  def conv_layer(self, ld):
    numFilter = ld['filters']
    filterSize = ld['filterSize']
    numColor = ld['channels']
    padding = ld['padding']
    stride = ld['stride']
    initW = ld['initW']
    initB = ld.get('initB', 0.0)
    name = ld['name']
    epsW = ld['epsW']
    epsB = ld['epsB']

    momW = ld['momW']
    momB = ld['momB']

    wc = ld['wc']

    bias = ld.get('biases', None)
    weight = ld.get('weights', None)

    filter_shape = (numFilter, numColor, filterSize, filterSize)
    img_shape = ld['imgShape']
    return ConvLayer(name, filter_shape, img_shape, padding, stride, initW, initB, 0, 0, epsW, epsB, momW
        = momW, momB = momB, wc = wc, bias = bias, weight = weight)

  def pool_layer(self, ld):
    stride = ld['stride']
    start = ld['start']
    poolSize = ld['sizeX']
    img_shape = ld['imgShape']
    name = ld['name']
    pool = ld['pool']
    if pool == 'max':
      return MaxPoolLayer(name, img_shape, poolSize, stride, start)
    else:
      return AvgPoolLayer(name, img_shape, poolSize, stride, start)


  def neuron_layer(self, ld):
    if ld['neuron']['type'] == 'relu':
      img_shape = ld['imgShape']
      name = ld['name']
      #e = ld['neuron']['e']
      return NeuronLayer(name, img_shape, type='relu')
    if ld['neuron']['type'] == 'tanh':
      name = ld['name']
      img_shape = ld['imgShape']
      a = ld['neuron']['a']
      b = ld['neuron']['b']
      return NeuronLayer(name, img_shape, 'tanh', a=a, b=b)

    assert False, 'No implementation for the neuron type' + ld['neuron']['type']


  def fc_layer(self, ld):
    epsB = ld['epsB']
    epsW = ld['epsW']
    initB = ld.get('initB', 0.0)
    initW = ld['initW']
    momB = ld['momB']
    momW = ld['momW']

    wc = ld['wc']
    dropRate = ld.get('dropRate', 0.0)

    n_out = ld['outputs']
    bias = ld.get('biases', None)
    weight = ld.get('weights', None)

    if bias is not None:
      bias = bias.transpose()
      bias = np.require(bias, dtype = np.float32, requirements = 'C')
    if weight is not None:
      weight = weight.transpose()
      weight = np.require(weight, dtype = np.float32, requirements = 'C')

    name = ld['name']
    input_shape = ld['inputShape']
    return FCLayer(name, input_shape, n_out, epsW, epsB, initW, initB, momW = momW, momB = momB, wc
        = wc, dropRate = dropRate, weight = weight, bias = bias)

  def rnorm_layer(self, ld):
    name = Builder.set_val(ld, 'name')
    pow = Builder.set_val(ld,'pow')
    size = Builder.set_val(ld, 'size')
    scale = Builder.set_val(ld, 'scale')
    scale = scale * size ** 2
    image_shape = Builder.set_val(ld, 'imgShape')
    return ResponseNormLayer(name, image_shape, pow, size, scale)

  def crm_layer(self, ld):
    name = Builder.set_val(ld, 'name')
    pow = Builder.set_val(ld, 'pow')
    size = Builder.set_val(ld, 'size')
    scale = Builder.set_val(ld, 'scale')
    scale = scale * size
    image_shape = Builder.set_val(ld, 'imgShape')
    blocked = bool(Builder.set_val(ld, 'blocked', default = 0))
    return CrossMapResponseNormLayer(name, image_shape, pow, size, scale, blocked)
  
  
def add_layers(builder, net, model):
  for layer in model:
    l = builder.make_layer(net, layer)
    if l is not None:
      net.append_layer(l)

def is_cudaconvnet_config(model):
  for layer in model:
    if 'filters' in layer or 'channels' in layer:
      return True
  return False
