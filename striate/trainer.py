from fastnet import *
from pycuda import gpuarray, driver as cuda, autoinit
from pycuda.gpuarray import GPUArray
from data import DataProvider, ParallelDataProvider, ImageNetDataProvider
from options import *
from util import timer
import re
import time
from scheduler import *
import sys
import numpy as n
import argparse
from parser import *
import pprint

class Trainer:
  CHECKPOINT_REGEX = None
  def __init__(self, test_id, data_dir, checkpoint_dir, train_range, test_range, test_freq,
      save_freq, batch_size, num_epoch, image_size, image_color, learning_rate, n_out,
      autoInit=True, initModel = None, adjust_freq = 1, factor = 1.0):
    self.test_id = test_id
    self.data_dir = data_dir
    self.checkpoint_dir = checkpoint_dir
    self.train_range = train_range
    self.test_range = test_range
    self.test_freq = test_freq
    self.save_freq = save_freq
    self.batch_size = batch_size
    self.num_epoch = num_epoch
    self.image_size = image_size
    self.image_color = image_color
    self.learning_rate = learning_rate
    self.n_out = n_out
    self.factor = factor
    self.adjust_freq = adjust_freq
    self.regex = re.compile('^test%d-(\d+)\.(\d+)$' % self.test_id)

    self.init_data_provider()
    self.image_shape = (self.batch_size, self.image_color, self.image_size, self.image_size)
    self.train_outputs = []
    self.test_outputs = []
    self.net = FastNet(self.learning_rate, self.image_shape, self.n_out, autoAdd = autoInit,
        initModel = initModel)

    self.num_batch = self.curr_epoch = self.curr_batch = 0
    self.train_data = None
    self.test_data = None

    self.num_train_minibatch = 0
    self.num_test_minibatch = 0
    self.checkpoint_file = ''

  def init_data_provider(self):
    self.train_dp = DataProvider(self.data_dir, self.train_range)
    self.test_dp = DataProvider(self.data_dir, self.test_range)


  def get_next_minibatch(self, i, train = TRAIN):
    if train == TRAIN:
      num = self.num_train_minibatch
      data = self.train_data
    else:
      num = self.num_test_minibatch
      data = self.test_data


#    if not isinstance(data['data'], GPUArray):
#      data['data'] = gpuarray.to_gpu(data['data']).astype(np.float32)
#
#    if not isinstance(data['labels'], GPUArray):
#      data['labels'] = gpuarray.to_gpu(data['labels']).astype(np.float32)

    batch_data = data['data']
    batch_label = data['labels']
    batch_size = self.batch_size


    mh, mw = batch_data.shape

    if i == num -1:
      input = gpuarray.to_gpu(n.require((batch_data[:, i * batch_size: (i +1)* batch_size]), dtype= np.float32, requirements = 'C'))
      #input = gpuarray.empty((mh, mw - i*batch_size), dtype = np.float32)
      #gpu_partial_copy_to(batch_data, input, 0, mh, i * batch_size, (i + 1) * batch_size)
      label = batch_label[i* batch_size : mw]
    else:
      #input = gpuarray.empty((mh, batch_size), dtype = np.float32)
      #gpu_partial_copy_to(batch_data, input, 0, mh, i * batch_size, (i + 1) * batch_size)
      input = gpuarray.to_gpu(n.require((batch_data[:, i * batch_size: (i +1)* batch_size]), dtype= np.float32, requirements = 'C'))
      #a = batch_data[:, i * batch_size:(i+1)* batch_size]
      #input = cuda.mem_alloc(a.nbytes)
      #cuda.memcpy_htod(input, a)
      #input = gpuarray.GPUArray(a.shape, a.dtype, gpudata = input)
      label = batch_label[i * batch_size: (i + 1) * batch_size]

    return input, label


  def save_checkpoint(self):
    model = {}
    model['batchnum'] = self.train_dp.get_batch_num()
    model['epoch'] = self.num_epoch + 1
    model['layers'] = self.net.get_dumped_layers()
    model['train_outputs'] = self.train_outputs
    model['test_outputs'] = self.test_outputs

    dic = {'model_state': model, 'op':None}
    saved_filename = [f for f in os.listdir(self.checkpoint_dir) if self.regex.match(f)]
    for f in saved_filename:
      os.remove(os.path.join(self.checkpoint_dir, f))
    checkpoint_filename = "test%d-%d.%d" % (self.test_id, self.curr_epoch, self.curr_batch)
    checkpoint_file_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
    self.checkpoint_file = checkpoint_file_path
    print checkpoint_file_path
    with open(checkpoint_file_path, 'w') as f:
      cPickle.dump(dic, f)

  def get_test_error(self):
    start = time.time()
    _, _, self.test_data = self.test_dp.get_next_batch()

    self.num_test_minibatch = ceil(self.test_data['data'].shape[1], self.batch_size)
    for i in range(self.num_test_minibatch):
      input, label = self.get_next_minibatch(i, TEST)
      self.net.train_batch(input, label, TEST)
    cost , correct, numCase, = self.net.get_batch_information()
    self.test_outputs += [({'logprob': [cost, 1-correct]}, numCase, time.time() - start)]
    print 'error: %f logreg: %f time: %f' % (1-correct, cost, time.time() -
      start)

  def check_continue_trainning(self):
    return self.curr_epoch <= self.num_epoch

  def check_test_data(self):
    return self.num_batch % self.test_freq == 0

  def check_save_checkpoint(self):
    return self.num_batch % self.save_freq == 0

  def check_adjust_lr(self):
    return self.num_batch % self.adjust_freq == 0

  def train(self):
    self.curr_epoch, self.curr_batch, self.train_data = self.train_dp.get_next_batch()#self.train_dp.wait()
    while self.check_continue_trainning():
      start = time.time()
      self.num_train_minibatch = ceil(self.train_data['data'].shape[1], self.batch_size)
      t = 0
      for i in range(self.num_train_minibatch):
        input, label = self.get_next_minibatch(i)
        stime = time.time()
        self.net.train_batch(input, label)
        t += time.time() - stime

      cost , correct, numCase = self.net.get_batch_information()
      self.train_outputs += [({'logprob': [cost, 1-correct]}, numCase, time.time() - start)]
      print '%d.%d: error: %f logreg: %f time: %f' % (self.curr_epoch, self.curr_batch, 1-correct,cost, time.time() - start)

      self.num_batch += 1
      if self.check_test_data():
        print '---- test ----'
        self.get_test_error()
        print '------------'

      if self.factor != 1.0 and self.check_adjust_lr():
        print '---- adjust learning rate ----'
        self.net.adjust_learning_rate(self.factor)
        print '--------'

      if self.check_save_checkpoint():
        print '---- save checkpoint ----'
        self.save_checkpoint()
        print '------------'

      self.curr_epoch, self.curr_batch, self.train_data = self.train_dp.get_next_batch()##self.train_dp.wait()

    if self.num_batch % self.save_freq != 0:
      print '---- save checkpoint ----'
      self.save_checkpoint()

    self.report()

  def report(self):
    print self.net.get_report()
    timer.report()



class AutoStopTrainer(Trainer):
  def __init__(self, test_id, data_dir, checkpoint_dir, train_range, test_range, test_freq,
      save_freq, batch_size, num_epoch, image_size, image_color, learning_rate, n_out,
      auto_init=True, initModel = None,  auto_stop_alg = 'smooth'):
    Trainer.__init__(self, test_id, data_dir, checkpoint_dir, train_range, test_range, test_freq,
        save_freq, batch_size,num_epoch, image_size, image_color, learning_rate, n_out, auto_init,
        initModel = initModel)

    self.scheduler = Scheduler.makeScheduler(auto_stop_alg, self)


  def check_continue_trainning(self):
    return Trainer.check_continue_trainning(self) and self.scheduler.check_continue_trainning()

  def check_save_checkpoint(self):
    return Trainer.check_save_checkpoint(self) and self.scheduler.check_save_checkpoint()


class AdaptiveLearningRateTrainer(Trainer):
  def __init__(self, test_id, data_dir, checkpoint_dir, train_range, test_range, test_freq,
      save_freq, batch_size, num_epoch, image_size, image_color, learning_rate, n_out, initModel =
      None,  adjust_freq = 10, factor = [1.0]):
    Trainer.__init__(self, test_id, data_dir, checkpoint_dir, train_range, test_range, test_freq,
        save_freq, batch_size, num_epoch, image_size, image_color, learning_rate, n_out, adjust_freq
        = adjust_freq, initModel = initModel, factor = factor, autoInit = False)
    _, batch, self.train_data = self.train_dp.get_next_batch()
    #if self.train_data['data'].shape[1] > 1000:
    #  train_data = (self.train_data['data'][:, :1000] , self.train_data['labels'][:1000])
    #else:
    #  train_data = self.train_data

    train_data = self.get_next_minibatch(0)
    self.train_dp.del_batch(batch)

    _, batch, self.test_data = self.test_dp.get_next_batch()
    #if self.test_data['data'].shape[1] > 1000:
    #  test_data = (self.test_data['data'][:, :1000], self.train_data['labels'][:1000])
    #else:
    #  test_data = self.test_data
    test_data = self.get_next_minibatch(0, TEST)
    self.test_dp.del_batch(batch)

    #test_data = self.get_next_minibatch(0)
    #test_data = train_data

    #train_data= self.train_data
    #test_data = self.test_data
    self.net = AdaptiveFastNet(self.learning_rate, self.image_shape, self.n_out, train_data,
        test_data, autoAdd = True)

  def report(self):
    lis = self.net.get_report()
    print 'Iteration:', self.adjust_freq
    print 'learningRare'
    for l in lis:
      print l[0]




class LayerwisedTrainer(AutoStopTrainer):
  def __init__(self, test_id, data_dir, checkpoint_dir, train_range, test_range, test_freq,
      save_freq, batch_size, num_epoch, image_size, image_color, learning_rate, n_filters,
      size_filters, fc_nouts):
    AutoStopTrainer.__init__(self, test_id, data_dir, checkpoint_dir, train_range, test_range, test_freq,
        save_freq, batch_size, num_epoch, image_size, image_color, learning_rate, 0, False)
    if len(n_filters) == 1:
      self.layerwised = False
    else:
      self.layerwised = True


    self.n_filters = n_filters
    self.size_filters = size_filters
    self.fc_nouts = fc_nouts

    init_n_filter = [self.n_filters[0]]
    init_size_filter = [self.size_filters[0]]


    self.net.add_parameterized_layers(init_n_filter, init_size_filter, self.fc_nouts)

  def train(self):
    AutoStopTrainer.train(self)

    if self.layerwised:
      for i in range(len(self.n_filters) - 1):
        next_n_filter = [self.n_filters[i +1]]
        next_size_filter = [self.size_filters[i+1]]
        model = load(self.checkpoint_file)
        self.net = FastNet(self.learning_rate, self.image_shape, 0, initModel = model)
        self.net.del_layer()
        self.net.del_layer()
        self.net.disable_bprop()

        self.net.add_parameterized_layers(next_n_filter, next_size_filter, self.fc_nouts)
        self.init_data_provider()
        self.scheduler = Scheduler(self)
        self.test_outputs = []
        self.train_outputs = []
        AutoStopTrainer.train(self)


class ImageNetLayerwisedTrainer(AutoStopTrainer):
  def __init__(self, test_id, data_dir, checkpoint_dir, train_range, test_range, test_freq,
      save_freq, batch_size, num_epoch, image_size, image_color, learning_rate, n_output, params):

    self.origin_test_range = test_range
    if len(test_range) != 1:
      test_range = [test_range[0]]
    AutoStopTrainer.__init__(self, test_id, data_dir, checkpoint_dir, train_range, test_range, test_freq,
        save_freq, batch_size, num_epoch, image_size, image_color, learning_rate, n_output, False)

    self.conv_params = []
    self.fc_params = []
    self.softmax_param = None

    self.params = params

    conv = True
    for ld in self.params:
      if ld['type'] in ['conv', 'rnorm', 'pool', 'neuron'] and conv:
        self.conv_params.append(ld)
      elif ld['type'] == 'fc' or (not conv and ld['type'] == 'neuron'):
        self.fc_params.append(ld)
        conv = False
      else:
        self.softmax_param = ld

    self.conv_stack =  FastNet.split_conv_to_stack(self.conv_params)
    self.fc_stack = FastNet.split_fc_to_stack(self.fc_params)

    pprint.pprint(self.conv_stack)
    pprint.pprint(self.fc_stack)

    self.fakefc_param = self.fc_stack[-1][0]

  def report(self):
    pass

  def init_data_provider(self):
    self.train_dp = ImageNetDataProvider(self.data_dir, self.train_range)
    self.test_dp = ImageNetDataProvider(self.data_dir, self.test_range)

  def train(self):
    #train conv stack layer by layer
    for i, stack in enumerate(self.conv_stack):
      if self.checkpoint_file != '':
        model = load(self.checkpoint_file)
        self.net = FastNet(self.learning_rate, self.image_shape, self.n_out, initModel = model)
        #delete softmax layer
        self.net.del_layer()
        self.net.del_layer()

        #for i in range(len(self.fc_params)):
        #  self.net.del_layer()

        self.net.disable_bprop()

      layerParam = stack + [self.fakefc_param, self.softmax_param]
      self.net.append_layers_from_dict(layerParam)

      self.init_data_provider()
      self.scheduler.reset()
      self.scheduler.set_level(i)
      self.test_outputs = []
      self.train_output = []
      AutoStopTrainer.train(self)

    #train fc layer
    for i, stack in enumerate(self.fc_stack):
      model = load(self.checkpoint_file)
      self.net = FastNet(self.learning_rate, self.image_shape, self.n_out, initModel = model)
      self.net.del_layer()
      self.net.del_layer()

      self.net.disable_bprop()

      if i == len(self.fc_stack) - 1:
        layerParam = stack + [self.softmax_param]
      else:
        layerParam = stack + [self.fakefc_param, self.softmax_param]
      self.net.append_layers_from_dict(layerParam)

      self.init_data_provider()
      self.scheduler.reset()
      self.scheduler.set_level(i)
      self.test_outputs = []
      self.train_output = []
      AutoStopTrainer.train(self)

    model = load(self.checkpoint_file)
    self.test_id += 1
    self.net = FastNet(self.learning_rate, self.image_shape, self.n_out, initModel = model)
    self.test_range = self.origin_test_range
    self.init_data_provider()
    self.scheduler = Scheduler(self)
    self.num_epoch /= 2
    AutoStopTrainer.train(self)



if __name__ == '__main__':
  test_des_file = './testdes'
  factor = [1.5, 1.3, 1.2, 1.1, 1.05, 0.95, 0.9, 0.8, 0.75,  0.66]
  test_id = 30
  description = 'compare to test 27, another try'

#  lines = [line for line in open(test_des_file)]
#  test_des = {int(line.split()[0]):line.split()[1] for line in lines }
#
#  if test_id in  test_des.keys():
#    print test_id, 'is already in test des file and the purpose is', test_des[test_id]
#    sys.exit(1)
#  else:
#    print 'test id is', test_id, 'for', description
#    line= '%d %s\n' % (test_id, description)
#    with open(test_des_file, 'a') as f:
#      f.write(line)
  #parameters for imagenet
  data_dir = '/hdfs/imagenet/batches/imagesize-256/'
  param_file = './imagenet.cfg'
  train_range = range(1, 401)
  test_range = range(401, 650)
  save_freq = test_freq = 10
  adjust_freq = 10
  image_size = 224
  n_out = 1000

  #parameter for cifar10
  #data_dir = '/hdfs/cifar/data/cifar-10-python/'
  #param_file = './cifar10.cfg'
  #train_range = range(1, 41)
  #test_range = range(41, 49)
  #save_freq = test_freq = 10
  #adjust_freq = 40
  #image_size = 32
  #n_out = 10

  checkpoint_dir = './checkpoint/'

  batch_size = 128
  num_epoch = 30

  image_color = 3
  learning_rate = 1.28
  n_filters = [64, 64]
  size_filters = [5, 5]
  fc_nouts = [10]

  #model = load('./checkpoint/test29-17.20')
  #trainer = Trainer(test_id, data_dir, checkpoint_dir, train_range, test_range, test_freq, save_freq,
  #     batch_size, num_epoch, image_size, image_color, learning_rate, 10, initModel = model)
  #trainer = LayerwisedTrainer(test_id, data_dir, checkpoint_dir, train_range, test_range, test_freq,
  #    save_freq, batch_size, num_epoch, image_size, image_color, learning_rate, n_filters,
  #    size_filters, fc_nouts)
  #trainer = AutoStopTrainer(test_id, data_dir, checkpoint_dir, train_range, test_range, test_freq,
  #    save_freq, batch_size, num_epoch, image_size, image_color, learning_rate, 10)
  #trainer = AdaptiveLearningRateTrainer(test_id, data_dir, checkpoint_dir, train_range, test_range, test_freq,
  #    save_freq, batch_size, num_epoch, image_size, image_color, learning_rate, 10, adjust_freq, factor)
  params = Parser(param_file).get_result()
  pprint.pprint(params)
  trainer = ImageNetLayerwisedTrainer(test_id, data_dir, checkpoint_dir, train_range,
      test_range, test_freq, save_freq, batch_size, num_epoch,
      image_size, image_color, learning_rate, n_out, params)
  trainer.train()
  '''

  parser = argparse.ArgumentParser()
  parser.add_argument('--des_file', help = 'The description file', default='./testdes')
  parser.add_argument('--test_id', help = 'The test id, Need to be exclusive', type = int)
  description = ''
  parser.add_argument('--data_dir', help = 'The data directory')
  parser.add_argument('--checkpoint', help = 'The checkpoint file directory',default='./checkpoint')

  parser.add_argument('--train_range', help = 'The range of train file')
  parser.add_argument('--test_range', help = 'The range of test file')
  parser.add_argument('--data', help = 'The type of data', choices=['cifar', 'imagenet'])
  parser.add_argument('--save_freq', help = 'The frequency to save checkpoint', default= 10, type = int)
  parser.add_argument('--test_freq', help = 'The frequency to test model', default = 10, type = int)
  parser.add_argument('--num_epoch', help = 'The number of epoch', default =30, type = int)
  parser.add_argument('--adjust_freq', help = 'The frequency to adjust learning rate', default = 40, type = int)
  parser.add_argument('--batch_size', help = 'Batch size', default = 128, type = int)
  parser.add_argument('--n_output', help = 'The number of output', type = int, default = 10)
  parser.add_argument('--param_file', help = 'The layer parameter file', default = None)
  parser.add_argument('--type', help = 'Model type', default= 'normal', choices = ['normal',
    'adpative', 'layerwised', 'autostop', 'imagenet'])

  factor = [1.5, 1.3, 1.2, 1.1, 1.05, 0.95, 0.9, 0.8, 0.75,  0.66]
  args = parser.parse_args()

  image_color = 3
  if args.data == 'cifar':
    image_size = 32
  else args.data == 'imagenet':
    image_size = 224

  args.train_range = string_to_int_list(args.train_range)
  args.test_range = string_to_int_list(args.test_range)

  n_filters = [64, 64]
  size_filters = [5, 5]
  fc_nouts = [10]

  if args.type == 'normal':
    trainer = Trainer(args.test_id, args.data_dir, args.checkpoint_dir, args.train_range,
        args.test_range, args.test_freq, args.save_freq, args.batch_size, args.num_epoch,
        image_size, image_color, args.learning_rate, args.n_output)
  elif args.type == 'adpative':
    trainer = AdaptiveLearningRateTrainer(args.test_id, args.data_dir, args.checkpoint_dir,
        args.train_range, args.test_range, args.test_freq, args.save_freq, args.batch_size,
        args.num_epoch, image_size, image_color, args.learning_rate, args.n_output, args.adjust_freq, factor)
  elif args.type == 'autostop':
    trainer = AutoStopTrainer(args.test_id, args.data_dir, args.checkpoint_dir, args.train_range,
        args.test_range, args.test_freq, args.save_freq, args.batch_size, args.num_epoch,
        image_size, image_color, args.learning_rate, args.n_output)
  elif args.type == 'layerwised':
    trainer = LayerwisedTrainer(args.test_id, args.data_dir, args.checkpoint_dir, args.train_range,
        args.test_range, args.test_freq, args.save_freq, args.batch_size, args.num_epoch,
        image_size, image_color, args.learning_rate, n_filters, size_filters, fc_nouts)
  else:
    params = parser(args.param_file).get_result()
    trainer = ImageNetLayerwisedTrainer(args.test_id, args.data_dir, args.checkpoint_dir, args.train_range,
        args.test_range, args.test_freq, args.save_freq, args.batch_size, args.num_epoch,
        image_size, image_color, args.learning_rate, args.n_output, params)

  trainer.train()
  '''
