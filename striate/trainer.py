from data import DataProvider, ImageNetDataProvider
from pycuda import gpuarray, driver
from striate import util
from striate.data import load
from striate.fastnet import FastNet, AdaptiveFastNet
from striate.layer import TRAIN, TEST
from striate.parser import Parser
from striate.scheduler import Scheduler
from striate.util import divup, timer
import cPickle
import numpy as np
import os
import pprint
import re
import sys
import time


class Trainer:
  CHECKPOINT_REGEX = None
  def __init__(self, test_id, data_dir, data_provider, checkpoint_dir, train_range, test_range, test_freq, save_freq, batch_size, num_epoch, image_size, 
               image_color, learning_rate, n_out, autoInit=True, initModel=None, adjust_freq=1, factor=1.0):
    self.test_id = test_id
    self.data_dir = data_dir
    self.data_provider = data_provider
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
    self.net = FastNet(self.learning_rate, self.image_shape, self.n_out, autoAdd=autoInit,
                       initModel=initModel)

    self.num_batch = self.curr_epoch = self.curr_batch = 0
    self.train_data = None
    self.test_data = None

    self.num_train_minibatch = 0
    self.num_test_minibatch = 0
    self.checkpoint_file = ''

  def init_data_provider(self):
    self.train_dp = DataProvider.get_by_name(self.data_provider)(self.data_dir, self.train_range)
    self.test_dp = DataProvider.get_by_name(self.data_provider)(self.data_dir, self.test_range)


  def get_next_minibatch(self, i, train=TRAIN):
    if train == TRAIN:
      data = self.train_data
    else:
      data = self.test_data

    batch_data = data.data
    batch_label = data.labels
    batch_size = self.batch_size

    mini_data = batch_data[:, i * batch_size: (i + 1) * batch_size]
    locked_data = driver.pagelocked_empty(mini_data.shape, mini_data.dtype, order='C',
                                          mem_flags=driver.host_alloc_flags.DEVICEMAP)
    locked_data[:] = mini_data

    input = gpuarray.to_gpu(locked_data)
    label = batch_label[i * batch_size : (i + 1) * batch_size]
    #label = gpuarray.to_gpu(np.require(batch_label[i * batch_size : (i + 1) * batch_size],  dtype =
    #  np.float, requirements = 'C'))

    return input, label


  def save_checkpoint(self):
    model = {}
    model['batchnum'] = self.train_dp.get_batch_num()
    model['epoch'] = self.num_epoch + 1
    model['layers'] = self.net.get_dumped_layers()

    #for param in model['layers']:
    #  print param.keys()
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
    self.test_data = self.test_dp.get_next_batch()

    self.num_test_minibatch = divup(self.test_data.data.shape[1], self.batch_size)
    for i in range(self.num_test_minibatch):
      input, label = self.get_next_minibatch(i, TEST)
      self.net.train_batch(input, label, TEST)
    cost , correct, numCase, = self.net.get_batch_information()
    self.test_outputs += [({'logprob': [cost, 1 - correct]}, numCase, time.time() - start)]
    print 'error: %f logreg: %f time: %f' % (1 - correct, cost, time.time() - start)
    self.print_net_summary()

  def print_net_summary(self):
    print '--------------------------------------------------------------'
    for s in self.net.get_summary():
      name = s[0]
      values = s[1]
      print "Layer '%s' weight: %e [%e]" % (name, values[0], values[1])
      print "Layer '%s' bias: %e [%e]" % (name, values[2], values[3])


  def check_continue_trainning(self):
    return self.curr_epoch <= self.num_epoch

  def check_test_data(self):
    return self.num_batch % self.test_freq == 0

  def check_save_checkpoint(self):
    return self.num_batch % self.save_freq == 0

  def check_adjust_lr(self):
    return self.num_batch % self.adjust_freq == 0

  def train(self):
    self.print_net_summary() 
    util.log('Starting training...')
    while self.check_continue_trainning():
      self.train_data = self.train_dp.get_next_batch()  # self.train_dp.wait()
      self.curr_epoch = self.train_data.epoch
      self.curr_batch = self.train_data.batchnum
       
      start = time.time()
      self.num_train_minibatch = divup(self.train_data.data.shape[1], self.batch_size)
      t = 0
      for i in range(self.num_train_minibatch):
        input, label = self.get_next_minibatch(i)
        stime = time.time()
        self.net.train_batch(input, label)
        t += time.time() - stime

      cost , correct, numCase = self.net.get_batch_information()
      self.train_outputs += [({'logprob': [cost, 1 - correct]}, numCase, time.time() - start)]
      print '%d.%d: error: %f logreg: %f time: %f' % (self.curr_epoch, self.curr_batch, 1 - correct, cost, time.time() - start)

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

      wait_time = time.time()

      #print 'waitting', time.time() - wait_time, 'secs to load'
      #print 'time to train a batch file is', time.time() - start

    if self.num_batch % self.save_freq != 0:
      print '---- save checkpoint ----'
      self.save_checkpoint()

    self.report()

  def report(self):
    print self.net.get_report()
    timer.report()



class AutoStopTrainer(Trainer):
  def __init__(self, test_id, data_dir, provider, checkpoint_dir, train_range, test_range, test_freq,
      save_freq, batch_size, num_epoch, image_size, image_color, learning_rate, n_out,
      auto_init=True, initModel=None, auto_stop_alg='smooth'):
    Trainer.__init__(self, test_id, data_dir, provider, checkpoint_dir, train_range, test_range, test_freq,
        save_freq, batch_size, num_epoch, image_size, image_color, learning_rate, n_out, auto_init,
        initModel=initModel)

    self.scheduler = Scheduler.makeScheduler(auto_stop_alg, self)


  def check_continue_trainning(self):
    return Trainer.check_continue_trainning(self) and self.scheduler.check_continue_trainning()

  def check_save_checkpoint(self):
    return Trainer.check_save_checkpoint(self) and self.scheduler.check_save_checkpoint()


class AdaptiveLearningRateTrainer(Trainer):
  def __init__(self, test_id, data_dir, provider, checkpoint_dir, train_range, test_range, test_freq,
      save_freq, batch_size, num_epoch, image_size, image_color, learning_rate, n_out, initModel=
      None, adjust_freq=10, factor=[1.0]):
    Trainer.__init__(self, test_id, data_dir, provider, checkpoint_dir, train_range, test_range, test_freq,
        save_freq, batch_size, num_epoch, image_size, image_color, learning_rate, n_out, adjust_freq
=adjust_freq, initModel=initModel, factor=factor, autoInit=False)
    self.train_data = self.train_dp.get_next_batch()
    batch = self.train_data.batchnum
    
    # if self.train_data.data.shape[1] > 1000:
    #  train_data = (self.train_data.data[:, :1000] , self.train_data.labels[:1000])
    # else:
    #  train_data = self.train_data

    train_data = self.get_next_minibatch(0)
    self.train_dp.del_batch(batch)

    _, batch, self.test_data = self.test_dp.get_next_batch()
    # if self.test_data['data'].shape[1] > 1000:
    #  test_data = (self.test_data['data'][:, :1000], self.train_data.labels[:1000])
    # else:
    #  test_data = self.test_data
    test_data = self.get_next_minibatch(0, TEST)
    self.test_dp.del_batch(batch)

    # test_data = self.get_next_minibatch(0)
    # test_data = train_data

    # train_data= self.train_data
    # test_data = self.test_data
    self.net = AdaptiveFastNet(self.learning_rate, self.image_shape, self.n_out, train_data,
        test_data, autoAdd=True)

  def report(self):
    lis = self.net.get_report()
    print 'Iteration:', self.adjust_freq
    print 'learningRare'
    for l in lis:
      print l[0]




class LayerwisedTrainer(AutoStopTrainer):
  def __init__(self, test_id, data_dir, provider, checkpoint_dir, train_range, test_range, test_freq,
      save_freq, batch_size, num_epoch, image_size, image_color, learning_rate, n_filters,
      size_filters, fc_nouts):
    AutoStopTrainer.__init__(self, test_id, data_dir,provider,  checkpoint_dir, train_range, test_range, test_freq,
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
        next_n_filter = [self.n_filters[i + 1]]
        next_size_filter = [self.size_filters[i + 1]]
        model = load(self.checkpoint_file)
        self.net = FastNet(self.learning_rate, self.image_shape, 0, initModel=model)
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
  def __init__(self, test_id, data_dir, provider, checkpoint_dir, train_range, test_range, test_freq,
      save_freq, batch_size, num_epoch, image_size, image_color, learning_rate, n_output, params):

    self.origin_test_range = test_range
    if len(test_range) != 1:
      test_range = [test_range[0]]
    AutoStopTrainer.__init__(self, test_id, data_dir, provider, checkpoint_dir, train_range, test_range, test_freq,
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

    self.conv_stack = FastNet.split_conv_to_stack(self.conv_params)
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
    # train conv stack layer by layer
    for i, stack in enumerate(self.conv_stack):
      if self.checkpoint_file != '':
        model = load(self.checkpoint_file)
        self.net = FastNet(self.learning_rate, self.image_shape, self.n_out, initModel=model)
        # delete softmax layer
        self.net.del_layer()
        self.net.del_layer()

        # for i in range(len(self.fc_params)):
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

    # train fc layer
    for i, stack in enumerate(self.fc_stack):
      model = load(self.checkpoint_file)
      self.net = FastNet(self.learning_rate, self.image_shape, self.n_out, initModel=model)
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
    self.net = FastNet(self.learning_rate, self.image_shape, self.n_out, initModel=model)
    self.test_range = self.origin_test_range
    self.init_data_provider()
    self.scheduler = Scheduler(self)
    self.num_epoch /= 2
    AutoStopTrainer.train(self)



if __name__ == '__main__':
  test_des_file = './testdes'
  factor = [1.5, 1.3, 1.2, 1.1, 1.05, 0.95, 0.9, 0.8, 0.75, 0.66]
  test_id = int(sys.argv[1])
  description = 'first try with momentum'

  # parameters for imagenet
  #data_dir = '/ssd/nn-data/imagenet/'
  #param_file = 'striate/imagenet.cfg'
  #data_provider = 'imagenet'
  #train_range = range(1, 1200)
  #test_range = range(1200, 1300)
  #save_freq = test_freq = 100
  #adjust_freq = 100
  #image_size = 224
  #n_out = 10

  data_dir = '/hdfs/cifar/data/cifar-10-python/'
  param_file = 'striate/cifar10.cfg'
  train_range = range(1, 41)
  test_range = range(41, 49)
  data_provider = 'cifar10'
  save_freq = test_freq = 20
  adjust_freq = 1
  image_size = 32
  n_out = 10

  checkpoint_dir = './striate/checkpoint/'

  batch_size = 128
  num_epoch = 10

  image_color = 3
  learning_rate = 1
  n_filters = [64, 64]
  size_filters = [5, 5]
  fc_nouts = [10]

  model = Parser(param_file).get_result()
  import pprint
  pprint.pprint(model)
  #model = util.load('./striate/stdmodel')

  trainer = Trainer(test_id, data_dir, data_provider, checkpoint_dir, train_range,
                    test_range, test_freq, save_freq, batch_size, num_epoch,
                    image_size, image_color, learning_rate, n_out, initModel=model)
  trainer.train()
