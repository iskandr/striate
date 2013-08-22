from data import DataProvider, ImageNetDataProvider
from pycuda import gpuarray, driver
from striate import util, layer
from striate.fastnet import FastNet, AdaptiveFastNet
from striate.layer import TRAIN, TEST
from striate.parser import Parser
from striate.scheduler import Scheduler
from striate.util import divup, timer, load
import argparse
import cPickle
import glob
import numpy as np
import os
import pprint
import re
import sys
import time

class DataDumper(object):
  def __init__(self, target_path):
    self.target_path = target_path
    self.data = []
    self.sz = 0
    self.count = 0
    self.max_mem_size = 500e6
    
  def add(self, data):
    for k, v in data.iteritems():
      self.sz += np.prod(v.shape)
    self.data.append(data)
    
    if self.sz > self.max_mem_size:
      self.flush()
      
  def flush(self):
    if self.sz == 0:
      return

    out = {}
    for k in self.data[0].keys():
      items = [d[k] for d in self.data]
      out[k] = np.concatenate(items, axis=0)
    
    with open('%s.%d' % (self.target_path, self.count), 'w') as f:
      cPickle.dump(out, f, -1)

    util.log('Wrote layer dump.')
    self.data = []    
    self.sz = 0
    self.count += 1
    


class Trainer:
  CHECKPOINT_REGEX = None
  def __init__(self, test_id, data_dir, data_provider, checkpoint_dir, train_range, test_range, test_freq, save_freq, batch_size, num_epoch, image_size,
               image_color, learning_rate, init_model=None, adjust_freq=1, factor=1.0):
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
    # doesn't matter anymore
    self.n_out = 10
    self.factor = factor
    self.adjust_freq = adjust_freq
    self.regex = re.compile('^test%d-(\d+)\.(\d+)$' % self.test_id)

    self.init_data_provider()
    self.image_shape = (self.image_color, self.image_size, self.image_size, self.batch_size)

    if init_model is not None and 'model_state' in init_model:
      self.train_outputs = init_model['model_state']['train_outputs']
      self.test_outputs = init_model['model_state']['test_outputs']
    else:
      self.train_outputs = []
      self.test_outputs = []

    self.curr_minibatch = self.num_batch = self.curr_epoch = self.curr_batch = 0
    self.net = FastNet(self.learning_rate, self.image_shape, self.n_out, init_model=init_model)

    self.train_data = None
    self.test_data = None

    self.num_train_minibatch = 0
    self.num_test_minibatch = 0
    self.checkpoint_file = ''
    
    self.train_dumper = None #DataDumper('/scratch1/imagenet-pickle/train-data.pickle')
    self.test_dumper = None #DataDumper('/scratch1/imagenet-pickle/test-data.pickle')
    self.input = None


  def init_data_provider(self):
    dp = DataProvider.get_by_name(self.data_provider)
    self.train_dp = dp(self.data_dir, self.train_range)
    self.test_dp = dp(self.data_dir, self.test_range)


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
                                          mem_flags=driver.host_alloc_flags.PORTABLE)
    locked_data[:] = mini_data

    if self.input is not None and locked_data.shape == self.input.shape:
      self.input.set(locked_data)
    else:
      self.input = gpuarray.to_gpu(locked_data)
    
    label = batch_label[i * batch_size : (i + 1) * batch_size]
    #label = gpuarray.to_gpu(label)

    #label = gpuarray.to_gpu(np.require(batch_label[i * batch_size : (i + 1) * batch_size],  dtype =
    #  np.float, requirements = 'C'))

    return self.input, label


  def save_checkpoint(self):
    model = {}
    model['batchnum'] = self.train_dp.get_batch_num()
    model['epoch'] = self.num_epoch + 1
    model['layers'] = self.net.get_dumped_layers()

    model['train_outputs'] = self.train_outputs
    model['test_outputs'] = self.test_outputs

    dic = {'model_state': model, 'op':None}
    self.print_net_summary()
    
    if not os.path.exists(self.checkpoint_dir):
      os.system('mkdir -p \'%s\'' % self.checkpoint_dir)
    
    saved_filename = [f for f in os.listdir(self.checkpoint_dir) if self.regex.match(f)]
    for f in saved_filename:
      os.remove(os.path.join(self.checkpoint_dir, f))
    checkpoint_filename = "test%d-%d.%d" % (self.test_id, self.curr_epoch, self.curr_batch)
    checkpoint_file_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
    self.checkpoint_file = checkpoint_file_path
    print >> sys.stderr,  checkpoint_file_path
    with open(checkpoint_file_path, 'w') as f:
      cPickle.dump(dic, f, protocol=-1)
    util.log('save file finished')

  def get_test_error(self):
    start = time.time()
    self.test_data = self.test_dp.get_next_batch()

    self.num_test_minibatch = divup(self.test_data.data.shape[1], self.batch_size)
    for i in range(self.num_test_minibatch):
      input, label = self.get_next_minibatch(i, TEST)
      self.net.train_batch(input, label, TEST)
      self._capture_test_data()
    
    cost , correct, numCase, = self.net.get_batch_information()
    self.test_outputs += [({'logprob': [cost, 1 - correct]}, numCase, time.time() - start)]
    print >> sys.stderr,  '[%d] error: %f logreg: %f time: %f' % (self.test_data.batchnum, 1 - correct, cost, time.time() - start)

  def print_net_summary(self):
    print >> sys.stderr,  '--------------------------------------------------------------'
    for s in self.net.get_summary():
      name = s[0]
      values = s[1]
      print >> sys.stderr,  "Layer '%s' weight: %e [%e]" % (name, values[0], values[1])
      print >> sys.stderr,  "Layer '%s' bias: %e [%e]" % (name, values[2], values[3])


  def should_continue_training(self):
    return self.curr_epoch <= self.num_epoch

  def check_test_data(self):
    return self.num_batch % self.test_freq == 0

  def check_save_checkpoint(self):
    return self.num_batch % self.save_freq == 0

  def check_adjust_lr(self):
    return self.num_batch % self.adjust_freq == 0
  
  def _finished_training(self):
    if self.train_dumper is not None:
      self.train_dumper.flush()
    
    if self.test_dumper is not None:
      self.test_dumper.flush()
      
  def _capture_training_data(self):
    if not self.train_dumper:
      return

    self.train_dumper.add({'labels' : self.net.label.get(),
                           'fc' : self.net.outputs[-3].get().transpose() })
    
  def _capture_test_data(self):
    if not self.test_dumper:
      return
    self.test_dumper.add({'labels' : self.net.label.get(),
                           'fc' : self.net.outputs[-3].get().transpose() })

  def train(self):
    self.print_net_summary()
    util.log('Starting training...')
    while self.should_continue_training():
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
        self._capture_training_data()
        t += time.time() - stime
        self.curr_minibatch += 1

      cost , correct, numCase = self.net.get_batch_information()
      self.train_outputs += [({'logprob': [cost, 1 - correct]}, numCase, time.time() - start)]
      print >> sys.stderr,  '%d.%d: error: %f logreg: %f time: %f' % (self.curr_epoch, self.curr_batch, 1 - correct, cost, time.time() - start)

      self.num_batch += 1
      if self.check_test_data():
        print >> sys.stderr,  '---- test ----'
        self.get_test_error()
        print >> sys.stderr,  '------------'

      if self.factor != 1.0 and self.check_adjust_lr():
        print >> sys.stderr,  '---- adjust learning rate ----'
        self.net.adjust_learning_rate(self.factor)
        print >> sys.stderr,  '--------'

      if self.check_save_checkpoint():
        print >> sys.stderr,  '---- save checkpoint ----'
        self.save_checkpoint()
        print >> sys.stderr,  '------------'

      wait_time = time.time()

      #print 'waitting', time.time() - wait_time, 'secs to load'
      #print 'time to train a batch file is', time.time() - start)


    self.get_test_error()
    self.save_checkpoint()
    self.report()
    self._finished_training()

  def predict(self, save_layers = None, filename = None):
    self.net.save_layerouput(save_layers)
    self.print_net_summary()
    util.log('Starting predict...')
    save_output = []
    while self.curr_epoch < 2:
      start = time.time()
      self.test_data = self.test_dp.get_next_batch()
      self.curr_epoch = self.test_data.epoch
      self.curr_batch = self.test_data.batchnum

      self.num_test_minibatch = divup(self.test_data.data.shape[1], self.batch_size)
      for i in range(self.num_test_minibatch):
        input, label = self.get_next_minibatch(i, TEST)
        self.net.train_batch(input, label, TEST)
      cost , correct, numCase = self.net.get_batch_information()
      print >> sys.stderr,  '%d.%d: error: %f logreg: %f time: %f' % (self.curr_epoch, self.curr_batch, 1 - correct, cost, time.time() - start)
      if save_layers is not None:
        save_output.extend(self.net.get_save_output())

    if save_layers is not None:
      if filename is not None:
        with open(filename, 'w') as f:
          cPickle.dump(save_output, f, protocol = -1)
        util.log('save layer output finished')


  def report(self):
    rep = self.net.get_report()
    if rep is not None:
      print rep
    #timer.report()
  
  @staticmethod
  def get_trainer_by_name(name, param_dict, rest_args):
    if name == 'normal':
      param_dict['num_epoch'] = args.num_epoch
      return Trainer(**param_dict)

    if name == 'layerwise':
      param_dict['num_epoch'] = args.num_epoch
      return ImageNetLayerwisedTrainer(**param_dict)
    
    num_minibatch = util.string_to_int_list(args.num_minibatch)
    if len(num_minibatch) == 1:
      param_dict['num_minibatch'] = num_minibatch[0]
    else:
      param_dict['num_minibatch'] = num_minibatch

    if name == 'minibatch':
      return MiniBatchTrainer(**param_dict)

    if name == 'catewise':
      param_dict['num_caterange_list'] = util.string_to_int_list(args.num_caterange_list)
      return ImageNetCatewisedTrainer(**param_dict)

    if name == 'categroup':
      param_dict['num_group_list'] = util.string_to_int_list(args.num_group_list)
      return ImageNetCateGroupTrainer(**param_dict)

    raise Exception, 'No trainer found for name: %s' % name


class MiniBatchTrainer(Trainer):
  def __init__(self, test_id, data_dir, data_provider, checkpoint_dir, train_range, test_range,
      test_freq, save_freq, batch_size, num_minibatch, image_size, image_color, learning_rate,
      init_model=None, adjust_freq=1, factor=1.0):

    self.num_minibatch = num_minibatch
    fake_num_epoch = 100
    Trainer.__init__(self, test_id, data_dir, data_provider, checkpoint_dir, train_range,
        test_range, test_freq, save_freq, batch_size, fake_num_epoch, image_size, image_color,
        learning_rate,  init_model = init_model, adjust_freq = adjust_freq, factor = factor)

  def should_continue_training(self):
    return self.curr_minibatch <= self.num_minibatch


class AutoStopTrainer(Trainer):
  def __init__(self, test_id, data_dir, provider, checkpoint_dir, train_range, test_range, test_freq,
      save_freq, batch_size, num_epoch, image_size, image_color, learning_rate,
      init_model=None, auto_stop_alg='smooth'):
    Trainer.__init__(self, test_id, data_dir, provider, checkpoint_dir, train_range, test_range, test_freq,
        save_freq, batch_size, num_epoch, image_size, image_color, learning_rate, init_model=init_model)

    self.scheduler = Scheduler.makeScheduler(auto_stop_alg, self)


  def should_continue_training(self):
    return Trainer.should_continue_training(self) and self.scheduler.should_continue_training()

  def check_save_checkpoint(self):
    return Trainer.check_save_checkpoint(self) and self.scheduler.check_save_checkpoint()


class AdaptiveLearningRateTrainer(Trainer):
  def __init__(self, test_id, data_dir, provider, checkpoint_dir, train_range, test_range, test_freq,
      save_freq, batch_size, num_epoch, image_size, image_color, learning_rate, init_model= None, adjust_freq=10, factor=[1.0]):
    Trainer.__init__(self, test_id, data_dir, provider, checkpoint_dir, train_range, test_range, test_freq,
        save_freq, batch_size, num_epoch, image_size, image_color, learning_rate,  adjust_freq = adjust_freq,
        init_model = None, factor=factor)
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
        test_data, init_model = init_model)

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
        
  def add_parameterized_layers(self, net, n_filters=None, size_filters=None, fc_nout=[10]):
    for i in range(len(n_filters)):
      prev = n_filters[i - 1] if i > 0 else net.imgShapes[-1][1]
      filter_shape = (n_filters[i], prev, size_filters[i], size_filters[i])
      conv = layer.ConvLayer('conv' + str(net.numConv), filter_shape, net.imgShapes[-1])
      net.append_layer(conv)

      neuron = layer.NeuronLayer('neuron' + str(net.numConv), net.imgShapes[-1], type='tanh')
      net.append_layer(neuron)

      pool = layer.MaxPoolLayer('pool' + str(net.numConv), net.imgShapes[-1])
      net.append_layer(pool)

      rnorm = layer.ResponseNormLayer('rnorm' + str(net.numConv), net.imgShapes[-1])
      net.append_layer(rnorm)

    for i in range(len(fc_nout)):
      fc = layer.FCLayer('fc' + str(i + 1), net.inputShapes[-1], fc_nout[-1])
      net.append_layer(fc)

    net.append_layer(layer.SoftmaxLayer('softmax', net.inputShapes[-1]))


class ImageNetLayerwisedTrainer(Trainer):
  def __init__(self, test_id, data_dir, data_provider, checkpoint_dir, train_range, test_range, test_freq,
      save_freq, batch_size, num_epoch, image_size, image_color, learning_rate, init_model = None,
      factor = 1.0, adjust_freq = 1):
    
    self.curr_model = []
    self.complete_model = init_model
    self.fc_params = []
    self.final_num_epoch = num_epoch
    conv = True
    for ld in init_model:
      if ld['type'] in ['conv', 'rnorm', 'pool', 'neuron'] and conv:
        self.curr_model.append(ld)
      elif ld['type'] == 'fc' or (not conv and ld['type'] == 'neuron'):
        self.fc_params.append(ld)
        conv = False
      else:
        self.softmax_param = ld

    self.fc_stack = FastNet.split_fc_to_stack(self.fc_params)

  
    self.curr_model.append(self.fc_stack[-1][0])
    self.curr_model.append(self.softmax_param)
    del self.fc_stack[-1]
    pprint.pprint(self.fc_stack)
    
    Trainer.__init__(self, test_id, data_dir, data_provider, checkpoint_dir, train_range, test_range, test_freq,
        save_freq, batch_size, 1, image_size, image_color, learning_rate, init_model = self.curr_model)


  def report(self):
    pass

  def train(self):
    # train fc layer
    Trainer.train(self)
    for i, stack in enumerate(self.fc_stack):
      pprint.pprint(stack)
      self.curr_model = load(self.checkpoint_file)
      self.num_batch = self.curr_epoch = self.curr_batch = 0
      self.curr_minibatch = 0
      
      stack[0]['epsW'] *= self.learning_rate
      stack[0]['epsB'] *= self.learning_rate
      self.curr_model['model_state']['layers'].insert(-2, stack[0])
      self.curr_model['model_state']['layers'].insert(-2, stack[1])
      
      if i == len(self.fc_stack) - 1:
        self.num_epoch = self.final_num_epoch
      else:
        l = self.curr_model['model_state']['layers'][-2]
        assert l['type'] == 'fc'

        l['weight'] = None
        l['bias'] = None
        l['weightIncr'] = None
        l['biasIncr'] = None

      self.init_data_provider()
      self.net = FastNet(self.learning_rate, self.image_shape, self.n_out, init_model = self.curr_model)
      Trainer.train(self)



class ImageNetCatewisedTrainer(MiniBatchTrainer):
  def __init__(self, test_id, data_dir, data_provider, checkpoint_dir, train_range, test_range,
      test_freq, save_freq, batch_size, num_minibatch, image_size, image_color, learning_rate,
      init_model, num_caterange_list, adjust_freq = 100, factor = 1.0):
    # no meaning
    assert len(num_caterange_list) == len(num_minibatch) and num_caterange_list[-1] == 1000

    self.init_output = num_caterange_list[0]
    self.range_list = num_caterange_list[1:]
    self.train_minibatch_list  = num_minibatch[1:]

    fc = init_model[-2]
    fc['outputSize'] = self.init_output

    self.learning_rate = learning_rate[0]
    self.learning_rate_list = learning_rate[1:]

    MiniBatchTrainer.__init__(self, test_id, data_dir, data_provider, checkpoint_dir, train_range,
        test_range, test_freq, save_freq, batch_size, num_minibatch[0], image_size, image_color,
        self.learning_rate,  init_model = init_model)

  def init_data_provider(self):
    ''' we begin with 100 categories'''
    self.set_category_range(self.init_output)

  def set_category_range(self, r):
    dp = DataProvider.get_by_name(self.data_provider)
    self.train_dp = dp(self.data_dir, self.train_range, category_range = range(r))
    self.test_dp = dp(self.data_dir, self.test_range, category_range = range(r))


  def train(self):
    MiniBatchTrainer.train(self)

    for i, cate in enumerate(self.range_list):
      self.set_category_range(cate)
      self.num_batch = self.curr_epoch = self.curr_batch = 0
      self.curr_minibatch = 0
      self.num_minibatch = self.train_minibatch_list[i]

      model = load(self.checkpoint_file)
      layers = model['model_state']['layers']

      fc = layers[-2]
      fc['weight'] = None
      fc['bias'] = None
      fc['weightIncr'] = None
      fc['biasIncr'] = None
      #for l in layers:
      #  if l['type'] == 'fc':
      #    l['weight'] = None
      #    l['bias'] = None
      #    l['weightIncr'] = None
      #    l['biasIncr'] = None

      #fc = layers[-2]
      fc['outputSize'] = cate

      self.learning_rate = self.learning_rate_list[i]
      self.net = FastNet(self.learning_rate, self.image_shape, self.n_out, init_model = model)

      self.net.clear_weight_incr()
      MiniBatchTrainer.train(self)




class ImageNetCateGroupTrainer(MiniBatchTrainer):
  def __init__(self, test_id, data_dir, data_provider, checkpoint_dir, train_range, test_range,
      test_freq, save_freq, batch_size, num_minibatch, image_size, image_color, learning_rate,
      num_group_list, init_model, adjust_freq = 100, factor = 1.0):

    self.train_minibatch_list = num_minibatch[1:]
    self.num_group_list = num_group_list[1:]
    self.learning_rate_list = learning_rate[1:]

    layers = init_model
    fc = layers[-2]
    fc['outputSize'] = num_group_list[0]

    MiniBatchTrainer.__init__(self, test_id, data_dir, data_provider, checkpoint_dir, train_range, test_range,
        test_freq, save_freq, batch_size, num_minibatch[0], image_size, image_color, learning_rate[0], init_model = init_model)


  def set_num_group(self, n):
    dp = DataProvider.get_by_name(self.data_provider)
    self.train_dp = dp(self.data_dir, self.train_range, n)
    self.test_dp = dp(self.data_dir, self.test_range, n)

  def init_data_provider(self):
    self.set_num_group(self.n_out)

  def train(self):
    MiniBatchTrainer.train(self)

    for i, group in enumerate(self.num_group_list):
      self.set_num_group(group)
      self.num_batch = self.curr_epoch = self.curr_batch = 0
      self.curr_minibatch = 0
      self.num_minibatch = self.train_minibatch_list[i]

      model = load(self.checkpoint_file)
      layers = model['model_state']['layers']

      fc = layers[-2]
      fc['outputSize'] = group
      fc['weight'] = None
      fc['bias'] = None
      fc['weightIncr'] = None
      fc['biasIncr'] = None

      self.learning_rate = self.learning_rate_list[i]
      self.net = FastNet(self.learning_rate, self.image_shape, self.n_out, init_model = model)

      self.net.clear_weight_incr()
      MiniBatchTrainer.train(self)





if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--test_id', help = 'Test Id', default = None, type = int)
  parser.add_argument('--data_dir', help = 'The directory that data stored')
  parser.add_argument('--param_file', help = 'The param_file or checkpoint file')
  parser.add_argument('--data_provider', help = 'The data provider', choices =['cifar10','imagenet', 'imagenetcategroup'])
  parser.add_argument('--train_range', help = 'The range of the train set')
  parser.add_argument('--test_range', help = 'THe range of the test set')
  parser.add_argument('--save_freq', help = 'How often should I save the checkpoint file', default = 100, type = int)
  parser.add_argument('--test_freq', help = 'How often should I test the model', default = 100, type = int)
  parser.add_argument('--adjust_freq', help = 'How often should I adjust the learning rate', default = 100, type = int)
  parser.add_argument('--factor', help = 'The factor used to adjust the learning rate', default ='1.0')
  parser.add_argument('--learning_rate' , help = 'The scale learning rate', default = '0.1')
  parser.add_argument('--batch_size', help = 'The size of batch', default = 128, type = int)
  parser.add_argument('--checkpoint_dir', help = 'The directory to save checkpoint file')

  parser.add_argument('--trainer', help = 'The type of the trainer', default = 'normal', choices =
      ['normal', 'catewise', 'categroup', 'minibatch', 'layerwise'])


  # extra argument
  extra_argument = ['num_group_list', 'num_caterange_list', 'num_epoch', 'num_minibatch']
  parser.add_argument('--num_group_list', help = 'The list of the group you want to split the data to')
  parser.add_argument('--num_caterange_list', help = 'The list of category range you want to train')
  parser.add_argument('--num_epoch', help = 'The number of epoch you want to train', default = 30, type = int)
  parser.add_argument('--num_minibatch', help = 'The number of minibatch you want to train(num*1000)')

  args = parser.parse_args()

  for a in [att for att in dir(args) if not att.startswith('__')]:
    if not getattr(args, a) and a not in extra_argument:
      assert False, 'You have to specify a value of %s' % a


  param_dict = {}
  param_dict['image_color'] = 3
  param_dict['test_id'] = args.test_id
  param_dict['data_dir'] = args.data_dir
  param_dict['data_provider'] = args.data_provider
  if args.data_provider.startswith('imagenet'):
    param_dict['image_size'] = 224
  elif args.data_provider.startswith('cifar'):
    param_dict['image_size'] = 32
  else:
    assert False, 'Unknown data_provider %s' % args.data_provider
 
  param_dict['train_range'] = util.string_to_int_list(args.train_range)
  param_dict['test_range'] = util.string_to_int_list(args.test_range)
  param_dict['save_freq'] = args.save_freq
  param_dict['test_freq'] = args.test_freq
  param_dict['adjust_freq'] = args.adjust_freq
  factor = util.string_to_float_list(args.factor)
  if len(factor) == 1:
    param_dict['factor'] = factor[0]
  else:
    param_dict['factor'] = factor


  learning_rate = util.string_to_float_list(args.learning_rate)
  if len(learning_rate) == 1:
    param_dict['learning_rate'] = learning_rate[0]
  else:
    param_dict['learning_rate'] = learning_rate

  param_dict['batch_size'] = args.batch_size
  param_dict['checkpoint_dir'] = args.checkpoint_dir
  trainer = args.trainer

  cp_pattern = param_dict['checkpoint_dir'] + '/test%d$' % param_dict['test_id']
  cp_files = glob.glob('%s*' % cp_pattern)

  if not cp_files:
    util.log('No checkpoint, starting from scratch.')
    param_dict['init_model'] = Parser(args.param_file).get_result()
  else:
    cp_file = sorted(cp_files, key=os.path.getmtime)[-1]
    util.log('Loading from checkpoint file: %s', cp_file)
    param_dict['init_model'] = util.load(cp_file)

  trainer = Trainer.get_trainer_by_name(trainer, param_dict, args)
  util.log('start to train...')
  trainer.train()
  #trainer.predict(['pool5'], 'image.opt')
