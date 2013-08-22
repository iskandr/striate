from mpi4py import MPI
from striate import trainer
from trainer import Trainer
from striate import fastnet
from fastnet import DistFastNet
from striate.layer import TEST, TRAIN
import argparse

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


class DummyTrainer(Trainer):
  def __init__(self, test_id, data_dir, data_provider, checkpoint_dir, train_range, test_range,
      test_freq, save_freq, batch_size, num_epoch, image_size, image_color, learning_rate,
      init_model, adjust_freq = 1, factor = 1.0):
    Trainer.__init__(self, data_dir, data_provider, checkpoint_dir, train_range, test_range,
        test_freq, save_freq, batch_size, num_epoch, image_size, image_color, learning_rate,
        init_model = None, adjust_freq = adjust_freq, factor = factor)
    self.net = DistFastNet(self.learning_rate, self.image_shape, self.n_out, init_model)

  def get_from_master(self, train = TRAIN):
    if train == TRAIN:
      data = comm.bcast(self.train_data, root = 0)
    else:
      data = comm.bcast(self.test_data, root = 0)
    comm.barrier()
    return data

  def train(self):
    util.log('rank %d starting training...' % rank)
    while self.should_continue_training():
      self.train_data = self.get_from_master(TRAIN)
      self.curr_epoch = self.train_data.epoch
      self.curr_batch = self.train_data.batchnum

      self.num_train_minibatch = divup(self.train_data.data.shape[1], self.batch_size)
      for i in range(self.num_train_minibatch):
        input, label = self.get_next_minibatch(i)
        self.net.train_batch(input, label)
        self.curr_minbatch += 1
      
      self.num_batch += 1
      if self.check_test_data():
        self.get_test_error()

      if self.factor != 1.0 and self.check_adjust_lr():
        self.net_adjust_learning_rate(self.factor)

  def get_test_error(self):
      self.test_data = self.get_from_master(TEST)

      self.num_test_minibatch = divup(self.test_data.data.shape[1], self.batch_size)
      for i in range(self.num_test_minibatch):
        input, label = self.get_next_minibatch(i, TEST)
        self.net.train_batch(input, label, TEST)

  def save_checkpoint(self):
    self.net.get_dumped_layers()
      

class ServerTrainer(Trainer):
  def __init__(self, test_id, data_dir, data_provider, checkpoint_dir, train_range, test_range,
      test_freq, save_freq, batch_size, num_epoch, image_size, image_color, learning_rate,
      init_model, adjust_freq = 1, factor = 1.0):
    Trainer.__init__(self, data_dir, data_provider, checkpoint_dir, train_range, test_range,
        test_freq, save_freq, batch_size, num_epoch, image_size, image_color, learning_rate,
        init_model = None, adjust_freq = adjust_freq, factor = factor)
    self.net = DistFastNet(self.learning_rate, self.image_shape, self.n_out, init_model)

  def get_next_minibatch(self, i, train=TRAIN):
    if train == TRAIN:
      data = self.train_data
    else:
      data = self.test_data

    batch_data = data.data
    batch_label = data.labels
    batch_size = self.batch_size

    mini_data = batch_data[:, i * batch_size: (i + 1) * batch_size]
    num_case = mini_data.size / (self.image_color * self.image_size * self.image_size)
    mini_data = mini_data.reshape((self.image_color, self.image_size, self.image_size, num_case))
    label = batch_label[i * batch_size : (i + 1) * batch_size]
    return mini_data, label

  def scatter_to_worker(self, train = TRAIN):
    if train == TRAIN:
      comm.bcast(self.train_data, root = 0)
    else:
      comm.bcast(self.test_data, root = 0)
    comm.barrier()
    
  def train(self):
    self.print_net_summary()
    util.log('Starting training...')
    while self.should_continue_training():
      self.train_data = self.train_dp.get_next_batch()  # self.train_dp.wait()
      self.scatter_to_worker(TRAIN)
      self.curr_epoch = self.train_data.epoch
      self.curr_batch = self.train_data.batchnum

      start = time.time()
      self.num_train_minibatch = divup(self.train_data.data.shape[1], self.batch_size)
      
      for i in range(self.num_train_minibatch):
        input, label = self.get_next_minibatch(i)
        self.net.train_batch(input, label)
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

    self.get_test_error()
    self.save_checkpoint()
  
  def get_test_error(self):
    start = time.time()
    self.test_data = self.test_dp.get_next_batch()
    self.scatter_to_worker(TEST)

    self.num_test_minibatch = divup(self.test_data.data.shape[1], self.batch_size)
    for i in range(self.num_test_minibatch):
      input, label = self.get_next_minibatch(i, TEST)
      self.net.train_batch(input, label, TEST)
    
    cost , correct, numCase, = self.net.get_batch_information()
    self.test_outputs += [({'logprob': [cost, 1 - correct]}, numCase, time.time() - start)]
    print >> sys.stderr,  '[%d] error: %f logreg: %f time: %f' % (self.test_data.batchnum, 1 - correct, cost, time.time() - start)


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

  param_dict['num_epoch'] = args.num_epoch
  if rank == 0:
    trainer = ServerTrainer(**param_dict)
  else:
    trainer = DummyTrainer(**param_dict)
  trainer.train()
