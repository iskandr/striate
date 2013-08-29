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
  def _finish_init(self):
    self.net = DistFastNet(self.learning_rate, self.image_shape, self.n_out, self.init_model)

  def get_from_master(self, data):
    data = comm.bcast(data, root = 0)
    comm.barrier()
    return data

  def train(self):
    util.log('rank %d starting training...' % rank)
    while self.should_continue_training():
      train_data = self.get_from_master(None)
      self.curr_epoch = self.train_data.epoch

      input, label = train_data.data, train_data.labels
      self.net.train_batch(input, label)
      self.curr_batch += 1

      if self.check_test_data():
        self.get_test_error()

      if self.factor != 1.0 and self.check_adjust_lr():
        self.adjust_lr()

  def get_test_error(self):
      test_data = self.get_from_master(None)
      input, label = test_data.data, test_data.labels
      self.net.train_batch(input, label, TEST)

  def save_checkpoint(self):
    self.net.get_dumped_layers()


class ServerTrainer(Trainer):
  def _finish_init(self):
    self.net = DistFastNet(self.learning_rate, self.image_shape, self.init_model)

  def reshape_data(self, data):
    batch_data = data.data.get()
    batch_size = self.batch_size

    num_case = batch_data.size / (self.image_color * self.image_size * self.image_size)
    batch_data = batch_data.reshape((self.image_color, self.image_size, self.image_size, num_case))
    data.data = batch_data

  def scatter_to_worker(self, data):
    comm.bcast(data, root = 0)
    comm.barrier()

  def train(self):
    self.print_net_summary()
    util.log('Starting training...')
    while self.should_continue_training():
      train_data = self.train_dp.get_next_batch()  # self.train_dp.wait()
      self.reshape_data(train_data)
      self.scatter_to_worker(train_data)
      self.curr_epoch = train_data.epoch

      input, label = train_data.data, train_data.labels
      self.net.train_batch(input, label)

      self.curr_batch += 1
      cost , correct, numCase = self.net.get_batch_information()
      self.train_outputs += [({'logprob': [cost, 1 - correct]}, numCase, time.time() - start)]
      print >> sys.stderr,  '%d.%d: error: %f logreg: %f time: %f' % (self.curr_epoch, self.curr_batch, 1 - correct, cost, time.time() - start)

      self.num_batch += 1
      if self.check_test_data():
        self.get_test_error()

      if self.factor != 1.0 and self.check_adjust_lr():
        self.adjust_lr()

      if self.check_save_checkpoint():
        self.save_checkpoint()

    self.get_test_error()
    self.save_checkpoint()

  def get_test_error(self):
    start = time.time()
    test_data = self.test_dp.get_next_batch()
    self.reshape_data(test_data)
    self.scatter_to_worker(test_data)

    input, label = test_data.data, test_data.labels
    self.net.train_batch(input, label, TEST)

    cost , correct, numCase, = self.net.get_batch_information()
    self.test_outputs += [({'logprob': [cost, 1 - correct]}, numCase, time.time() - start)]
    print >> sys.stderr,  'error: %f logreg: %f time: %f' % (1 - correct, cost, time.time() - start)


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
  extra_argument = ['num_group_list', 'num_caterange_list', 'num_epoch', 'num_batch']
  parser.add_argument('--num_group_list', help = 'The list of the group you want to split the data to')
  parser.add_argument('--num_caterange_list', help = 'The list of category range you want to train')
  parser.add_argument('--num_epoch', help = 'The number of epoch you want to train', default = 30, type = int)
  parser.add_argument('--num_batch', help = 'The number of minibatch you want to train(num*1000)')

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


  #create a checkpoint dumper
  image_shape = (param_dict['image_color'], param_dict['image_size'], param_dict['image_size'], param_dict['batch_size'])
  param_dict['image_shape'] = image_shape
  cp_dumper = CheckpointDumper(param_dict['checkpoint_dir'], param_dict['test_id'])
  param_dict['checkpoint_dumper'] = cp_dumper

  #create the init_model
  init_model = cp_dumper.get_checkpoint()
  if init_model is None:
    init_model = parse_config_file(args.param_file)
  param_dict['init_model'] = init_model

  #create train dataprovider and test dataprovider
  dp_class = DataProvider.get_by_name(param_dict['data_provider'])
  train_dp = dp_class(param_dict['data_dir'], param_dict['train_range'])
  test_dp = dp_class(param_dict['data_dir'], param_dict['test_range'])
  param_dict['train_dp'] = train_dp
  param_dict['test_dp'] = test_dp


  #get all extra information
  param_dict['num_epoch'] = args.num_epoch
  num_batch = util.string_to_int_list(args.num_batch)
  if len(num_batch) == 1:
    param_dict['num_batch'] = num_batch[0]
  else:
    param_dict['num_batch'] = num_batch

  param_dict['num_group_list']  = util.string_to_int_list(args.num_group_list)
  param_dict['num_caterange_list'] = util.string_to_int_list(args.num_caterange_list)


  if rank == 0:
    trainer = ServerTrainer(**param_dict)
  else:
    trainer = DummyTrainer(**param_dict)
  trainer.train()
