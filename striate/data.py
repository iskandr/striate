import os
import cPickle
import numpy as np
import re
import random
import threading

def load(filename):
  with open(filename, 'rb') as f:
    d = cPickle.load(f)
  return d


class DataProvider(object):
  BATCH_REGEX = re.compile('^data_batch_(\d+)$')
  def __init__(self, batch_size, data_dir='.', batch_range = None):
    self.data_dir = data_dir
    self.meta_file = os.path.join(data_dir, 'batches.meta')

    self.batch_meta = load(self.meta_file)
    self.batch_size = batch_size
    if batch_range is None:
      self.batch_range = get_batch_indexes(self.data_dir)
    else:
      self.batch_range = batch_range
    random.shuffle(self.batch_range)

    self.curr_batch_index = -1
    self.curr_batch = None
    self.curr_epoch = 0
    self.data = None


  def get_next_index(self):
    self.curr_batch_index = (self.curr_batch_index + 1 ) % len(self.batch_range)
    return self.curr_batch_index


  @staticmethod
  def get_batch_filenames(src_dir):
    return sorted([f for f in os.listdir(src_dir) if DataProvider.BATCH_REGEX.match(f)], key =
        alphanum_key)

  @staticmethod
  def get_batch_indexes(src_dir):
    names = get_batch_filename(src_dir)
    return sorted(list(set(int(DataProvider.BATCH_REGEX.match(n).group(1)) for n in names)))

  def get_next_batch(self):
    self.get_next_index()
    if self.curr_batch_index == 0:
      random.shuffle(self.batch_range)
      self.curr_epoch += 1
    self.curr_batch = self.batch_range[self.curr_batch_index]

    filename = os.path.join(self.data_dir, 'data_batch_%d' % self.curr_batch)

    self.data = load(filename)
    self.data['data'] = np.require((self.data['data']-self.batch_meta['data_mean']), dtype = np.single,
        requirements = 'C')
    self.data['labels'] = np.require(self.data['labels'], dtype = np.single, requirements = 'C')
    return  self.curr_epoch, self.curr_batch, self.data


  def get_batch_size(self):
    return self.batch_size

  def get_batch_num(self):
    return len(self.batch_range)

class ParallelDataProvider(DataProvider):
  def __init__(self, batch_size, data_dir='.', batch_range = None):
    DataProvider.__init__(self, batch_size, data_dir, batch_range)
    self.batch_return = None

  def run_in_back(self):
    self.batch_return = DataProvider.get_next_batch(self)

  def get_next_batch(self):
    self.thread = threading.Thread(target=self.run_in_back)
    self.thread.start()

  def wait(self):
    self.thread.join()
    return  self.batch_return



if __name__ == "__main__":
  data_dir = '/hdfs/cifar/data/cifar-10-python'
  dp = DataProvider(data_dir, [1, 2, 3, 4,5 ])
  for i in range(12):
    epoch, batch, data = dp.get_next_batch()
    print epoch, batch
