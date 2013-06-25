import os
import cPickle
import numpy as np
import re


def load(filename):
  with open(filename, 'rb') as f:
    d = cPickle.load(f)
  return d


class DataProvider(object):
  BATCH_REGEX = re.compile('^data_batch_(\d+)$')
  def __init__(self, data_dir='.', batch_range = None):
    self.data_dir = data_dir
    self.meta_file = os.path.join(data_dir, 'batches.meta')

    self.batch_meta = load(self.meta_file)
    if batch_range is None:
      self.batch_range = get_batch_indexes(self.data_dir)
    else:
      self.batch_range = batch_range

    self.curr_batch_index = -1
    self.curr_batch = None
    self.curr_epoch = 0


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
      self.curr_epoch += 1
    self.curr_batch = self.batch_range[self.curr_batch_index]

    filename = os.path.join(self.data_dir, 'data_batch_%d' % self.curr_batch)

    data = load(filename)
    data['data'] = np.require((data['data']-self.batch_meta['data_mean']), dtype = np.single,
        requirements = 'C')
    return  self.curr_epoch, self.curr_batch, data


  def get_batch_size(self):
    return self.batch_size

  def get_batch_num(self):
    return len(self.batch_range)

if __name__ == "__main__":
  data_dir = '/hdfs/cifar/data/cifar-10-python'
  dp = DataProvider(data_dir, [1, 2, 3, 4,5 ])
  for i in range(12):
    epoch, data = dp.get_next_batch()
    print epoch
