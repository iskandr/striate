import os
import cPickle
import numpy as np


def load(filename):
  with open(filename, 'rb') as f:
    d = cPickle.load(f)
  return d


class DataProvider(object):
  def __init__(self, data_dir='.', batch_size = 64):
    self.data_dir = data_dir
    self.data_file = os.path.join(data_dir, 'data_batch_1')
    self.meta_file = os.path.join(data_dir, 'batches.meta')
    
    self.batch_meta = load(self.meta_file)
    self.batch_data = load(self.data_file)
  
    self.batch_size = batch_size
    self.batch_data['data'] = np.require((self.batch_data['data'] - self.batch_meta['data_mean']),
        dtype = np.single, requirements = 'C')
    self.index = 0

    self.total_num = self.batch_data['data'].shape[1]
    

  def get_batch_data(self):
  
    if self.index + self.batch_size > self.total_num:
      self.index = 0
    d = {}
    d['data'] = self.batch_data['data'][:, self.index: self.index + self.batch_size]
    d['label'] = self.batch_data['labels'][self.index: self.index + self.batch_size]
    self.index += self.batch_size

    return d
    

  def get_batch_size(self):
    return self.batch_size


if __name__ == "__main__":
  data_dir = '/hdfs/cifar/data/cifar-10-python/'
  dp = DataProvider(data_dir)
  batch_data = dp.get_batch_data()
  print batch_data
