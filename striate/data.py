from PIL import Image
from os.path import basename
from striate import util
import Queue
import cPickle
import collections
import glob
import numpy as np
import os
import random
import re
import sys
import threading
import time

def load(filename):
  with open(filename, 'rb') as f:
    d = cPickle.load(f)
  return d


def join(lsts):
  out = []
  for l in lsts:
    for item in l:
      out.append(item)
  return out


BatchData = collections.namedtuple('BatchData', 
                                   ['data', 'labels', 'epoch', 'batchnum'])


dp_dict = {}

class DataProvider(object):
  BATCH_REGEX = re.compile('^data_batch_(\d+)$')
  def __init__(self, data_dir='.', batch_range=None):
    self.data_dir = data_dir
    self.meta_file = os.path.join(data_dir, 'batches.meta')

    self.curr_batch_index = -1
    self.curr_batch = None
    self.curr_epoch = 0
    self.data = None

    if os.path.exists(self.meta_file):
      self.batch_meta = load(self.meta_file)
    else:
      print 'No default meta file \'batches.meta\', using another meta file'

    if batch_range is None:
      self.batch_range = self.get_batch_indexes()
    else:
      self.batch_range = batch_range
    random.shuffle(self.batch_range)


  def get_next_index(self):
    self.curr_batch_index = (self.curr_batch_index + 1) % len(self.batch_range)
    return self.curr_batch_index

  def get_next_batch(self):
    return self._get_next_batch()

  def del_batch(self, batch):
    print 'delete batch', batch
    self.batch_range.remove(batch)
    print self.batch_range

  def get_batch_num(self):
    return len(self.batch_range)

  @classmethod
  def register_data_provider(cls, name, _class):
    if name in dp_dict:
      print 'Data Provider', name, 'already registered'
    else:
      dp_dict[name] = _class

  @classmethod
  def get_by_name(cls, name):
    if name not in dp_dict:
      print >> sys.stderr, 'There is no such data provider --', name, '--'
      sys.exit(-1)
    else:
      return dp_dict[name]
    
    
class ParallelDataProvider(DataProvider):
  def __init__(self, data_dir='.', batch_range=None):
    DataProvider.__init__(self, data_dir, batch_range)
    self._reader = None
    self._batch_return = None
    self._data_queue = Queue.Queue(1)

  def _start_read(self):
    assert self._reader is None
    self._reader = threading.Thread(target=self.run_in_back)
    self._reader.setDaemon(True)
    self._reader.start()

  def run_in_back(self):
    while 1:
      result = self._get_next_batch()
      self._data_queue.put(result)

  def get_next_batch(self):
    if self._reader is None:
      self._start_read()

    return self._data_queue.get()


class ImageNetDataProvider(ParallelDataProvider):
  def __init__(self, data_dir, batch_range=None, category_range=None, batch_size=128):
    ParallelDataProvider.__init__(self, data_dir, batch_range)
    self.img_size = 256
    self.border_size = 16
    self.inner_size = 224
    self.batch_size = batch_size
    
    # self.multiview = dp_params['multiview_test'] and test
    self.multiview = 0
    self.num_views = 5 * 2
    self.data_mult = self.num_views if self.multiview else 1

    self.buffer_idx = 0
    
    dirs = glob.glob(data_dir + '/n*')
    synid_to_dir = {}
    for d in dirs:
      synid_to_dir[basename(d)[1:]] = d
    
    if category_range is None:
      cat_dirs = dirs
    else:
      cat_dirs = []
      for i in category_range:
        synid = self.batch_meta['label_to_synid'][i]
        util.log('Using category: %d, synid: %s, label: %s',
                 i, synid, self.batch_meta['label_names'][i])
        cat_dirs.append(synid_to_dir[synid])

    self.images = []
    batch_dict = dict((k, k) for k in self.batch_range)
    for d in cat_dirs:
      imgs = [v for i, v in enumerate(glob.glob(d + '/*.jpg'))
              if i in batch_dict]
      self.images.extend(imgs)
    self.images = np.array(self.images)
    
    # build index vector into 'images' and split into groups of batch-size
    image_index = np.arange(len(self.images))
    np.random.shuffle(image_index)
    
    self.batches = np.array_split(image_index, 
                                  util.divup(len(self.images), batch_size))
    
    self.batch_range = range(len(self.batches))
    
    util.log('Starting data provider with %d batches', len(self.batches))
    np.random.shuffle(self.batch_range)
    
    imagemean = cPickle.loads(open(data_dir + "image-mean.pickle").read())
    self.data_mean = (imagemean['data']
        .astype(np.single)
        .T
        .reshape((3, 256, 256))[:, self.border_size:self.border_size + self.inner_size, self.border_size:self.border_size + self.inner_size]
        .reshape((self.get_data_dims(), 1)))
  

  def __trim_borders(self, images, target):
    for idx, img in enumerate(images):
      startY, startX = np.random.randint(0, self.border_size * 2 + 1), np.random.randint(0, self.border_size * 2 + 1)
      endY, endX = startY + self.inner_size, startX + self.inner_size
      pic = img[:, startY:endY, startX:endX]
      if np.random.randint(2) == 0:  # also flip the image with 50% probability
        pic = pic[:, :, ::-1]
      target[:, idx] = pic.reshape((self.get_data_dims(),))

  def _get_next_batch(self):
    start = time.time()
    self.get_next_index()
    
    self.curr_batch = self.batch_range[self.curr_batch_index]
    epoch = self.curr_epoch
    batchnum = self.curr_batch
    names = self.images[self.batches[batchnum]]
    num_imgs = len(names)

    labels = np.zeros((1, num_imgs))
    cropped = np.ndarray((self.get_data_dims(), num_imgs * self.data_mult), dtype=np.uint8)

    # load in parallel for training
    st = time.time()
    images = []
    for idx, filename in enumerate(names):
      jpeg = Image.open(filename)
      if jpeg.mode != "RGB": jpeg = jpeg.convert("RGB")
      # starts as rows * cols * rgb, tranpose to rgb * rows * cols
      img = np.asarray(jpeg, np.uint8).transpose(2, 0, 1)
      images.append(img)

    self.__trim_borders(images, cropped)

    load_time = time.time() - st

    # extract label from the filename
    for idx, filename in enumerate(names):
      filename = os.path.basename(filename)
      synid = filename[1:].split('_')[0]
      label = self.batch_meta['synid_to_label'][synid]
      labels[0, idx] = label

    st = time.time()
    cropped = cropped.astype(np.single)
    cropped = np.require(cropped, dtype=np.single, requirements='C')
    cropped -= self.data_mean

    align_time = time.time() - st

    labels = np.array(labels)
    labels = labels.reshape(cropped.shape[1],)
    labels = np.require(labels, dtype=np.single, requirements='C')

    util.log("Loaded %d images in %.2f seconds (%.2f load, %.2f align)",
             num_imgs, time.time() - start, load_time, align_time)
    # self.data = {'data' : SharedArray(cropped), 'labels' : SharedArray(labels)}

    return BatchData(cropped, labels, epoch, batchnum)

  # Returns the dimensionality of the two data matrices returned by get_next_batch
  # idx is the index of the matrix.
  def get_data_dims(self, idx=0):
    return self.inner_size ** 2 * 3 if idx == 0 else 1

  def get_plottable_data(self, data):
    return np.require(
       (data + self.data_mean).T
       .reshape(data.shape[1], 3, self.inner_size, self.inner_size)
       .swapaxes(1, 3)
       .swapaxes(1, 2) / 255.0,
        dtype=np.single)


class CifarDataProvider(DataProvider):
  def _get_next_batch(self):
    self.get_next_index()
    if self.curr_batch_index == 0:
      random.shuffle(self.batch_range)
      self.curr_epoch += 1
    self.curr_batch = self.batch_range[self.curr_batch_index]
    # print self.batch_range, self.curr_batch

    filename = os.path.join(self.data_dir, 'data_batch_%d' % self.curr_batch)

    self.data = load(filename)
    return BatchData(self.data['data'] - self.batch_meta['data_mean'], 
                     np.array(self.data['labels']), 
                     self.curr_epoch, 
                     self.curr_batch)

  def get_batch_filenames(self):
    return sorted([f for f in os.listdir(self.data_dir) if DataProvider.BATCH_REGEX.match(f)], key=alphanum_key)

  def get_batch_indexes(self):
    names = self.get_batch_filenames()
    return sorted(list(set(int(DataProvider.BATCH_REGEX.match(n).group(1)) for n in names)))


DataProvider.register_data_provider('cifar10', CifarDataProvider)
DataProvider.register_data_provider('imagenet', ImageNetDataProvider)


if __name__ == "__main__":
  data_dir = '/ssd/nn-data/imagenet/'
  dp = ImageNetDataProvider(data_dir, range(1, 10), category_range = range(1, 10))
  # data_dir = '/hdfs/cifar/data/cifar-10-python/'
  # dp = DataProvider(data_dir, [1, 2, 3, 4, 5 ])
  for i in range(1):
    data = dp.get_next_batch()
    print data.data.shape
