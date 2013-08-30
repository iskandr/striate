from PIL import Image
from pycuda import gpuarray, driver
from striate.cuda_kernel import gpu_partial_copy_to, print_matrix
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
BatchData = collections.namedtuple('BatchData',
                                   ['data', 'labels', 'epoch'])


dp_dict = {}



class DataProvider(object):
  def __init__(self, data_dir='.', batch_range=None):
    self.data_dir = data_dir
    self.meta_file = os.path.join(data_dir, 'batches.meta')

    self.curr_batch_index = 0
    self.curr_batch = None
    self.curr_epoch = 1
    self.data = None

    if os.path.exists(self.meta_file):
      self.batch_meta = util.load(self.meta_file)
    else:
      print 'No default meta file \'batches.meta\', using another meta file'

    if batch_range is None:
      self.batch_range = self.get_batch_indexes()
    else:
      self.batch_range = batch_range
    random.shuffle(self.batch_range)
    self.data_on_GPU = None
    self.data = None
    self.labels = None
    self.index = 0

  def copy_to_GPU(self):
      self.data_on_GPU = gpuarray.to_gpu(self.data.astype(np.float32))

  def reset(self):
    self.curr_batch_index = 0
    self.curr_batch = None
    self.curr_epoch = 1
    self.data = None

    random.shuffle(self.batch_range)
    self.data_on_GPU = None
    self.data = None
    self.labels = None


  def get_next_index(self):
    self.curr_batch_index = (self.curr_batch_index + 1) % len(self.batch_range)
    return self.curr_batch_index

  def get_next_batch(self, batch_size):
    if self.data_on_GPU is  None:
      self._get_next_batch()
      self.copy_to_GPU()

    height, width = self.data_on_GPU.shape
    if self.index + batch_size >  width:
      labels = self.labels[self.index:]
      width = width - self.index
      data = gpuarray.zeros((height, width), dtype = np.float32)
      gpu_partial_copy_to(self.data_on_GPU, data, 0, height, self.index, self.index + width)
      self._get_next_batch()
      self.copy_to_GPU()
      self.index = 0
    else:
      data = gpuarray.zeros((height, batch_size), dtype = np.float32)
      gpu_partial_copy_to(self.data_on_GPU, data, 0, height, self.index, self.index + batch_size)
      labels = self.labels[self.index:self.index + batch_size]
      self.index += batch_size
    return BatchData(data, labels, self.curr_epoch)



  def del_batch(self, batch):
    print 'delete batch', batch
    self.batch_range.remove(batch)
    print self.batch_range

  def get_batch_num(self):
    return len(self.batch_range)

  @staticmethod
  def register_data_provider(name, _class):
    if name in dp_dict:
      print 'Data Provider', name, 'already registered'
    else:
      dp_dict[name] = _class

  @staticmethod
  def get_by_name(name):
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
    self._command_queue = Queue.Queue(1)
    self.reserved_epoch = 0
    self.reserved_labels = None
    self.reserved_data_on_GPU = None

  def _start_read(self):
    assert self._reader is None
    self._reader = threading.Thread(target=self.run_in_back)
    self._reader.setDaemon(True)
    self._reader.start()
    self._command_queue.put(1)

  def reset(self):
    DataProvider.reset(self)
    self._reader = None
    self._batch_return = None
    self._data_queue = Queue.Queue(1)
    self._command_queue = Queue.Queue(1)
    self.reserved_epoch = 0
    self.reserved_labels = None
    self.reserved_data_on_GPU = None


  def run_in_back(self):
    while 1:
      self._command_queue.get()
      self._get_next_batch()
      self._data_queue.put(1)

  def _fill_reserved_data(self):
    self._data_queue.get()
    self.copy_to_GPU()
    self.reserved_epoch = self.curr_epoch
    self.reserved_labels = self.labels.copy()
    self.reserved_data_on_GPU = self.data_on_GPU.copy()
    assert self.reserved_data_on_GPU.shape[1] == self.reserved_labels.shape[0]
    self._command_queue.put(1)

  def get_next_batch(self, batch_size):
    if self._reader is None:
      self._start_read()

    if self.data_on_GPU is None:
      self._fill_reserved_data()

    height, width = self.reserved_data_on_GPU.shape
    if self.index + batch_size >=  width:
      labels = self.reserved_labels[self.index:]
      width = width - self.index
      data = gpuarray.zeros((height, width), dtype = np.float32)
      gpu_partial_copy_to(self.reserved_data_on_GPU, data, 0, height, self.index, self.index + width)
      self.index = 0
      self._fill_reserved_data()
    else:
      data = gpuarray.zeros((height, batch_size), dtype = np.float32)
      gpu_partial_copy_to(self.reserved_data_on_GPU, data, 0, height, self.index, self.index + batch_size)
      labels = self.reserved_labels[self.index:self.index + batch_size]
      self.index += batch_size
    return BatchData(data, labels, self.reserved_epoch)




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
        # util.log('Using category: %d, synid: %s, label: %s', i, synid, self.batch_meta['label_names'][i])
        cat_dirs.append(synid_to_dir[synid])

    self.images = []
    batch_dict = dict((k, k) for k in self.batch_range)

    for d in cat_dirs:
      imgs = [v for i, v in enumerate(glob.glob(d + '/*.jpg')) if i in batch_dict]
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
      # startY, startX = np.random.randint(0, self.border_size * 2 + 1), np.random.randint(0, self.border_size * 2 + 1)
      startY, startX = 0, 0
      endY, endX = startY + self.inner_size, startX + self.inner_size
      pic = img[:, startY:endY, startX:endX]
      if np.random.randint(2) == 0:  # also flip the image with 50% probability
        pic = pic[:, :, ::-1]
      target[:, idx] = pic.reshape((self.get_data_dims(),))

  def _get_next_batch(self):
    start = time.time()
    self.get_next_index()

    self.curr_batch = self.batch_range[self.curr_batch_index]
    if self.curr_batch_index == 0:
      self.curr_epoch += 1

    epoch = self.curr_epoch
    batchnum = self.curr_batch
    names = self.images[self.batches[batchnum]]
    num_imgs = len(names)
    labels = np.zeros((1, num_imgs))
    cropped = np.ndarray((self.get_data_dims(), num_imgs * self.data_mult), dtype=np.uint8)
    # _load in parallel for training
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

    clabel = []
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

    # util.log("Loaded %d images in %.2f seconds (%.2f _load, %.2f align)",
    #         num_imgs, time.time() - start, load_time, align_time)
    # self.data = {'data' : SharedArray(cropped), 'labels' : SharedArray(labels)}
    self.data = cropped
    self.labels = labels
    #return cropped, labels, epoch

  # Returns the dimensionality of the two data matrices returned by get_next_batch
  # idx is the index of the matrix.
  def get_data_dims(self, idx=0):
    return self.inner_size ** 2 * 3 if idx == 0 else 1

  @property
  def image_shape(self):
    return (3, self.inner_size, self.inner_size)


class CifarDataProvider(ParallelDataProvider):
  BATCH_REGEX = re.compile('^data_batch_(\d+)$')
  def _get_next_batch(self):
    self.get_next_index()
    if self.curr_batch_index == 0:
      random.shuffle(self.batch_range)
      self.curr_epoch += 1
    self.curr_batch = self.batch_range[self.curr_batch_index]
    # print self.batch_range, self.curr_batch

    filename = os.path.join(self.data_dir, 'data_batch_%d' % self.curr_batch)

    data = util.load(filename)
    img = data['data'] - self.batch_meta['data_mean']
    self.labels = np.array(data['labels'])
    self.data = np.require(img, requirements='C', dtype=np.float32)
    #return data, self.labels, self.curr_epoch

  def get_batch_filenames(self):
    return sorted([f for f in os.listdir(self.data_dir) if CifarDataProvider.BATCH_REGEX.match(f)], key=alphanum_key)

  def get_batch_indexes(self):
    names = self.get_batch_filenames()
    return sorted(list(set(int(DataProvider.BATCH_REGEX.match(n).group(1)) for n in names)))

  @property
  def image_shape(self):
    return (3, 32, 32)



class ImageNetCateGroupDataProvider(ImageNetDataProvider):
  TOTAL_CATEGORY = 1000
  def __init__(self, data_dir, batch_range, num_group = 100, batch_size=128):
    ImageNetDataProvider.__init__(self, data_dir, batch_range)
    self.num_group = num_group

  def _get_next_batch(self):
    ImageNetDataProvider._get_next_batch(self)
    labels = self.labels / (ImageNetCateGroupDataProvider.TOTAL_CATEGORY / self.num_group)
    labels = labels.astype(np.int).astype(np.float)
    self.labels = labels
    #return data.data, labels, data.epoch


class IntermediateDataProvider(ParallelDataProvider):
  def __init__(self, data_dir, batch_range, data_name):
    ParallelDataProvider.__init__(self, data_dir, batch_range)
    self.data_name = data_name

  def _get_next_batch(self):
    self.get_next_index()

    if self.curr_batch_index == 0:
      random.shuffle(self.batch_range)
      self.curr_epoch += 1
    self.curr_batch = self.batch_range[self.curr_batch_index]

    filename = os.path.join(self.data_dir + '.%s' % self.curr_batch)

    data_dic = util.load(filename)
    #data = np.concantenate([data[self.data_name] for data in data_list], axis = 1)
    #labels = np.concatenate([np.array( data['labels'].tolist() ) for data in data_list])
    data  = data_dic[self.data_name].transpose()
    labels = data_dic['labels']
    self.labels = labels
    data = np.require(data, requirements='C', dtype=np.float32)
    self.data = data
    #return data, labels, self.curr_epoch


DataProvider.register_data_provider('cifar10', CifarDataProvider)
DataProvider.register_data_provider('imagenet', ImageNetDataProvider)
DataProvider.register_data_provider('imagenetcategroup', ImageNetCateGroupDataProvider)
DataProvider.register_data_provider('intermediate', IntermediateDataProvider)


if __name__ == "__main__":
  data_dir = '/ssd/nn-data/cifar-10.old/'
  dp = CifarDataProvider(data_dir, [1])
  batch_size = 128
  # data_dir = '/hdfs/cifar/data/cifar-10-python/'
  # dp = DataProvider(data_dir, [1, 2, 3, 4, 5 ])
  data_list = []
  for i in range(11000):
    data = dp.get_next_batch(batch_size)
    data = data.data.get()
    data_list.append(data)

    if data.shape[1] != batch_size:
      break
  data = np.concatenate(data_list, axis = 1)
  print_matrix(data, 'data')
