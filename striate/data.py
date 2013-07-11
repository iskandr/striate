import os
import cPickle
import numpy as np
import re
import logging
import random
import threading
import time
import zipfile
from PIL import Image
import cStringIO as c
import synids

def load(filename):
  with open(filename, 'rb') as f:
    d = cPickle.load(f)
  return d


class DataProvider(object):
  BATCH_REGEX = re.compile('^data_batch_(\d+)$')
  def __init__(self, data_dir='.', batch_range = None):
    self.data_dir = data_dir
    self.meta_file = os.path.join(data_dir, 'batches.meta')

    if os.path.exists(self.meta_file):
      self.batch_meta = load(self.meta_file)
    else:
      print 'No default meta file \'batches.meta\', using another meta file'
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
    #print self.batch_range, self.curr_batch

    filename = os.path.join(self.data_dir, 'data_batch_%d' % self.curr_batch)

    self.data = load(filename)
    self.data['data'] = self.data['data']-self.batch_meta['data_mean']
    self.data['labels'] = np.array(self.data['labels'])
    return  self.curr_epoch, self.curr_batch, self.data

  def del_batch(self, batch):
    print 'delete batch', batch
    self.batch_range.remove(batch)
    print self.batch_range



  def get_batch_num(self):
    return len(self.batch_range)

class ParallelDataProvider(DataProvider):
  def __init__(self,  data_dir='.', batch_range = None):
    DataProvider.__init__(self,  data_dir, batch_range)
    self.batch_return = None

  def run_in_back(self):
    self.batch_return = DataProvider.get_next_batch(self)

  def get_next_batch(self):
    self.thread = threading.Thread(target=self.run_in_back)
    self.thread.start()

  def wait(self):
    self.thread.join()
    return  self.batch_return



class ImageNetDataProvider(DataProvider):
  BATCHES_PER_FILE = 1
  def __init__(self, data_dir, batch_range = None):
    DataProvider.__init__(self, data_dir, batch_range)
    self.img_size = 256
    self.border_size = 16
    self.inner_size = 224
    #self.multiview = dp_params['multiview_test'] and test
    self.multiview = 0
    self.num_views = 5*2
    self.data_mult = self.num_views if self.multiview else 1

    self.buffer_idx = 0

    imagemean= cPickle.loads(open(data_dir + "image-mean.pickle").read())
    self.data_mean = (imagemean['data']
        .astype(np.single)
        .T
        .reshape((3, 256, 256))[:, self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size]
        .reshape((self.get_data_dims(), 1)))

  def __trim_borders(self, images, target):
    num_images = len(images)
    #if self.test: # don't need to loop over cases
    #  if self.multiview:
    #    start_positions = [(0,0),  (0, self.border_size*2),
    #        (self.border_size, self.border_size),
    #        (self.border_size*2, 0), (self.border_size*2, self.border_size*2)]
    #    end_positions = [(sy+self.inner_size, sx+self.inner_size) for (sy,sx) in start_positions]
    #    for idx, img in enumerate(images):
    #      for i in xrange(self.num_views/2):
    #        pic = img[:, start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1]]
    #        target[:,i * num_images:(i+1)* num_images] = pic.reshape((self.get_data_dims(),))
    #        target[:,(self.num_views/2 + i) * num_images:(self.num_views/2 +i+1)* num_images] = pic[:,:,::-1,:].reshape((self.get_data_dims(),))
    #  else:
    #    for idx, img in enumerate(images):
    #      pic = img[:, self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size]
    #      target[:, idx] = pic.reshape((self.get_data_dims(),))
    #else:
    for idx, img in enumerate(images):
      startY, startX = np.random.randint(0, self.border_size * 2 + 1), np.random.randint(0, self.border_size * 2 + 1)
      endY, endX = startY + self.inner_size, startX + self.inner_size
      pic = img[:, startY:endY, startX:endX]
      if np.random.randint(2) == 0:  # also flip the image with 50% probability
        pic = pic[:, :, ::-1]
      target[:, idx] = pic.reshape((self.get_data_dims(),))

  def get_next_batch(self):
    start = time.time()
    self.get_next_index()
    if self.curr_batch_index == 0:
      random.shuffle(self.batch_range)
      self.curr_epoch += 1
    self.curr_batch = self.batch_range[self.curr_batch_index]
    epoch = self.curr_epoch

    batchnum = self.curr_batch
    filenum = batchnum / ImageNetDataProvider.BATCHES_PER_FILE
    batch_offset = batchnum % ImageNetDataProvider.BATCHES_PER_FILE

    print 'Loading from %s' % filenum
    zf = zipfile.ZipFile(self.data_dir + '/part-%05d' % filenum)

    names = zf.namelist()
    start_off = len(names) * batch_offset / ImageNetDataProvider.BATCHES_PER_FILE
    stop_off = len(names) * (batch_offset + 1) / ImageNetDataProvider.BATCHES_PER_FILE
    names = names[start_off:stop_off]
    num_imgs = len(names)

    labels = np.zeros((1, num_imgs))
    cropped = np.ndarray((self.get_data_dims(), num_imgs * self.data_mult), dtype=np.uint8)

    # load in parallel for training
    st = time.time()
    images = []
    for idx, filename in enumerate(names):
      jpeg = Image.open(c.StringIO(zf.read(filename)))
      if jpeg.mode != "RGB": jpeg = jpeg.convert("RGB")
      # starts as rows * cols * rgb, tranpose to rgb * rows * cols
      img = np.asarray(jpeg, np.uint8).transpose(2, 0, 1)
      images.append(img)

    self.__trim_borders(images, cropped)

    load_time = time.time() - st

    # extract label from the filename
    for idx, filename in enumerate(names):
      synid = filename[1:].split('_')[0]
      label = synids.SYNID_TO_LABEL[synid]
      labels[0, idx] = label

    st = time.time()
    cropped = cropped.astype(np.single)
    cropped = np.require(cropped, dtype=np.single, requirements='C')
    cropped -= self.data_mean
    align_time = time.time() - st

    labels = np.array(labels)
    labels = labels.reshape(cropped.shape[1], )
    labels = np.require(labels, dtype=np.single, requirements='C')

    logging.info("Loaded %d images in %.2f seconds (%.2f load, %.2f align)",
                 num_imgs, time.time() - start, load_time, align_time)
    self.data = {'data':cropped, 'labels':labels}
    return epoch, batchnum, self.data

  # Returns the dimensionality of the two data matrices returned by get_next_batch
  # idx is the index of the matrix.
  def get_data_dims(self, idx=0):
    return self.inner_size**2 * 3 if idx == 0 else 1

  def get_plottable_data(self, data):
    return np.require(
       (data + self.data_mean).T
       .reshape(data.shape[1], 3, self.inner_size, self.inner_size)
       .swapaxes(1,3)
       .swapaxes(1,2) / 255.0,
        dtype=np.single)

if __name__ == "__main__":
  data_dir = '/hdfs/imagenet/batches/imagesize-256/'
  dp = ImageNetDataProvider(data_dir, [1, 2, 3, 4, 5 ])
  #data_dir = '/hdfs/cifar/data/cifar-10-python/'
  #dp = DataProvider(data_dir, [1, 2, 3, 4, 5 ])
  for i in range(1):
    epoch, batch, data = dp.get_next_batch()
    print data['data'].shape, data['labels'].shape
    print epoch, batch
