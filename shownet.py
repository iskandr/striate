import numpy as n
from math import sqrt, ceil, floor
import sys
import getopt as opt
import os
import random as r
import numpy.random as nr
import pylab as pl
import striate.util as util
import matplotlib.pyplot as plt



class ShowNetError(Exception):
    pass


class ShowConvNet:
  def __init__(self, checkpoint, show_filters, channels = 3):
    self.checkpoint = checkpoint
    self.model = util.load(self.checkpoint)
    self.layers = self.model['model_state']['layers']
    self.show_filters = show_filters
    self.channels = channels

  def plot_filters(self):
    print 'HERE'
    filter_start = 0 # First filter to show
    layer_names = [l['name'] for l in self.layers]
    if self.show_filters not in layer_names:
      raise ShowNetError("Layer with name '%s' not defined by given convnet." % self.show_filters)
    layer = self.layers[layer_names.index(self.show_filters)]
    filters = layer['weight']
    if layer['type'] == 'fc': # Fully-connected layer
      num_filters = layer['outputSize']
      channels = self.channels
    elif layer['type'] in ('conv', 'local'): # Conv layer
      num_filters = layer['numFilter']
      channels = layer['numColor']

    print num_filters, channels

    filters = filters.reshape(channels, filters.shape[0]/channels, filters.shape[1])
    # Convert YUV filters to RGB
    #if self.yuv_to_rgb and channels == 3:
    #  R = filters[0,:,:] + 1.28033 * filters[2,:,:]
    #    G = filters[0,:,:] + -0.21482 * filters[1,:,:] + -0.38059 * filters[2,:,:]
    #    B = filters[0,:,:] + 2.12798 * filters[1,:,:]
    #    filters[0,:,:], filters[1,:,:], filters[2,:,:] = R, G, B
    combine_chans = True #not self.no_rgb and channels == 3

    # Make sure you don't modify the backing array itself here -- so no -= or /=
    filters = filters - filters.min()
    filters = filters / filters.max()
    print 'try to make filter fig'
    self.make_filter_fig(filters, filter_start, 2, 'Layer %s' % self.show_filters, num_filters, combine_chans)


  def make_filter_fig(self, filters, filter_start, fignum, _title, num_filters, combine_chans):
    FILTERS_PER_ROW = 16
    MAX_ROWS = 16
    MAX_FILTERS = FILTERS_PER_ROW * MAX_ROWS
    num_colors = filters.shape[0]
    f_per_row = int(ceil(FILTERS_PER_ROW / float(1 if combine_chans else num_colors)))
    filter_end = min(filter_start+MAX_FILTERS, num_filters)
    filter_rows = int(ceil(float(filter_end - filter_start) / f_per_row))

    filter_size = int(sqrt(filters.shape[1]))
    fig = pl.figure(fignum)
    fig.set_figheight(16)
    fig.set_figwidth(8)

    fig.text(.5, .95, '%s %dx%d filters %d-%d' % (_title, filter_size, filter_size, filter_start, filter_end-1), horizontalalignment='center') 
    num_filters = filter_end - filter_start
    if not combine_chans:
      bigpic = n.zeros((filter_size * filter_rows + filter_rows + 1, filter_size*num_colors * f_per_row + f_per_row + 1), dtype=n.single)
    else:
      bigpic = n.zeros((3, filter_size * filter_rows + filter_rows + 1, filter_size * f_per_row + f_per_row + 1), dtype=n.single)

    for m in xrange(filter_start,filter_end ):
      filter = filters[:,:,m]
      y, x = (m - filter_start) / f_per_row, (m - filter_start) % f_per_row
      if not combine_chans:
        for c in xrange(num_colors):
          filter_pic = filter[c,:].reshape((filter_size,filter_size))
          bigpic[1 + (1 + filter_size) * y:1 + (1 + filter_size) * y + filter_size, 1 + (1 + filter_size*num_colors) * x + filter_size*c:1 + (1 + filter_size*num_colors) * x + filter_size*(c+1)] = filter_pic
      else:
        filter_pic = filter.reshape((3, filter_size,filter_size))
        bigpic[:,
            1 + (1 + filter_size) * y:1 + (1 + filter_size) * y + filter_size,
            1 + (1 + filter_size) * x:1 + (1 + filter_size) * x + filter_size] = filter_pic

    pl.xticks([])
    pl.yticks([])
    if not combine_chans:
      pl.imshow(bigpic, cmap=pl.cm.gray, interpolation='nearest')
    else:
      print bigpic.shape
      print bigpic.dtype
      bigpic = bigpic.swapaxes(0,2).swapaxes(0,1)
      pl.imshow(bigpic, interpolation='nearest')        
    pl.show()


if __name__ == '__main__':
  checkpoint = sys.argv[1]
  print 'shownet from', checkpoint

  show_filters = sys.argv[2]
  print 'showing layer', show_filters

  channel = 3

  ShowConvNet(checkpoint, show_filters, channel).plot_filters()
