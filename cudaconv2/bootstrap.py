#import pycuda.autoinit

# MAGIC MAGIC
import pycuda.driver as cuda
cuda.init()
from pycuda.tools import make_default_context
context = make_default_context()
device = context.get_device()
import atexit
atexit.register(context.detach)

import pycuda.driver as cuda
import sys
from pycuda import gpuarray, driver
import cudaconv2

import numpy as np
from scipy.signal import convolve2d

imgSize = 32
filterSize = 5
padding = 2
color = 1
imgNum = 1
filterNum = 64

stride = 1
modulesX = 1 + int(((2 * padding + imgSize - filterSize) / float(stride)))


img = gpuarray.to_gpu(np.ones((imgSize * imgSize * color, imgNum)).astype(np.float32))
filter = gpuarray.to_gpu(np.ones((filterSize * filterSize * color, filterNum)).astype(np.float32))
target = gpuarray.to_gpu(np.ones((modulesX * modulesX * filterNum, imgNum)).astype(np.float32))

print 'standard output for convolution'
print convolve2d(np.ones((imgSize, imgSize)).astype(np.float32), np.ones((filterSize, filterSize)).astype(np.float32),'valid')
cudaconv2.convFilterActs(img, filter, target, imgSize, modulesX, modulesX, -padding, stride, color, 1, 0.0, 1.0)

print 'pycuda output for convolution'
atarget = target.get()

for i in range(atarget.shape[0]):
  print atarget[i, 0]


#from pycuda.compiler import *
#mod = SourceModule(open('foo.cu').read(), no_extern_c=True, include_dirs=['/home/justin/guppy/include'])
#kernel = mod.get_function('kernel')
#def i(x): return np.int32(x)
#
#grid = (1, 32 * 32 * 64 / (4 * 8), 1)
#blocks = (32, 4, 1)
#kernel(img, filter, target, i(1), i(64), i(32), i(32), i(5), i(-2), i(1), i(32), i(32), i(1),
#    np.float32(0.0), np.float32(1.0), np.int32(True), block=blocks, grid=grid)

