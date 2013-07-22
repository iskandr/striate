from pycuda import autoinit, driver, gpuarray
import numpy as N
import time


DIM = 10000
src = N.ndarray((DIM, DIM), dtype=N.float)

x = gpuarray.GPUArray((DIM, DIM), dtype=N.float)

st = time.time()
for i in range(50):
  x.set(src)

print time.time() - st

st = time.time()
for i in range(50):
  x = gpuarray.to_gpu(src)

print time.time() - st

