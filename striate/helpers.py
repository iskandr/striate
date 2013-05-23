import numpy as np 
import pycuda 
import pycuda.autoinit
from pycuda.gpuarray import GPUArray 

rng = numpy.random.RandomState(23455)

def gpu_copy(x):
  y = pycuda.gpuarray.empty_like(x)
  pycuda.driver.memcpy_dtod(y.gpudata, x.gpudata, x.nbytes)
  return y

def mean(gpu_arrays):
  acc = gpu_copy(gpu_arrays[0])
  n = len(gpu_arrays)
  if n == 1:
    return acc
  recip = 1.0 / n
  acc *= recip 
  for x in gpu_arrays[1:]:
    acc = acc.mul_add(1.0, x, recip)
  return acc 

def weighted_mean(gpu_arrays, weights):
  acc = gpu_arrays[0]
  weights /= np.sum(weights)
  acc *= weights[0]
  for x, w in zip(gpu_arrays[1:], weights[1:]):
    acc.mul_add(1.0, x, w) 
  return acc 

def memcpy(dest, src):
  assert len(dest.shape) == 1
  assert len(src.shape) == 1
  assert dest.nbytes == src.nbytes
  pycuda.driver.memcpy_dtod(dest.gpudata, src.gpudata, src.nbytes)

def concat(xs):
  if isinstance(xs[0], GPUArray):
    # Stupid GPUArray doesn't support strides
    # so have to go to stupid lengths to 
    # stack some gosh-darned vectors 
    elt_shape = xs[0].shape
    
    assert len(elt_shape) == 1
    elt_dtype = xs[0].dtype
    assert all(x.shape == elt_shape for x in xs)
    assert all(x.dtype == elt_dtype for x in xs)
    nrows = len(xs)
    row_nelts = elt_shape[0]
    total_nelts = nrows * elt_shape[0]
    result = pycuda.gpuarray.empty(shape=(total_nelts,), dtype=elt_dtype)
    for (i,x) in enumerate(xs):
      output_slice = result[i*row_nelts:(i+1)*row_nelts]
      memcpy(output_slice, x)
    final_shape = (nrows,) + elt_shape 
    return result.reshape(final_shape)
  else:
    return np.array(xs)

def scalar(x):
  if isinstance(x, GPUArray):
    return x.get().reshape(1)[0]
  else:
    assert np.isscalar(x)
    return x

def getidx(x, i):
  if isinstance(x, GPUArray):
    return x[i:i+1].get()[0]
  else:
    return x[i]

def dot(x,y):
  if isinstance(x, GPUArray):
    assert isinstance(y, GPUArray)
    if len(x.shape) == 1 and len(y.shape) == 1:
      return scalar(pycuda.gpuarray.dot(x,y))
    else:
      if len(x.shape) == 1:
        needs_ravel = True
        x = x.reshape((1,) + x.shape)
      if len(y.shape) == 1:
        needs_ravel = True
        y = y.reshape(y.shape + (1,))
      
      result = scikits.cuda.linalg.dot(x,y)
      if needs_ravel:
        assert result.shape[1] == 1 or result.shape[0] == 1
        result = result.ravel()
      return result 
  else:
    return np.dot(x,y)

def norm(x):
  if isinstance(x, GPUArray):
    return cublas.cublasSnrm2(cublas_handle, x.size, x.gpudata, 1)
  else:
    return np.linalg.norm(x) 

def diag_dot(diag, X):
  """
  Reweight the rows of X, as if multiplying on the left by a diagonal
  """
  d = len(diag)
  assert d <= X.shape[0]
  result = X[:d, :]    
  for row_idx in xrange(d): 
    row_slice = result[row_idx, :]
    row_slice *= getidx(diag, row_idx)
  return result 

def vecmin(x):
  if isinstance(x, GPUArray):
    return scalar(pycuda.gpuarray.min(x))
  else:
    return np.min(x)

def vecmax(x):
  if isinstance(x, GPUArray):
    return scalar(pycuda.gpuarray.max(x))
  else:
    return np.max(x)

def vecsum(x):
  if isinstance(x, GPUArray):
    return scalar(pycuda.gpuarray.sum(x))
  else:
    return np.sum(x)

def svd(X):
  if isinstance(X, GPUArray):
    return scikits.cuda.linalg.svd(X, jobu = 'S', jobvt='S')
  else:
    return np.linalg.svd(X, full_matrices=False)

def argmax(x):
  if isinstance(x, GPUArray):
    x = x.get()
  return np.argmax(x)

def transpose(X):
  if isinstance(X, GPUArray):
    return scikits.cuda.linalg.transpose(X)
  else:
    return X.T 

def take_rows(X, k):
  if isinstance(X, GPUArray):
    nrows, ncols = X.shape
    X_flat = X.ravel()
    assert k <= nrows
    dtype = X.dtype 
    result = pycuda.gpuarray.empty(shape = (k * ncols), dtype=dtype)
    input_slice = X_flat[:k*ncols]
    memcpy(result, input_slice)
    return result.reshape( (k, ncols) )
  else:
    return X[:k]
 
def take_cols(X, k):
  if isinstance(X, GPUArray):
    X = X.get()
    X = X[:, :k]
    return pycuda.gpuarray.to_gpu(X)
  else:
    return X[:, :k]
  
