import numpy as np 

from theano.misc.pycuda_utils import to_cudandarray, to_gpuarray 
from theano.sandbox.cuda import CudaNdarray
import pycuda 
import pycuda.autoinit
from pycuda.gpuarray import GPUArray 
import scikits.cuda
import scikits.cuda.linalg 
from pycuda.elementwise import ElementwiseKernel

scikits.cuda.linalg.init() 

import scikits.cuda.cublas as cublas 
cublas_handle =  cublas.cublasCreate()


from pycuda.compiler import SourceModule


def max_reduce(mat, x):
  mh, mw = mat.shape
  vh, vw = x.shape
  
  assert(vw == 1 and vh == mh)
  
  mod = SourceModule('''
    __global__
    void row_max_reduce(float* mat, float* vec, int leading) {
    const int INTERNAL_SIZE = 256;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    
    __shared__ float buffer[INTERNAL_SIZE];
    buffer[i] = mat[i + j * leading];
    __syncthreads();

    int index = 1;
    while(threadIdx.x + index * INTERNAL_SIZE < blockDim.x) {
      if (buffer[threadIdx.x] < mat[threadIdx.x + index * INTERNAL_SIZE])
        buffer[threadIdx.x] = mat[threadIdx.x + index * INTERNAL_SIZE];
      index ++;
    }
    __syncthreads();

    int total = blockDim.x;
    while(total > 1) {
      int halfPoint = ((1+total) >> 1);
      if (threadIdx.x < halfPoint)  {
        if(threadIdx.x+halfPoint < total) {
          if(buffer[threadIdx.x] < buffer[threadIdx.x + halfPoint])
            buffer[threadIdx.x] = buffer[threadIdx.x + halfPoint];
        }
      }
      __syncthreads();
      total = ((1+total) >> 1);
    }
    __syncthreads();
    if(threadIdx.x == 0)
      vec[blockIdx.y] = buffer[0];
   }'''
   )
  row_max_reduce = mod.get_function('row_max_reduce')
  grid = (1, mat.shape[0])
  block = (mat.shape[1], 1,  1)

  row_max_reduce(mat.gpudata, x.gpudata,  np.int32(mat.strides[0]/4), block = block, grid= grid)

def find_row_max_id(mat, x):
  mh, mw = mat.shape
  vh, vw = x.shape
  
  assert(vw == 1 and vh == mh)
  
  mod = SourceModule('''
    __global__
    void row_max_id(float* mat, float* vec, int leading) {
    const int INTERNAL_SIZE = 256;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    
    __shared__ float buffer[INTERNAL_SIZE];
    __shared__ int mind[INTERNAL_SIZE];
    buffer[i] = mat[i + j * leading];
    mind[i] = threadIdx.x;
    __syncthreads();

    int index = 1;
    while(threadIdx.x + index * INTERNAL_SIZE < blockDim.x) {
      if (buffer[threadIdx.x] < mat[threadIdx.x + index * INTERNAL_SIZE]) {
        buffer[threadIdx.x] = mat[threadIdx.x + index * INTERNAL_SIZE];
        mind[threadIdx.x] = threadIdx.x + index * INTERNAL_SIZE;
      }
      index ++;
    }
    __syncthreads();

    int total = blockDim.x;
    while(total > 1) {
      int halfPoint = ((1+total) >> 1);
      if (threadIdx.x < halfPoint)  {
        if(threadIdx.x+halfPoint < total) {
          if(buffer[threadIdx.x] < buffer[threadIdx.x + halfPoint]) {
            buffer[threadIdx.x] = buffer[threadIdx.x + halfPoint];
            mind[threadIdx.x] = mind[threadIdx.x + halfPoint];
          }
        }
      }
      __syncthreads();
      total = ((1+total) >> 1);
    }
    __syncthreads();
    if(threadIdx.x == 0)
      vec[blockIdx.y] = mind[0];
   }'''
   )
  row_max_id = mod.get_function('row_max_id')
  grid = (1, mat.shape[0])
  block = (mat.shape[1], 1,  1)

  row_max_id(mat.gpudata, x.gpudata,  np.int32(mat.strides[0]/4), block = block, grid= grid)
  
def add_vec_to_rows(mat, vec, dest = None,  alpha = 1.0, beta = 1.0):
  w, h = mat.shape
  vw, vh = vec.shape

  assert(vw == 1 or vh == 1)
  if vw == 1:
    assert(vh == h)
  else:
    assert(vw == w)
  mod = SourceModule('''
    __global__
    void add_vec_to_rows( float alpha, float* row, float beta, float* mat, float* dst,int leading, int rows, int cols) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int j = blockIdx.y * blockDim.y + threadIdx.y;
      int index = i + j*leading;
      if ( i < cols   &&  j < rows)
        dst[index] = alpha* row[j] + beta * mat[index];
    }'''
    )

  add_func = mod.get_function('add_vec_to_rows')
  if dest == None:
    dest = mat
  block = (32, 32, 1)
  grid = (mat.shape[0] / 32 + 1, mat.shape[1]/32 + 1)
  add_func( np.float(alpha), vec.gpudata,  np.float32(beta), mat.gpudata, dest.gpudata,
      np.int32(mat.strides[0]/4), np.int32(mat.shape[0]), np.int32(mat.shape[1]),  block = block, grid = grid)

def div_vec_to_rows(mat, vec, dest = None,  alpha = 1.0, beta = 1.0):
  w, h = mat.shape
  vw, vh = vec.shape

  assert(vw == 1 or vh == 1)
  if vw == 1:
    assert(vh == h)
  else:
    assert(vw == w)
  mod = SourceModule('''
    __global__
    void div_vec_to_rows(float* row, float* mat, float* dst,int leading, int rows, int cols) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int j = blockIdx.y * blockDim.y + threadIdx.y;
      int index = i + j*leading;
      if ( i < cols   &&  j < rows)
        dst[index] = mat[index] / row[j]
    }'''
    )

  div_func = mod.get_function('div_vec_to_rows')
  if dest == None:
    dest = mat
  block = (32, 32, 1)
  grid = (mat.shape[0] / 32 + 1, mat.shape[1]/32 + 1)
  div_func( vec.gpudata,   mat.gpudata, dest.gpudata,
      np.int32(mat.shape[0]), np.int32(mat.shape[1]),  block = block, grid = grid)




def add_sum_to_vec(vec, mat, alpha = 1.0, beta = 1.0):
  h,w = mat.shape
  vh, vw = vec.shape

  assert(vw == 1 and vh == h or vh == 1 and vw == w)
  mod = SourceModule('''
  __global__ void add_row_sum(float* mat, float alpha, float* vec, float beta, int leading) {
    const int INTERNAL_SIZE = 256;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    
    __shared__ float buffer[INTERNAL_SIZE];
    buffer[i] = mat[i + j * leading];
    __syncthreads();

    int index = 1;
    while(threadIdx.x + index * INTERNAL_SIZE < blockDim.x) {
      buffer[threadIdx.x] += mat[threadIdx.x + index * INTERNAL_SIZE];
      index ++;
    }
    __syncthreads();
  
    int total = INTERNAL_SIZE;
    while(total > 1) {
      int halfPoint = ((1+total) >> 1);
      if (threadIdx.x < halfPoint)  {
        float temp = 0.0;
        if(threadIdx.x+halfPoint < total) {
          temp = buffer[threadIdx.x + halfPoint];
        }
        buffer[threadIdx.x] += temp;
      }
      __syncthreads();
      total = ((1+total) >> 1);
    }
    __syncthreads();

    if(threadIdx.x == 0)
      vec[j]  = alpha* vec[j] + beta * buffer[0];
  }'''
  )

  add_row_sum = mod.get_function('add_row_sum')
  grid = (1, mat.shape[0])
  block = (mat.shape[1], 1,  1)

  add_row_sum(mat.gpudata, np.float32(alpha), vec, np.float32(beta), np.int32(mat.strides[0]/4), block = block, grid= grid)

def same_reduce(target, vec):
  mod = SourceModule('''
    __global__
    void same(float* tgt, float* vec, float* tmp) {
      int i = threadIdx.x;
      if( tgt[i] == vec[i] )
        tmp[i] = 1;
      else
        tmp[i] = 0;
      
    }'''
    )

  block = (target.size, 1, 1)
  grid = (1, 1)
  same_func = mod.get_function('same')
  tmp = gpuarray.zeros_like(target);
  same_func(target, vec, tmp, block = block, grid = grid)
  tmp.shape = (tmp.shape[1], tmp.shape[0])
  res = gpuarray.to_gpu(np.zeros((1,1)).astype(np.float32))
  add_sum_to_vec(res, tmp)
  return int(res.get()[0, 0])

def logreg_cost_reduce(mat, label, cost):
  mh, mw = mat.shape
  vh, vw = label.shape
  assert(vh == 1 and vw == mh)

  mod = SourceModule('''
    __global__
    void log_reg(float* mat, float* label, float* cost, int leading){
      int i = threadIdx.x;
      int idx = i * leading + label[i];
      cost[i] = 0 - __logf(mat[idx]);
    }'''
    )
    
  log_reg_func = mod.get_function('log_reg')
  block = (mh, 1, 1)
  grid = (1, 1)
  log_reg_func(mat, label, cost, np.int32(mat.strides[0]/4), block = block, grid = grid)

def softmax_bprop(mat, label, grad):
  mh, mw = mat.shape
  vh, vw = label.shape

  assert(vh == 1 and vw == mh)
  
  mod = SourceModule(
      '''
      __global__
      void softmax_bprop_grad(float* mat, float* label, float* grad,
        int leading, int rows, int cols){
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;

        int idx= i + j * leading;
        if( i > cols) return;
        if( j > rows) return;

        grad[idx] = mat[idx];
        if(i == label[j])
          grad[idx] = grad[idx] -1;
      }
      '''
      )

  softmax_bprop_func = mod.get_function('softmax_bprop_grad')
  block = (32, 32, 1)
  grid = (mw/32 + 1, mh/ 32 + 1)
  softmax_bprop_func(mat, label, grad, np.int32(mat.strides[0]/4),
      np.int32(mh), np.int32(mw), block = block, grid = grid)

def relu_activate(input, output):
  relu_func = ElementwiseKernel(
      'float *x, float *y',
      'y[i] = fmaxf(x[i], 0.0)',
      'relu_activation')
  relu_func(input, output)


def relu_compute_grad(grad, output, inputGrad):
  relu_grad_func  = ElementwiseKernel(
      'float *x, float* y, float* z',
      'z[i] = x[i] * (y[i] > 0)',
      'relu_gradient'
      )
  relu_grad_func(grad, output, inputGrad)

def gpu_copy_to(x, y):
  pycuda.driver.memecpy_dtod(y.gpudata, x.gpudata, x.nbytes)


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
    host = x.get()
    reshaped = host.reshape(1)
    scalar = reshaped[0]   
    return scalar 
  else:
    assert np.isscalar(x)
    return x

def getidx(x, i):
  if isinstance(x, GPUArray):
    return x[i:i+1].get()[0]
  else:
    return x[i]

def dot(x,y):
  if isinstance(x, CudaNdarray):
    x = to_gpuarray(x)
  if isinstance(y, CudaNdarray):
      y = to_gpuarray(y)
  if isinstance(x, GPUArray):
    assert isinstance(y, GPUArray)
    if x.shape == (1,):
      assert y.shape[0] == 1
      y *= scalar(x)
      return y.ravel() 
    elif y.shape == (1,):
      assert x.shape[1] == 1
      x *= scalar(y) 
      return x.ravel()
    elif len(x.shape) == 1 and len(y.shape) == 1:
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
  
