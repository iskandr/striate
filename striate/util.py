import numpy as np 
import pycuda 
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.gpuarray import GPUArray
import scikits.cuda
import scikits.cuda.linalg 
from pycuda.elementwise import ElementwiseKernel

scikits.cuda.linalg.init() 

import scikits.cuda.cublas as cublas 
cublas_handle =  cublas.cublasCreate()


from pycuda.compiler import SourceModule
INTERNAL_SIZE = 256 
def I(i): return np.int32(i)
def F(f): return np.float32(f)
def NVBLOCK(x, base):
  if x / base * base == x:
    return x / base
  else:
    return x /base + 1

def row_max_reduce(x, mat):
  '''
  Return the max of each row to a vec, ONLY work on small matrix
  Small means the column of the matrix is up to 1024
  and the rows, seams like a little big, can be 2048, but the upper bound has  not been tested
  '''
  mh, mw = mat.shape
  vh, vw = x.shape
  
  assert(vw == 1 and vh == mh or vh == 1 and vw == mh)
  
  mod = SourceModule('''
    __global__
    void row_max_reduce(float* mat, float* vec, int leading, int rows, int cols) {
    const int INTERNAL_SIZE = 256;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    
    __shared__ float buffer[INTERNAL_SIZE];
    if(i < cols && i < INTERNAL_SIZE)
      buffer[i] = mat[i + j * leading];
    __syncthreads();

    int index = 1;
    if(cols > INTERNAL_SIZE) {
      if(threadIdx.x < INTERNAL_SIZE ) {
        int forwardInd = threadIdx.x + index * INTERNAL_SIZE;
        while(forwardInd < cols) {
          if (buffer[threadIdx.x] < mat[forwardInd + j* leading])
            buffer[threadIdx.x] = mat[forwardInd + j * leading];
          index ++;
          forwardInd = threadIdx.x + index * INTERNAL_SIZE;
        }
      }
    }
    __syncthreads();

    int total = INTERNAL_SIZE > cols? cols : INTERNAL_SIZE;
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
  row_max_reduce_func = mod.get_function('row_max_reduce')
  grid = (1,mh)
  block = (mw, 1,  1)
  leading = mat.strides[0]/4
  row_max_reduce_func(mat, x, I(leading), I(mh), I(mw), block = block, grid= grid)


def col_max_reduce(x, mat):
  '''
  Return the max of each column to a vec, ONLY work on small matrix
  Small means the row of the matrix is up to 1024
  and the column, seams like a little big, can be 2048, but the upper bound has  not been tested
  '''
  mh, mw = mat.shape
  vh, vw = x.shape
  assert(vw == 1 and vh == mw or vh == 1 and vw == mw)
  
  mod = SourceModule('''
    __global__
    void col_max_reduce(float* mat, float* vec, int leading, int rows, int cols) {
    const int INTERNAL_SIZE = 256;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    
    __shared__ float buffer[INTERNAL_SIZE];
    if(j < INTERNAL_SIZE && j < rows)
      buffer[j] = mat[i + j * leading];
    __syncthreads();

    int index = 1;
    if(rows > INTERNAL_SIZE) {
      if(threadIdx.y < INTERNAL_SIZE) {
        int forwardInd = threadIdx.y + index * INTERNAL_SIZE;
        while(forwardInd < rows) {
          if (buffer[threadIdx.y] < mat[i +forwardInd * leading])
            buffer[threadIdx.y] = mat[i  + forwardInd * leading];
          index ++;
          forwardInd = threadIdx.y + index * INTERNAL_SIZE;
        }
      }
    }
    __syncthreads();

    int total = INTERNAL_SIZE > rows ? rows : INTERNAL_SIZE;
    while(total > 1) {
      int halfPoint = ((1+total) >> 1);
      if (threadIdx.y < halfPoint)  {
        if(threadIdx.y+halfPoint < total) {
          if(buffer[threadIdx.y] < buffer[threadIdx.y + halfPoint])
            buffer[threadIdx.y] = buffer[threadIdx.y + halfPoint];
        }
      }
      __syncthreads();
      total = ((1+total) >> 1);
    }
    __syncthreads();
    if(threadIdx.y == 0)
      vec[i] = buffer[0];
   }'''
   )
  col_max_reduce_func = mod.get_function('col_max_reduce')
  grid = (mw, 1)
  block = (1, mh,   1)
  leading = mat.strides[0]/4
  col_max_reduce_func(mat, x, I(leading), I(mh), I(mw), block = block, grid= grid)


def find_row_max_id(x, mat):
  '''
  Return the id of max in each row to a vec(0-based), ONLY work on small matrix
  Small means the column of the matrix is up to 1024
  and the rows, seams like a little big, can be 2048, but the upper bound has  not been tested
  '''
  mh, mw = mat.shape
  vh, vw = x.shape
  assert(vw == 1 and vh == mh or vh == 1 and vw == mh)
  
  mod = SourceModule('''
    __global__
    void row_max_id(float* mat, float* vec, int leading, int rows, int cols) {
    const int INTERNAL_SIZE = 256;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    
    __shared__ float buffer[INTERNAL_SIZE];
    __shared__ int mind[INTERNAL_SIZE];
    if(i < INTERNAL_SIZE && i < cols){
      buffer[i] = mat[i + j * leading];
      mind[i] = threadIdx.x;
    }
    __syncthreads();

    int index = 1;
    if(cols > INTERNAL_SIZE)  {
      if(threadIdx.x < INTERNAL_SIZE) {
        int forwardInd = threadIdx.x + index * INTERNAL_SIZE;
        while(forwardInd < cols)  {
          if (buffer[threadIdx.x] < mat[forwardInd + j * leading]) {
            buffer[threadIdx.x] = mat[forwardInd + j * leading];
            mind[threadIdx.x] = forwardInd; 
          }
          index ++;
          forwardInd = threadIdx.x + index * INTERNAL_SIZE;
        }
      }
    }
    __syncthreads();

    int total = INTERNAL_SIZE > cols ? cols : INTERNAL_SIZE;
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
  grid = (1, mh)
  block = (mw, 1,  1)
  leading = mat.strides[0]/4
  row_max_id(mat, x, I(leading), I(mh), I(mw), block = block, grid= grid)


def find_col_max_id(x, mat):
  '''
  Return the id of max in each column to a vec, ONLY work on small matrix
  Small means the row of the matrix is up to 1024
  and the column, seams like a little big, can be 2048, but the upper bound has  not been tested
  '''
  mh, mw = mat.shape
  vh, vw = x.shape
  assert(vw == 1 and vh == mw or vh == 1 and vw == mw)
  
  mod = SourceModule('''
    __global__
    void col_max_id(float* mat, float* vec, int leading, int rows, int cols) {
    const int INTERNAL_SIZE = 256;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    
    __shared__ float buffer[INTERNAL_SIZE];
    __shared__ int mind[INTERNAL_SIZE];
    if( j < INTERNAL_SIZE && j < rows){
      buffer[j] = mat[i + j * leading];
      mind[j] = threadIdx.y;
     }
    __syncthreads();

    int index = 1;
    if(rows > INTERNAL_SIZE) {
      if(threadIdx.y < INTERNAL_SIZE ){
        int forwardInd = threadIdx.y + index * INTERNAL_SIZE; 
        while(forwardInd < rows) {
          if (buffer[threadIdx.y] < mat[i + forwardInd * leading]) {
            buffer[threadIdx.y] = mat[i + forwardInd * leading];
            mind[threadIdx.y] = forwardInd; 
          }
          index ++;
          forwardInd = threadIdx.y + index * INTERNAL_SIZE; 
        }
      }
    }
    __syncthreads();

    int total = INTERNAL_SIZE > rows ? rows : INTERNAL_SIZE;
    while(total > 1) {
      int halfPoint = ((1+total) >> 1);
      if (threadIdx.y < halfPoint)  {
        if(threadIdx.y+halfPoint < total) {
          if(buffer[threadIdx.y] < buffer[threadIdx.y  + halfPoint]) {
            buffer[threadIdx.y] = buffer[threadIdx.y + halfPoint];
            mind[threadIdx.y] = mind[threadIdx.y + halfPoint];
          }
        }
      }
      __syncthreads();
      total = ((1+total) >> 1);
    }
    __syncthreads();
    if(threadIdx.y == 0)
      vec[i] = mind[0];
   }'''
   )
  col_max_id = mod.get_function('col_max_id')
  grid = (mw, 1)
  block = (1, mh, 1)
  leading = mat.strides[0]/4

  col_max_id(mat, x, I(leading), I(mh), I(mw), block = block, grid= grid)



def add_vec_to_rows(mat, vec, dest = None,  alpha = 1.0, beta = 1.0):
  '''
  Add the element in vec to every element in mat in corresponding rows
  The function behaves exactly like mat + vec in numpy
  '''
  mh, mw = mat.shape
  vh, vw = vec.shape

  assert(vw == 1 and vh == mh or vh == 1 and vw == mh)
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
  if not dest:
    dest = mat
  block = (32, 32, 1)
  grid = (NVBLOCK(mw, 32), NVBLOCK(mh, 32))
  leading = mat.strides[0]/4
  add_func(F(alpha), vec, F(beta), mat, dest, I(leading), I(mh), I(mw), block = block, grid = grid)

def add_vec_to_cols(mat, vec, dest = None,  alpha = 1.0, beta = 1.0):
  '''
  Add the element in vec to every element in mat in corresponding cols
  The function behaves exactly like mat + vec in numpy
  '''
  mh, mw = mat.shape
  vh, vw = vec.shape

  assert(vw == 1 and vh == mw or vh == 1 and vw == mw)
  mod = SourceModule('''
    __global__
    void add_vec_to_cols( float alpha, float* row, float beta, float* mat, float* dst,int leading, int rows, int cols) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int j = blockIdx.y * blockDim.y + threadIdx.y;
      int index = i + j*leading;
      if ( i < cols   &&  j < rows)
        dst[index] = alpha* row[i] + beta * mat[index];
    }'''
    )

  add_func = mod.get_function('add_vec_to_cols')
  if not dest:
    dest = mat
  block = (32, 32, 1)
  grid = (NVBLOCK(mw, 32), NVBLOCK(mh, 32))
  leading = mat.strides[0] / 4
  add_func(F(alpha), vec,  F(beta), mat, dest, I(leading), I(mh), I(mw),  block = block, grid = grid)


def div_vec_to_rows(mat, vec, dest = None):
  '''
  Divide the element in corresponding row of matrix by the element in the vec
  '''
  mh, mw = mat.shape
  vh, vw = vec.shape

  assert(vw == 1 and vh == mh or vh == 1 and vw == mh)
  mod = SourceModule('''
    __global__
    void div_vec_to_rows(float* row, float* mat, float* dst,int leading, int rows, int cols) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int j = blockIdx.y * blockDim.y + threadIdx.y;
      int index = i + j*leading;
      if ( i < cols   &&  j < rows)
        dst[index] = mat[index] / row[j];
    }'''
    )

  div_func = mod.get_function('div_vec_to_rows')
  if not dest:
    dest = mat
  block = (32, 32, 1)
  grid = (NVBLOCK(mw, 32), NVBLOCK(mh, 32))
  leading = mat.strides[0] /4
  div_func( vec,  mat, dest, I(leading),I(mh), I(mw), block = block, grid = grid)



def div_vec_to_cols(mat, vec, dest = None):
  '''
  Divide the element in corresponding column of matrix by the element in the vec
  '''
  mh, mw = mat.shape
  vh, vw = vec.shape

  assert(vw == 1 and vh == mw or vh == 1 and vw == mw)
  mod = SourceModule('''
    __global__
    void div_vec_to_cols(float* row, float* mat, float* dst,int leading, int rows, int cols) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int j = blockIdx.y * blockDim.y + threadIdx.y;
      int index = i + j*leading;
      if ( i < cols   &&  j < rows)
        dst[index] = mat[index] / row[i];
    }'''
    )

  div_func = mod.get_function('div_vec_to_cols')
  if not dest:
    dest = mat
  block = (32, 32, 1)
  grid = (NVBLOCK(mw , 32), NVBLOCK(mh, 32))
  leading = mat.strides[0] /4
  div_func(vec, mat, dest, I(leading), I(mh), I(mw), block = block, grid = grid)



def add_row_sum_to_vec(vec, mat, alpha = 1.0, beta = 1.0):
  '''
  This function would sum up the element int a matrix row and store the result to
  the corresponding position of the vec
  Unlike other function that only provide small computation, this function raise the
  upper bound for the number of column to 2^16, actually it could be 2^20
  '''
  mh, mw = mat.shape
  vh, vw = vec.shape
  assert(vw == 1 and vh == mh or vh == 1 and vw == mh)

  mod = SourceModule('''
  __global__ void add_row_sum(float* mat, float alpha, float* vec, float beta, int leading, int
  rows, int cols) {
    const int INTERNAL_SIZE = 256;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    
    __shared__ float buffer[INTERNAL_SIZE];
    if(i < cols)  
      buffer[threadIdx.x] = mat[i + j * leading];
    __syncthreads();

    int total = INTERNAL_SIZE > cols ? cols : INTERNAL_SIZE;
    while(total > 1) {
      int halfPoint = ((1+total) >> 1);
      if (threadIdx.x < halfPoint && i < cols)  {
        float temp = 0.0;
        if(threadIdx.x+halfPoint < total && i + halfPoint < cols) {
          temp = buffer[threadIdx.x + halfPoint];
        }
        buffer[threadIdx.x] += temp;
      }
      __syncthreads();
      total = ((1+total) >> 1);
    }
    __syncthreads();

    if(threadIdx.x == 0)
      vec[blockIdx.y * gridDim.x + blockIdx.x]  = alpha* vec[blockIdx.y * gridDim.x + blockIdx.x] + beta * buffer[0];
      //vec[j] = alpha*vec[j] + beta * buffer[0];
  }'''
  )

  add_row_sum = mod.get_function('add_row_sum')
  if mat.shape[1] <= INTERNAL_SIZE:
    grid = (1, mh)
    block = (mw, 1,  1)
    leading = mat.strides[0] /4
    add_row_sum(mat, F(alpha), vec, F(beta),I(leading), I(mh), I(mw), block = block, grid= grid)
  else:
    block = (INTERNAL_SIZE, 1, 1)
    grid = (NVBLOCK(mw, INTERNAL_SIZE), mh)
    tmp  = gpuarray.to_gpu(np.zeros((mh, NVBLOCK(mw, INTERNAL_SIZE)) ).astype(np.float32))
    leading = mat.strides[0]/4
    add_row_sum(mat, F(alpha), tmp, F(beta), I(leading), I(mh),I(mw), block = block, grid = grid) 
    add_row_sum_to_vec(vec, tmp)


def add_col_sum_to_vec(vec, mat, alpha = 1.0, beta = 1.0):
  '''
  This function would sum up the element int a matrix column and store the result to
  the corresponding position of the vec
  ONLY work on small matrix
  Small means the row of the matrix is up to 1024
  and the column, seams like a little big, can be 2048, but the upper bound has  not been tested
  '''
  mh, mw = mat.shape
  vh, vw = vec.shape
  assert(vw == 1 and vh == mw or vh == 1 and vw == mw)

  mod = SourceModule('''
  __global__ void add_col_sum(float* mat, float alpha, float* vec, float beta, int leading, int
  rows, int cols) {
  /*
    vec[blockIdx.x] = 0;
    for (int i = 0; i < rows; ++i) {
      vec[blockIdx.x] += mat[cols * i + blockIdx.x];
    }
    return;
*/
    const int INTERNAL_SIZE = 256;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    
    __shared__ float buffer[INTERNAL_SIZE];
    if(j < INTERNAL_SIZE && j < rows)
      buffer[j] = mat[i + j * cols];

    __syncthreads();

    int index = 1;
    if(rows > INTERNAL_SIZE) {
      if(threadIdx.y < INTERNAL_SIZE) {
        int forwardInd = threadIdx.y + index * INTERNAL_SIZE;
        while( forwardInd < rows) {
          buffer[threadIdx.y] += mat[i  + forwardInd * leading];
          index ++;
          forwardInd = threadIdx.y + index * INTERNAL_SIZE;
        }
      }
    }
    __syncthreads();
  
    int total = INTERNAL_SIZE > rows ? rows : INTERNAL_SIZE;
    while(total > 1) {
      int halfPoint = ((1+total) >> 1);
      if (threadIdx.y < halfPoint)  {
        float temp = 0.0;
        if(threadIdx.y+halfPoint < total) {
          temp = buffer[threadIdx.y + halfPoint];
        }
        buffer[threadIdx.y] += temp;
      }
      __syncthreads();
      total = ((1+total) >> 1);
    }
    __syncthreads();

    if(threadIdx.y == 0)
      vec[i]  = alpha* vec[i] + beta * buffer[0];
  }'''
  )

  add_col_sum_func = mod.get_function('add_col_sum')
  #block = (1, 1, 1)
  #grid = (mat.shape[0], 1, 1)

  grid = (mw, 1)
  block = (1, mh, 1)
  leading = mat.strides[0] / 4
  add_col_sum_func(mat, F(alpha), vec, F(beta), I(leading), I(mh), I(mw), block = block, grid= grid)


def same_reduce(target, vec):
  '''
  Return the number of same values in the same offset of two vecs
  '''
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
  tmp.shape = (1, tmp.size)
  res = gpuarray.to_gpu(np.zeros((1,1)).astype(np.float32))
  add_row_sum_to_vec(res, tmp)
  return int(res.get()[0, 0])

def logreg_cost_row_reduce(mat, label, cost):
  
  mh, mw = mat.shape
  vh, vw = label.shape
  assert(vh == 1 and vw == mh or vw == 1 and vh == mh)

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


def logreg_cost_col_reduce(mat, label, cost):
  mh, mw = mat.shape
  vh, vw = label.shape
  assert(vh == 1 and vw == mw or vw == 1 and vh == mw)

  mod = SourceModule('''
    __global__
    void log_reg(float* mat, float* label, float* cost, int leading){
      int i = threadIdx.x;
      int idx = i + label[i] * leading;
      cost[i] = 0 - __logf(mat[idx]);
    }'''
    )
    
  log_reg_func = mod.get_function('log_reg')
  block = (mw,1,1)
  grid = (1, 1)
  log_reg_func(mat, label, cost, np.int32(mat.strides[0]/4), block = block, grid = grid)



def softmax_bprop(mat, label, grad):
  mh, mw = mat.shape
  vh, vw = label.shape

  assert(vh == 1 and vw == mw or vw == 1 and vh  == mw)
  
  mod = SourceModule(
      '''
      __global__
      void softmax_bprop_grad(float* mat, float* label, float* grad, int leading, int rows, int cols){
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
  grid = (NVBLOCK(mw, 32), NVBLOCK(mh, 32))
  softmax_bprop_func(mat, label, grad, I(mat.strides[0]/4), I(mh), I(mw), block = block, grid = grid)

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
  pycuda.driver.memcpy_dtod(y.gpudata, x.gpudata, x.nbytes)

def dot(x,y):
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
      needs_ravel = False
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

def transpose(X):
  if isinstance(X, GPUArray):
    return scikits.cuda.linalg.transpose(X)
  else:
    return X.T 

