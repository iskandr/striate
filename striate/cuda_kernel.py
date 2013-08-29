import init_cuda

from pycuda import gpuarray
from pycuda.compiler import SourceModule
from pycuda.elementwise import ElementwiseKernel
from pycuda.gpuarray import GPUArray
from scikits.cuda import cublas
from time import time
from util import *
import cPickle
import cudaconv2
import numpy as np
import pycuda
import sys
import pycuda.autoinit
#from scikits.cuda import linalg
#linalg.init()

try:
  cublas.cublasInit()
  sgemm = cublas.cublasSgemm
except AttributeError:
  handle = cublas.cublasCreate()
  def sgemm(*args):
    cublas.cublasSgemm(handle, *args)

class CompiledSource(object):
  def __init__(self, src, kernel):
    print >> sys.stderr, 'Compiling...', kernel
    self.module = SourceModule(src)
    self.kernel = self.module.get_function(kernel)

  def __call__(self, *args, **kw):
    self.kernel(*args, **kw)


def I(i): return np.int32(i)
def F(f): return np.float32(f)

INTERNAL_SIZE = 256
_row_max_reduce_ = CompiledSource('''
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
   }''', 'row_max_reduce')


_col_max_reduce_ = CompiledSource('''
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
   }
    ''', 'col_max_reduce')


_find_row_max_id_ = CompiledSource('''
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
   }
    ''', 'row_max_id')


_find_col_max_id_ = CompiledSource('''
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
   }
    ''', 'col_max_id')


_add_vec_to_rows_ = CompiledSource('''
    __global__
    void add_vec_to_rows( float alpha, float* row, float beta, float* mat, float* dst,int leading, int rows, int cols) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int j = blockIdx.y * blockDim.y + threadIdx.y;
      int index = i + j*leading;
      if ( i < cols   &&  j < rows)
        dst[index] = alpha* row[j] + beta * mat[index];
    }
    ''', 'add_vec_to_rows')


_add_vec_to_cols_ = CompiledSource('''
    __global__
    void add_vec_to_cols( float alpha, float* row, float beta, float* mat, float* dst,int leading, int rows, int cols) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int j = blockIdx.y * blockDim.y + threadIdx.y;
      int index = i + j*leading;
      if ( i < cols   &&  j < rows)
        dst[index] = alpha* row[i] + beta * mat[index];
    }
    ''', 'add_vec_to_cols')


_div_vec_to_rows_ = CompiledSource('''
    __global__
    void div_vec_to_rows(float* row, float* mat, float* dst,int leading, int rows, int cols) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int j = blockIdx.y * blockDim.y + threadIdx.y;
      int index = i + j*leading;
      if ( i < cols   &&  j < rows)
        dst[index] = __fdividef(mat[index], row[j]);
    }
    ''', 'div_vec_to_rows')

_div_vec_to_cols_ = CompiledSource('''
    __global__
    void div_vec_to_cols(float* row, float* mat, float* dst,int leading, int rows, int cols) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int j = blockIdx.y * blockDim.y + threadIdx.y;
      int index = i + j*leading;
      if ( i < cols   &&  j < rows)
        dst[index] = __fdividef(mat[index], row[i]);
    }
    ''', 'div_vec_to_cols')

_add_row_sum_to_vec_ = CompiledSource(
  '''__global__ void add_row_sum(float* mat, float alpha, float* vec, float beta, int leading, int
  rows, int cols) {
    const int INTERNAL_SIZE = 256;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ float buffer[INTERNAL_SIZE];
    if(i < cols)
      buffer[threadIdx.x] = mat[i + j * leading];
    __syncthreads();

    int total = INTERNAL_SIZE > cols ? cols : INTERNAL_SIZE;
    #pragma unroll
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
  }''', 'add_row_sum')


_add_col_sum_to_vec_ = CompiledSource('''
  __global__ void add_col_sum(float* mat, float alpha, float* vec, float beta, int leading, int
  rows, int cols) {
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
  }''', 'add_col_sum')

_same_reduce_ = CompiledSource('''
    __global__
    void same(float* tgt, float* vec, float* tmp) {
      int i = threadIdx.x;
      if( tgt[i] == vec[i] )
        tmp[i] = 1;
      else
        tmp[i] = 0;

    }
  ''', 'same')


_logreg_cost_row_reduce_ = CompiledSource('''
    __global__
    void log_reg_row(float* mat, float* label, float* cost, int leading){
      int i = threadIdx.x;
      int idx = i * leading + label[i];
      cost[i] = 0 - __logf(mat[idx]);
    }
    ''', 'log_reg_row')


_logreg_cost_col_reduce_ = CompiledSource('''
    __global__
    void log_reg_col(float* mat, float* label, float* cost, int leading){
      int i = threadIdx.x;
      int idx = i + label[i] * leading;
      cost[i] = 0 - __logf(mat[idx]);
    }
    ''', 'log_reg_col')


_softmax_bprop_ = CompiledSource(
      '''
      __global__
      void softmax_bprop_grad(float* mat, float* label, float* grad, int leading, int rows, int cols){
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;

        int idx= i + j * leading;
        if( i >= cols) return;
        if( j >= rows) return;

        if(j == label[i])
          grad[idx] = 1 - mat[idx];
        else
          grad[idx] = 0 - mat[idx];
      }
      ''', 'softmax_bprop_grad')

_relu_activate_ = CompiledSource('''
  __global__
  void relu_activate(float* input, float* output, float e,  int leading, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i >= cols) return ;
    if(j >= rows) return ;

    int idx = i + j * leading;

    output[idx] = fmaxf(input[idx], e);
  }''', 'relu_activate'
  )


_tanh_activate_ = CompiledSource('''
    __global__
    void tanh_activate(float* input, float *output, float a, float _n2b, int leading, int rows, int cols) {
     int i = blockIdx.x * blockDim.x + threadIdx.x;
      int j = blockIdx.y * blockDim.y + threadIdx.y;

      if(i >= cols) return ;
      if(j >= rows) return ;

      int idx = i + j * leading;

      output[idx] = a * (__fdividef(2.0f, 1.0f + __expf(input[idx]* _n2b)) - 1.0f);
    }''', 'tanh_activate'
    )

_relu_compute_grad_ = CompiledSource('''
  __global__
  void relu_compute_grad(float * grad, float * output, float* outGrad, float e, int leading, int rows, int
  cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i >= cols) return;
    if(j >= rows) return;

    int idx = i + j * leading;
    outGrad[idx] = grad[idx] * (output[idx] > e);
    //grad[idx] = grad[idx] * (output[idx] > e);
  }
  ''', 'relu_compute_grad')

_tanh_compute_grad_ = CompiledSource('''
  __global__
  void tanh_compute_grad(float * grad, float * output, float* outGrad, float a, float _n4ab,  int leading, int rows, int
  cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i >= cols) return;
    if(j >= rows) return;

    int idx = i + j * leading;
    float t = (1.0f - __fdividef(output[idx], a)) / 2.0f;
    outGrad[idx] = grad[idx] *_n4ab * (t * ( t - 1.0f));
    //grad[idx] = grad[idx] * (output[idx] > 0.0f);
  }
  ''', 'tanh_compute_grad')



_transpose_ = CompiledSource('''
  __global__
  void transpose(float * src, float* dst, int sleading, int dleading, int srows, int scols) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if(i >= scols) return ;
    if(j >= srows) return ;

    int sind = i + j * sleading;
    int dind = j + i * dleading;

    dst[dind] = src[sind];
  }''', 'transpose'
  )

_matrix_add_ = CompiledSource('''
  __global__
  void matrix_add(float* src, float* v, float* dest, float alpha, float beta,  int leading, int
  rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i >= cols) return ;
    if(j >= rows) return ;

    int idx = i + j * leading;

    dest[idx] = src[idx] * alpha + v[idx] * beta;
  }''', 'matrix_add'
  )


_gpu_partial_copy_to_ = CompiledSource('''
    __global__
    void gpu_partial_copy_to(float* src, float* dest, int row_from, int row_to, int col_from, int
    col_to, int sleading, int dleading) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int j = blockIdx.y * blockDim.y + threadIdx.y;

      if( i >= col_to - col_from) return;
      if( j >= row_to - row_from) return;

      int sidx = i+col_from  + (j+ row_from) * sleading;
      int didx = i+ j  * dleading;

      dest[didx] = src[sidx];
    }''', 'gpu_partial_copy_to')

_bigger_than_scaler_ = CompiledSource('''
    __global__
    void bigger_than_scaler(float* src, float* dest, float scaler, int rows, int cols, int leading)
    {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int j = blockIdx.y * blockDim.y + threadIdx.y;

      if (i >= cols) return ;
      if (j >= rows) return ;

      int idx = i + j * leading;

      dest[idx] = src[idx] >= scaler ? 1.0 : 0.0;
    }''', 'bigger_than_scaler')

_eltwise_exp_ = CompiledSource('''
    __global__
    void eltwise_exp(float* src, float* dest, int rows, int cols, int leading) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int j = blockIdx.y * blockDim.y + threadIdx.y;

      if( i >= cols ) return ;
      if( j >= rows ) return ;

      int idx = i + j * leading;
      dest[idx] = __expf(src[idx]);
    }''', 'eltwise_exp')

_eltwise_mul_ = CompiledSource('''
    __global__
    void eltwise_mul(float* src, float* right, float* dest, int rows, int cols, int leading) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int j = blockIdx.y * blockDim.y + threadIdx.y;

      if( i >= cols ) return ;
      if( j >= rows ) return ;

      int idx = i + j * leading;
      dest[idx] = src[idx] *right[idx];
    }''', 'eltwise_mul')

def row_max_reduce(x, mat):
  '''
  Return the max of each row to a vec, ONLY work on small matrix
  Small means the column of the matrix is up to 1024
  and the rows, seams like a little big, can be 2048, but the upper bound has  not been tested
  '''
  timer.start()
  mh, mw = mat.shape
  vh, vw = x.shape

  assert(vw == 1 and vh == mh or vh == 1 and vw == mh)

  grid = (1, mh)
  block = (mw, 1, 1)
  leading = mat.strides[0] / 4
  _row_max_reduce_(mat, x, I(leading), I(mh), I(mw), block=block, grid=grid)
  timer.end("row_max_reduce")


def col_max_reduce(x, mat):
  '''
  Return the max of each column to a vec, ONLY work on small matrix
  Small means the row of the matrix is up to 1024
  and the column, seams like a little big, can be 2048, but the upper bound has  not been tested
  '''
  timer.start()
  mh, mw = mat.shape
  vh, vw = x.shape
  assert(vw == 1 and vh == mw or vh == 1 and vw == mw)

  grid = (mw, 1)
  block = (1, mh, 1)
  leading = mat.strides[0] / 4
  _col_max_reduce_(mat, x, I(leading), I(mh), I(mw), block=block, grid=grid)
  timer.end('col_max_reduce')


def find_row_max_id(x, mat):
  '''
  Return the id of max in each row to a vec(0-based), ONLY work on small matrix
  Small means the column of the matrix is up to 1024
  and the rows, seams like a little big, can be 2048, but the upper bound has  not been tested
  '''
  timer.start()
  mh, mw = mat.shape
  vh, vw = x.shape
  assert(vw == 1 and vh == mh or vh == 1 and vw == mh)

  grid = (1, mh)
  block = (mw, 1, 1)
  leading = mat.strides[0] / 4
  _find_row_max_id_(mat, x, I(leading), I(mh), I(mw), block=block, grid=grid)
  timer.end('find_row_max_id')


def find_col_max_id(x, mat):
  '''
  Return the id of max in each column to a vec, ONLY work on small matrix
  Small means the row of the matrix is up to 1024
  and the column, seams like a little big, can be 2048, but the upper bound has  not been tested
  '''
  timer.start()
  mh, mw = mat.shape
  vh, vw = x.shape
  assert(vw == 1 and vh == mw or vh == 1 and vw == mw)

  grid = (mw, 1)
  block = (1, mh, 1)
  leading = mat.strides[0] / 4

  _find_col_max_id_(mat, x, I(leading), I(mh), I(mw), block=block, grid=grid)
  timer.end('find_col_max_id')



def add_vec_to_rows(mat, vec, dest=None, alpha=1.0, beta=1.0):
  '''
  Add the element in vec to every element in mat in corresponding rows
  The function behaves exactly like mat + vec in numpy
  '''
  timer.start()
  mh, mw = mat.shape
  vh, vw = vec.shape

  assert(vw == 1 and vh == mh or vh == 1 and vw == mh)

  if dest is None:
    dest = mat
  block = (32, 32, 1)
  grid = (divup(mw, 32), divup(mh, 32))
  leading = mat.strides[0] / 4
  _add_vec_to_rows_(F(alpha), vec, F(beta), mat, dest, I(leading), I(mh), I(mw), block=block, grid=grid)
  timer.end('add_vec_to_rows')

def add_vec_to_cols(mat, vec, dest=None, alpha=1.0, beta=1.0):
  '''
  Add the element in vec to every element in mat in corresponding cols
  The function behaves exactly like mat + vec in numpy
  '''
  timer.start()
  mh, mw = mat.shape
  vh, vw = vec.shape

  assert(vw == 1 and vh == mw or vh == 1 and vw == mw)

  if not dest:
    dest = mat
  block = (32, 32, 1)
  grid = (divup(mw, 32), divup(mh, 32))
  leading = mat.strides[0] / 4
  _add_vec_to_cols_(F(alpha), vec, F(beta), mat, dest, I(leading), I(mh), I(mw), block=block, grid=grid)
  timer.end('add_vec_to_cols')


def div_vec_to_rows(mat, vec, dest=None):
  '''
  Divide the element in corresponding row of matrix by the element in the vec
  '''
  timer.start()
  mh, mw = mat.shape
  vh, vw = vec.shape

  if not dest:
    dest = mat
  block = (32, 32, 1)
  grid = (divup(mw, 32), divup(mh, 32))
  leading = mat.strides[0] / 4
  _div_vec_to_rows_(vec, mat, dest, I(leading), I(mh), I(mw), block=block, grid=grid)
  timer.end('div_vec_to_rows')



def div_vec_to_cols(mat, vec, dest=None):
  '''
  Divide the element in corresponding column of matrix by the element in the vec
  '''
  timer.start()
  mh, mw = mat.shape
  vh, vw = vec.shape

  if not dest:
    dest = mat
  block = (32, 32, 1)
  grid = (divup(mw , 32), divup(mh, 32))
  leading = mat.strides[0] / 4
  _div_vec_to_cols_(vec, mat, dest, I(leading), I(mh), I(mw), block=block, grid=grid)
  timer.end('div_vec_to_cols')



def add_row_sum_to_vec(vec, mat, alpha=1.0, beta=1.0):
  '''
  This function would sum up the element int a matrix row and store the result to
  the corresponding position of the vec
  Unlike other function that only provide small computation, this function raise the
  upper bound for the number of column to 2^16, actually it could be 2^20
  '''
  timer.start()
  mh, mw = mat.shape
  vh, vw = vec.shape
  assert(vw == 1 and vh == mh or vh == 1 and vw == mh)
  if mw != 1:
    cudaconv2.sum(mat, 1, vec)
  else:
    gpu_partial_copy_to(mat, vec, 0, mh, 0, 1)
  # if mat.shape[1] <= INTERNAL_SIZE:
  #  grid = (1, mh)
  #  block = (mw, 1,  1)
  #  leading = mat.strides[0] /4
  #  _add_row_sum_to_vec_(mat, F(alpha), vec, F(beta),I(leading), I(mh), I(mw), block = block, grid= grid)
  # else:
  #  block = (INTERNAL_SIZE, 1, 1)
  #  grid = (divup(mw, INTERNAL_SIZE), mh)
  #  #tmp  = gpuarray.to_gpu(np.zeros((mh, divup(mw, INTERNAL_SIZE)) ).astype(np.float32))
  #  tmp = gpuarray.zeros((mh, divup(mw, INTERNAL_SIZE)), dtype=np.float32)
  #  #print 'TOGPU', tmp.shape

  #  leading = mat.strides[0]/4
  #  _add_row_sum_to_vec_(mat, F(alpha), tmp, F(beta), I(leading), I(mh),I(mw), block = block, grid = grid)
  #  add_row_sum_to_vec(vec, tmp)
  timer.end('add_row_sum_to_vec')


def add_col_sum_to_vec(vec, mat, alpha=1.0, beta=1.0):
  '''
  This function would sum up the element int a matrix column and store the result to
  the corresponding position of the vec
  ONLY work on small matrix
  Small means the row of the matrix is up to 1024
  and the column, seams like a little big, can be 2048, but the upper bound has  not been tested
  '''
  timer.start()
  mh, mw = mat.shape
  vh, vw = vec.shape
  assert(vw == 1 and vh == mw or vh == 1 and vw == mw)

  cudaconv2.sum(mat, 0, vec)
  #grid = (mw, 1)
  #block = (1, mh, 1)
  #leading = mat.strides[0] / 4
  #_add_col_sum_to_vec_(mat, F(alpha), vec, F(beta), I(leading), I(mh), I(mw), block=block, grid=grid)
  timer.end('add_col_sum_to_vec')


def same_reduce(target, vec):
  '''
  Return the number of same values in the same offset of two vecs
  '''
  timer.start()
  block = (target.size, 1, 1)
  grid = (1, 1)
  tmp = gpuarray.zeros_like(target);
  _same_reduce_(target, vec, tmp, block=block, grid=grid)
  tmp.shape = (1, tmp.size)
  res = gpuarray.to_gpu(np.zeros((1, 1)).astype(np.float32))
  add_row_sum_to_vec(res, tmp)
  timer.end('same_reduce')
  return int(res.get()[0, 0])

def logreg_cost_row_reduce(mat, label, cost):
  timer.start()
  mh, mw = mat.shape
  vh, vw = label.shape
  assert(vh == 1 and vw == mh or vw == 1 and vh == mh)

  block = (mh, 1, 1)
  grid = (1, 1)
  _logreg_cost_row_reduce_(mat, label, cost, np.int32(mat.strides[0] / 4), block=block, grid=grid)
  timer.end('logreg_cost_to_row_reduce')


def logreg_cost_col_reduce(mat, label, cost):
  timer.start()
  mh, mw = mat.shape
  vh, vw = label.shape
  #assert(vh == 1 and vw == mw or vw == 1 and vh == mw)
  if (vh != 1 or vw != mw)  and (vw != 1 or vh != mw):
    log('%s ==> %s', mat.shape, label.shape)
    assert False


  block = (mw, 1, 1)
  grid = (1, 1)
  _logreg_cost_col_reduce_(mat, label, cost, np.int32(mat.strides[0] / 4), block=block, grid=grid)
  timer.end('logreg_cost_to_col_reduce')



def softmax_bprop(mat, label, grad):
  timer.start()
  mh, mw = mat.shape
  vh, vw = label.shape

  assert(vh == 1 and vw == mw or vw == 1 and vh == mw)

  block = (32, 32, 1)
  grid = (divup(mw, 32), divup(mh, 32))
  _softmax_bprop_(mat, label, grad, I(mat.strides[0] / 4), I(mh), I(mw), block=block, grid=grid)
  timer.end('softmax_bprop')

def relu_activate(input, output, e):
  timer.start()
  mh, mw = input.shape

  block = (32, 32, 1)
  grid = (divup(mw, 32), divup(mh, 32))
  leading = input.strides[0] / 4
  _relu_activate_(input, output, F(e), I(leading), I(mh), I(mw), block=block , grid=grid)
  timer.end('relu_activate')


def relu_compute_grad(grad, output, outGrad, e):
  timer.start()
  mh, mw = grad.shape

  block = (32, 32, 1)
  grid = (divup(mw, 32), divup(mh, 32))
  leading = grad.strides[0] / 4
  _relu_compute_grad_(grad, output, outGrad, F(e), I(leading), I(mh), I(mw), block=block, grid=
      grid)
  timer.end('relu_compute_grad')

def tanh_activate(input, output, a, b):
  timer.start()
  mh, mw = input.shape

  block = (32, 32, 1)
  grid = (divup(mw, 32), divup(mh, 32))
  leading = input.strides[0] / 4
  _n2b = -2.0 * b
  _tanh_activate_(input, output, F(a), F(_n2b), I(leading), I(mh), I(mw), block=block , grid=grid)
  timer.end('tanh_activate')


def tanh_compute_grad(grad, output, outGrad, a, b):
  timer.start()
  mh, mw = output.shape

  block = (32, 32, 1)
  grid = (divup(mw, 32), divup(mh, 32))
  leading = output.strides[0] / 4
  _n4ab = -4.0 * a * b
  _tanh_compute_grad_(grad, output, outGrad, F(a), F(_n4ab), I(leading), I(mh), I(mw), block=block , grid=grid)
  timer.end('tanh_compute_grad')



def gpu_copy_to(x, y):
  timer.start()
  pycuda.driver.memcpy_dtod(y.gpudata, x.gpudata, x.nbytes)
  timer.end("gpu_copy_to")

def gpu_partial_copy_to(x, y, row_from, row_to, col_from, col_to):
  timer.start()
  mh, mw = x.shape
  row_to = min(row_to, mh)
  col_to = min(col_to, mw)
  r, c = row_to - row_from, col_to - col_from

  block = (32, 32, 1)
  grid = (divup(c, 32), divup(r, 32))
  sleading, dleading = x.strides[0] / 4, y.strides[0] / 4
  _gpu_partial_copy_to_(x, y, I(row_from), I(row_to), I(col_from), I(col_to), I(sleading), I(dleading), block=block, grid=grid)
  timer.end('gpu_partial_copy_to')

def dot(x, y):
  timer.start()
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
      return scalar(pycuda.gpuarray.dot(x, y))
    else:
      needs_ravel = False
      if len(x.shape) == 1:
        needs_ravel = True
        x = x.reshape((1,) + x.shape)
      if len(y.shape) == 1:
        needs_ravel = True
        y = y.reshape(y.shape + (1,))
      #result = linalg.dot(x, y)
      result = GPUArray((y.shape[1], x.shape[0]), dtype=x.dtype)
      sgemm('t', 't', x.shape[0], y.shape[1], x.shape[1], 1.0,
            x.gpudata, x.shape[1], y.gpudata, y.shape[1], 0.0,
            result.gpudata, result.shape[1])
      result = transpose(result)

      if needs_ravel:
        assert result.shape[1] == 1 or result.shape[0] == 1
        result = result.ravel()
      timer.end('dot')
      return result
  else:
    return np.dot(x, y)

def transpose(mat):
  timer.start()
  mh, mw = mat.shape
  dst = gpuarray.empty((mw, mh), dtype=np.float32)

  block = (32, 32, 1)
  grid = (divup(mw, 32), divup(mh, 32))
  sleading = mat.strides[0] / 4
  dleading = dst.strides[0] / 4
  _transpose_(mat, dst, I(sleading), I(dleading), I(mh), I(mw), block=block, grid=grid)

  timer.end('transpose')
  return dst

def matrix_add(src, v, dest=None, alpha=1.0, beta=1.0):
  sh, sw = src.shape
  vh, vw = v.shape

  #assert sh == vh and sw == vw
  if sh != vh or sw != vw:
    print '(%s, %s) + (%s, %s)' % (sh, sw, vh, vw)
    assert False

  block = (32, 32, 1)
  grid = (divup(sw, 32), divup(sh, 32))
  leading = src.strides[0] / 4
  if dest is None:
    dest = src
  _matrix_add_(src, v, dest, F(alpha), F(beta), I(leading), I(sh), I(sw), block=block , grid=
      grid)


def bigger_than_scaler(src, scaler, dest=None):
  if dest is not None:
    assert dest.shape == src.shape
  else:
    dest = src

  mh, mw = src.shape

  block = (32, 32, 1)
  grid = (divup(mw, 32), divup(mh, 32))
  leading = src.strides[0] / 4
  _bigger_than_scaler_(src, dest, F(scaler), I(mh), I(mw), I(leading), block=block , grid=grid)

def eltwise_exp(src, dest = None):
  if dest is None:
    dest = src
  mh, mw = src.shape

  block = (32, 32, 1)
  grid =  (divup(mw, 32), divup(mh, 32))
  leading = src.strides[0] / 4
  _eltwise_exp_(src, dest, I(mh), I(mw), I(leading), block = block, grid = grid)

def eltwise_mul(src, right, dest = None):
  assert src.shape == right.shape
  if dest is None:
    dest = src
  mh, mw = src.shape

  block = (32, 32, 1)
  grid = (divup(mw, 32), divup(mh, 32))
  leading = src.strides[0] / 4
  _eltwise_mul_(src, right, dest, I(mh), I(mw), I(leading), block = block, grid = grid)
