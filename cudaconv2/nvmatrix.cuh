#ifndef NVMATRIX_H
#define NVMATRIX_H

#include "nvmatrix_kernels.cuh"
#include "pyassert.cuh"
#include <stdio.h>

#define CHECK_CUDA(msg)\
    cudaError_t err = cudaGetLastError();\
    assert(cudaSuccess != err);

struct NVMatrix {
  int _numRows, _numCols, _stride;
  float* _devdata;

  NVMatrix(float* gpuarray, int num_rows, int num_cols, int stride) :
      _devdata(gpuarray), _numRows(num_rows), _numCols(num_cols) {
    _stride = stride;
  }

  int getNumRows() const {
    return _numRows;
  }

  int getNumCols() const {
    return _numCols;
  }
  
  float* getDevData() {
    return _devdata;
  }

  template<class Op>
  void apply(Op op) {
    NVMatrix& target = *this;
    int height = target.getNumRows();
    int width = target.getNumCols();
    dim3 blocks(std::min(NUM_BLOCKS_MAX, DIVUP(width, ELTWISE_THREADS_X)),
        std::min(NUM_BLOCKS_MAX, DIVUP(height, ELTWISE_THREADS_Y)));
    dim3 threads(ELTWISE_THREADS_X, ELTWISE_THREADS_Y);
kEltwiseUnaryOp<Op><<<blocks, threads>>>(getDevData(), target.getDevData(), height, width, getStride(), target.getStride(), op);
            CHECK_CUDA("kEltwiseUnaryOp: Kernel execution failed");
  }

  int getStride() const {
    return _stride;
  }

  int getNumElements() const {
    return getNumRows() * getNumCols();
  }
  bool isTrans() const {
    return false;
  }
  bool isSameDims(NVMatrix& other) const {
    return other.getNumRows() == getNumRows()
        && other.getNumCols() == getNumCols();
  }

  bool isContiguous() const {
    return true;
  }

  void resize(NVMatrix& like) {
    resize(like.getNumRows(), like.getNumCols());
  }

  void resize(int numRows, int numCols) const {
    if (!(_numRows == numRows && _numCols == numCols)) {
      throw Exception(
          StringPrintf("Cannot resize! (%d, %d) -> (%d, %d)", _numRows,
              _numCols, numRows, numCols), __FILE__, __LINE__);
    }
  }
};

#endif // NVMATRIX_H
