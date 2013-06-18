#ifndef NVMATRIX_H
#define NVMATRIX_H

#include "nvmatrix_kernels.cuh"
#include "nvmatrix_operators.cuh"
#include "pyassert.cuh"
#include <stdio.h>

#define CHECK_CUDA(msg)\
  cudaError_t err = cudaGetLastError();\
assert(cudaSuccess != err);

struct NVMatrix {
  int _numRows, _numCols, _stride;
  float* _devData;

  NVMatrix(float* gpuarray, int num_rows, int num_cols, int stride) :
    _devData(gpuarray), _numRows(num_rows), _numCols(num_cols) {
      _stride = stride;
    }

  int getNumRows() const {
    return _numRows;
  }

  int getNumCols() const {
    return _numCols;
  }

  float* getDevData() {
    return _devData;
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

  int getLeadingDim() {
    return _numCols;
  }

  int getFollowingDim() {
    return _numRows;
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
  template <class Op> 
    void applyBinaryV(Op op, NVMatrix& vec, NVMatrix& target) {
      assert(&target != &vec); // for now
      assert(vec.getNumRows() == 1 || vec.getNumCols() == 1);
      assert(vec.getNumRows() == _numRows || vec.getNumCols() == _numCols);
      assert(vec.isContiguous());

      target.resize(*this); // target must be same orientation as me for now

      int width = getLeadingDim(); //_isTrans ? _numRows : _numCols;
      int height = getFollowingDim(); //_isTrans ? _numCols : _numRows;
      dim3 threads(ADD_VEC_THREADS_X, ADD_VEC_THREADS_Y);
      dim3 blocks(MIN(NUM_BLOCKS_MAX, DIVUP(width, ADD_VEC_THREADS_X)), MIN(NUM_BLOCKS_MAX, DIVUP(height, ADD_VEC_THREADS_Y)));
      if (vec.getNumRows() == _numRows && !isTrans() || vec.getNumCols() == _numCols && isTrans()) {
        fprintf(stderr, "Col Vector\n");
        kColVectorOp<Op><<<blocks,threads>>>(_devData, vec._devData, target._devData, width, height, getStride(), target.getStride(), op);
      } else {
        kRowVectorOp<Op><<<blocks,threads>>>(_devData, vec._devData, target._devData, width, height, getStride(), target.getStride(), op);
      }
      CHECK_CUDA("Kernel execution failed");
      //    cudaThreadSynchronize();
    }

  void addVector(NVMatrix& vec) {
    applyBinaryV(NVMatrixBinaryOps::WeightedAdd(1, 1.0), vec, *this);
  }
};

#endif // NVMATRIX_H
