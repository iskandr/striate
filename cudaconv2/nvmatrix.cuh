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

  template<class Agg, class BinaryOp>
  void _aggregate(int axis, NVMatrix& target, Agg agg, BinaryOp op) {
    assert(axis == 0 || axis == 1);
    assert(isContiguous() && target.isContiguous());
    assert(&target != this);
    int width = isTrans() ? _numRows : _numCols;
    int height = isTrans() ? _numCols : _numRows;

    assert(width > 0);
    assert(height > 0);
    if (axis == 0 && !isTrans() || axis == 1 && isTrans()) { //col sum
      target.resize(!isTrans() ? 1 : _numRows, !isTrans() ? _numCols : 1);
      int numBlocks = DIVUP(width, NUM_SUM_COLS_THREADS_PER_BLOCK);
      assert(numBlocks * NUM_SUM_COLS_THREADS_PER_BLOCK >= width);
      assert(numBlocks < NUM_BLOCKS_MAX);
      kDumbAggCols<Agg, BinaryOp><<<numBlocks,NUM_SUM_COLS_THREADS_PER_BLOCK>>>(_devData, target._devData, width, height, agg, op);
      cutilCheckMsg("kDumbAggCols: Kernel execution failed");
    } else { // row sum
      target.resize(isTrans() ? 1 : _numRows, isTrans() ? _numCols : 1);
      if (width > 1) {
        if (height >= 16384) { // linear aggregation
          int numBlocksX = 1;
          int numBlocksY = DIVUP(height, AGG_SHORT_ROWS_THREADS_Y*AGG_SHORT_ROWS_LOOPS_Y);
          int numThreadsX = width <= 4 ? 4 : width <= 8 ? 8 : width <= 12 ? 12 :
            width <= 16 ? 16 : AGG_SHORT_ROWS_THREADS_X;
          int numThreadsY = AGG_SHORT_ROWS_THREADS_Y;
          while (numBlocksY > NUM_BLOCKS_MAX) {
            numBlocksY = DIVUP(numBlocksY,2);
            numBlocksX *= 2;
          }
          dim3 grid(numBlocksX, numBlocksY), threads(numThreadsX, numThreadsY);
          if (width <= 16) {
            if (width <= 4) {
              kAggShortRows<Agg, BinaryOp, 1, 4><<<grid, threads>>>(_devData, target._devData,width, height, agg, op);
            } else if (width <= 8) {
              kAggShortRows<Agg, BinaryOp, 1, 8><<<grid, threads>>>(_devData, target._devData,width, height, agg, op);
            } else if (width <= 12) {
              kAggShortRows<Agg, BinaryOp, 1, 12><<<grid, threads>>>(_devData, target._devData,width, height, agg, op);
            } else {
              kAggShortRows<Agg, BinaryOp, 1, 16><<<grid, threads>>>(_devData, target._devData,width, height, agg, op);
            }
          } else if (width <= 32) {
            kAggShortRows<Agg, BinaryOp, 2, AGG_SHORT_ROWS_THREADS_X><<<grid, threads>>>(_devData, target._devData,width, height, agg, op);
          } else if (width <= 48) {
            kAggShortRows<Agg, BinaryOp, 3, AGG_SHORT_ROWS_THREADS_X><<<grid, threads>>>(_devData, target._devData,width, height, agg, op);
          } else if (width <= 64) {
            kAggShortRows<Agg, BinaryOp, 4, AGG_SHORT_ROWS_THREADS_X><<<grid, threads>>>(_devData, target._devData,width, height, agg, op);
          } else {
            kAggShortRows2<Agg, BinaryOp><<<grid, threads>>>(_devData, target._devData,width, height, agg, op);
          }
        } else {
          if (width >= 512) {
            dim3 threads(AWR_NUM_THREADS);
            dim3 blocks(1, std::min(1024, height));
            kAggRows_wholerow_nosync<<<blocks, threads>>>(_devData, target._devData, width, height, agg, op);
            //                    dim3 threads(AWR_NUM_THREADS);
            //                    dim3 blocks(1, std::min(1024, height));
            //                    kAggRows_wholerow<<<blocks, threads>>>(_devData, target._devData, width, height, agg, op);

          } else {
            //                    dim3 threads(AWR_NUM_THREADS);
            //                    dim3 blocks(1, std::min(1024, height));
            //                    kAggRows_wholerow<<<blocks, threads>>>(_devData, target._devData, width, height, agg, op);
            NVMatrix *prevSum = this;
            while (prevSum->getLeadingDim() > 1) {
              int numThreadsX = width <= 64 ? 32 : (width <= 128 ? 64 : (width <= 256 ? 128 : (width <= 512 ? 256 : 512)));
              int numThreadsY = 1;
              int numBlocksX = DIVUP(width, 2*numThreadsX);
              int numBlocksY = std::min(height, NUM_BLOCKS_MAX);
              assert(target.getFollowingDim() == height && target.getLeadingDim() == numBlocksX);
              NVMatrix *nvSumAccum = &target;

              dim3 grid(numBlocksX, numBlocksY), threads(numThreadsX, numThreadsY);
              assert(numBlocksX <= NUM_BLOCKS_MAX);
              assert(numBlocksY <= NUM_BLOCKS_MAX);

              if (width <= 64) {
                kAggRows<Agg, BinaryOp, 32><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                    width, height, nvSumAccum->getLeadingDim(), agg, op);
              } else if (width <= 128) {
                kAggRows<Agg, BinaryOp, 64><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                    width, height, nvSumAccum->getLeadingDim(), agg, op);
              } else if (width <= 256) {
                kAggRows<Agg, BinaryOp, 128><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                    width, height, nvSumAccum->getLeadingDim(), agg, op);
              } else if (width <= 512) {
                kAggRows<Agg, BinaryOp, 256><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                    width, height, nvSumAccum->getLeadingDim(), agg, op);
              } else {
                kAggRows<Agg, BinaryOp, 512><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                    width, height, nvSumAccum->getLeadingDim(), agg, op);
              }
              cutilCheckMsg("agg rows: Kernel execution failed");
              cudaThreadSynchronize();
              width = numBlocksX; // only true in reduction agg, but for linear agg this doesn't matter anyway

              if (prevSum != this) {
                delete prevSum;
              }
              prevSum = nvSumAccum;
            }
          }
        }
      } else {
        assert(0);
        // copy(target);
      }
    }
  }


  template <class Op> 
    void applyBinaryV(Op op, NVMatrix& vec, NVMatrix& target) {
      assert(&target != &vec); // for now
      assert(vec.getNumRows() == 1 || vec.getNumCols() == 1);
      assert(vec.getNumRows() == _numRows || vec.getNumCols() == _numCols);
      assert(vec.isContiguous());

      target.resize(*this); // target must be same orientation as me for now

      int width = getLeadingDim(); //isTrans() ? _numRows : _numCols;
      int height = getFollowingDim(); //isTrans() ? _numCols : _numRows;
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
