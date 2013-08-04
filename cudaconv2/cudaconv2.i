%module cudaconv2
%{
#include "cudaconv2.cuh"
#include "conv_util.cuh"
#include "nvmatrix.cuh"
%}

%typemap(in) NVMatrix& {
  PyObject* shape = PyObject_GetAttrString($input, "shape");
  long rows, cols;
  PyObject* data = PyObject_GetAttrString($input, "gpudata");
  Py_DECREF(shape);
  PyObject* strides = PyObject_GetAttrString($input, "strides");
  Py_DECREF(strides);

  long stride, itemsize;

  float* gpudata = (float*)PyInt_AsLong(data);
  Py_DECREF(data);
  
  PyArg_ParseTuple(shape, "ll", &rows, &cols);
  PyArg_ParseTuple(strides, "ll", &stride, &itemsize);
  stride = stride / itemsize;
  $1 = new NVMatrix(gpudata, rows, cols, stride);
}

%typemap(typecheck,precedence=SWIG_TYPECHECK_INTEGER) NVMatrix& {
  if (PyObject_HasAttrString($input, "shape") && PyObject_HasAttrString($input, "gpudata")) {
    $1 = 1;
  } else {
    $1 = 0;
  }
}

%exception {
  try {
    $function
  } catch (Exception& e) {
    PyErr_Format(PyExc_RuntimeError, "%s (%s:%d", e.why_.c_str(), e.file_.c_str(), e.line_);
    return NULL;
  }
}


%include "cudaconv2.cuh"
%include "nvmatrix.cuh"

void sum(NVMatrix& src, int axis, NVMatrix& target);
void addVector(NVMatrix& target, NVMatrix& vec);
void convLocalMaxPool(NVMatrix& images, NVMatrix& target, int numFilters,
                   int subsX, int startX, int strideX, int outputsX);
void convLocalAvgPool(NVMatrix& images, NVMatrix& target, int numFilters,
                   int subsX, int startX, int strideX, int outputsX);
void convLocalMaxUndo(NVMatrix& images, NVMatrix& maxGrads, NVMatrix& maxActs, NVMatrix& target,
                      int subsX, int startX, int strideX, int outputsX);
void convLocalAvgUndo(NVMatrix& avgGrads, NVMatrix& target,
                      int subsX, int startX, int strideX, int outputsX, int imgSize);
void convLocalAvgUndo(NVMatrix& avgGrads, NVMatrix& target,
                      int subsX, int startX, int strideX, int outputsX, int imgSize,
                      float scaleTargets, float scaleOutput);
void convLocalMaxUndo(NVMatrix& images, NVMatrix& maxGrads, NVMatrix& maxActs, NVMatrix& target,
                      int subsX, int startX, int strideX, int outputsX, float scaleTargets, float scaleOutput);

void convResponseNorm(NVMatrix& images, NVMatrix& denoms, NVMatrix& target, int numFilters, int sizeX, float addScale, float powScale);
void convResponseNormUndo(NVMatrix& outGrads, NVMatrix& denoms, NVMatrix& inputs, NVMatrix& acts, NVMatrix& target, int numFilters,
                         int sizeX, float addScale, float powScale, float scaleTargets, float scaleOutput);
void convContrastNorm(NVMatrix& images, NVMatrix& meanDiffs, NVMatrix& denoms, NVMatrix& target, int numFilters, int sizeX, float addScale, float powScale);
void convContrastNormUndo(NVMatrix& outGrads, NVMatrix& denoms, NVMatrix& meanDiffs, NVMatrix& acts, NVMatrix& target, int numFilters,
                         int sizeX, float addScale, float powScale, float scaleTargets, float scaleOutput);

void convGaussianBlur(NVMatrix& images, NVMatrix& filter, NVMatrix& target, bool horiz, int numChannels,
                      float scaleTargets, float scaleOutputs);
void convBedOfNails(NVMatrix& images, NVMatrix& target, int numChannels, int imgSize, int startX,
                    int strideX, float scaleTargets, float scaleOutput);
void convBedOfNailsUndo(NVMatrix& actsGrad, NVMatrix& target, int numChannels, int imgSize,
                        int startX, int strideX, float scaleTargets, float scaleOutput);

void convResizeBilinear(NVMatrix& images, NVMatrix& target, int imgSize, int tgtSize, float scale);
void convRGBToYUV(NVMatrix& images, NVMatrix& target);
void convRGBToLAB(NVMatrix& images, NVMatrix& target, bool center);
void convCrop(NVMatrix& imgs, NVMatrix& target, int imgSize, int tgtSize, int startY, int startX);
void normalizeLocalWeights(NVMatrix& weights, int numModules, float norm);
void convTICAGrad(NVMatrix& images, NVMatrix& ticas, NVMatrix& target, int numFilters, int sizeX, float scaleTarget, float scaleOutput);
void convTICA(NVMatrix& images, NVMatrix& target, int numFilters, int sizeX, float scaleTarget, float scaleOutput);
void convContrastNormCrossMap(NVMatrix& images, NVMatrix& meanDiffs, NVMatrix& denoms, NVMatrix& target,
                             int numFilters, int sizeF, float addScale, float powScale, bool blocked);
void convResponseNormCrossMapUndo(NVMatrix& outGrads, NVMatrix& denoms, NVMatrix& inputs, NVMatrix& acts, NVMatrix& target, int numFilters,
                         int sizeF, float addScale, float powScale, bool blocked, float scaleTargets, float scaleOutput);
void convResponseNormCrossMap(NVMatrix& images, NVMatrix& denoms, NVMatrix& target, int numFilters, int sizeF, float addScale,
                              float powScale, bool blocked);
                              
                              
