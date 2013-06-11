%module cudaconv2
%{
#include "cudaconv2.cuh"
#include "nvmatrix.cuh"
%}

%typemap(in) NVMatrix& {
  PyObject* shape = PyObject_GetAttrString($input, "shape");
  long rows, cols;
  PyObject* data = PyObject_GetAttrString($input, "gpudata");
  Py_DECREF(shape);

  float* gpudata = (float*)PyInt_AsLong(data);
  Py_DECREF(data);
  
  PyArg_ParseTuple(shape, "ll", &rows, &cols);
  $1 = new NVMatrix(gpudata, rows, cols, 128);
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
