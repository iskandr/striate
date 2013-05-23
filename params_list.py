"""
The irregular collection of network parameters stored as 
a list of GPU arrays and scalars 
"""  


import theano.sandbox.cuda
from theano.misc.pycuda_utils import to_cudandarray, to_gpuarray 
import pycuda
import pycuda.autoinit 
from pycuda.gpuarray import GPUArray 

class ParamsList(object):
  def __init__(self, xs = None, copy_first=True):
    self.copy_first = copy_first 
    if xs is None:
      self.xs = None
      self.n_updates = 0
    else:
      self.xs = self._copy_params(xs) if copy_first else xs 
      self.n_updates = 1

  def _copy_param(self, x):
      if isinstance(x, theano.sandbox.cuda.CudaNdarray):
        return to_gpuarray(x, copyif=True)
      elif hasattr(x, 'copy'):
        return x.copy()
      else:
        return x

  def _copy_params(self, xs):
      return [self._copy_param(xi) for xi in xs]

  def iadd(self, ys):
    self.n_updates += 1

    if self.xs is None:
      self.xs = self._copy_params(ys) if self.copy_first else ys
    else:
      assert len(self.xs) == len(ys)
      for i, yi in enumerate(ys):
        # these are bizarrely faster
        if hasattr(self.xs[i], 'mul_add'):
           if not isinstance(yi, GPUArray):
             yi = to_gpuarray(yi, copyif=True)
           self.xs[i].mul_add(1.0, yi, 1.0)
        else:
           self.xs[i] += yi
  
  def flatten(self):
    assert self.xs is not None 
    elts_per_subarray = []
    is_array = []
    elt_type = None
    gpu_arrays = [] 
    for w in self.xs:
      if isinstance(w, theano.sandbox.cuda.CudaNdarray):
        w = to_gpuarray(w, copyif=True)
      if isinstance(w, (np.ndarray, GPUArray)):
        if not hasattr(w, 'gpudata'):
          w = pycuda.gpuarray.to_gpu(w)
        elts_per_subarray.append(w.size)
        is_array.append(True)
        if elt_type is None:
          elt_type = w.dtype 
      else:
        assert np.isscalar(w)
        elts_per_subarray.append(1)
        is_array.append(False)
      gpu_arrays.append(w) 
    total_elts = sum(elts_per_subarray)
    result = pycuda.gpuarray.empty((total_elts,), dtype = elt_type)  
    curr_idx = 0
    for (nelts, w) in zip(elts_per_subarray, gpu_arrays):
      if not np.isscalar(w):
        w = w.ravel()
      result_slice = result[curr_idx:(curr_idx + nelts)]
      if isinstance(w, GPUArray):
        pycuda.driver.memcpy_dtod(result_slice.gpudata, w.gpudata, w.nbytes)
      else:
        result_slice.set(w)
      curr_idx += nelts
    assert curr_idx == total_elts  
    return result
