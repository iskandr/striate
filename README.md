Striate
=========
A [convolutional neural network](http://yann.lecun.com/exdb/lenet/) framework, building on 
top of the convolutional kernel code from [cuda-convnet](https://code.google.com/p/cuda-convnet/).

**Usage**

  See `striate/trainer.py` for examples of running the trainer.

    (cd cudaconv2 && make -j)
    python striate/trainer.py


**Requires**

  * [NumPy](http://www.numpy.org/)
  * [CUDA](http://www.nvidia.com/object/cuda_home_new.html)
  * [PyCUDA](http://documen.tician.de/pycuda/)
