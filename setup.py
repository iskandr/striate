#!/usr/bin/env python

from setuptools import setup, Extension

import sys


extension_modules = []
setup(
    name="striate",
    description="Convolutional neural networks in Python", 
    long_description='''
Striate. 
=========
A simple implementation of a Convolutional Neural Network which is based on (and extends) the [LeNet5 implementation](http://deeplearning.net/tutorial/lenet.html) in Theano's tutorial. 
Though most of the code runs on the GPU (thanks Theano & PyCUDA!), it's still not nearly as fast as you could make it with hand-rolled kernels. Also, the provided training algorithm is just mini-batch 
stochastic gradient descent. I'm using this library as a testbed for experimenting with other learning algorithms: the code is a sloppy prototype cobbled together from other examples and 
the features are the minimum I need for my own needs. You've been warned! 

* If you want top-notch runtime and accuracy, go talk to [Alex K](https://code.google.com/p/cuda-convnet/).

* If you want to know about convolutional neural networks, go talk to [Yann LeCun](http://yann.lecun.com/). 

**Requires**
  
  * CUDA
  * Theano
  * PyCUDA
  * scikits.cuda 
  * NumPy
  * SciPy
''',
    classifiers=['Development Status :: 3 - Alpha',
		 'Topic :: Scientific/Engineering :: Artificial Intelligence'                 
                 'License :: OSI Approved :: BSD License',
                 'Intended Audience :: Developers',
                 'Programming Language :: Python :: 2.7',
                 ],
    author="Alex Rubinsteyn",
    author_email="alexr@cs.nyu.edu",
    license="BSD",
    version="0.1",
    url="http://github.com/iskandr/striate",
    packages=[ 'striate' ],
    package_dir={ '' : '.' },
    requires=[
      'pycuda', 
      'theano',
      'scikits.cuda',
      'numpy', 
      'scipy',
    ],
    ext_modules = extension_modules)
