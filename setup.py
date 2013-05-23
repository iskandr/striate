#!/usr/bin/env python

from setuptools import setup, Extension

import sys


extension_modules = []
setup(
    name="convnet",
    description="Convolutional neural networks in Python"
    long_description='''
ConvNet 
=========

Go talk to Yann LeCun. 
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
    url="http://github.com/iskandr/convnet",
    packages=[ 'convnet' ],
    package_dir={ '' : '.' },
    requires=[
      'pycuda', 
      'theano',
      'scikits.cuda',
      'numpy', 
      'scipy',
    ],
    ext_modules = extension_modules)
