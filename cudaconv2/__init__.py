from .cudaconv2 import *

def init():
  # MAGIC MAGIC
  import pycuda.driver as cuda
  cuda.init()
  from pycuda.tools import make_default_context
  context = make_default_context()
  device = context.get_device()
  import atexit
  atexit.register(context.detach)

