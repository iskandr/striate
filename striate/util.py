import time
import cPickle
class Timer:
  def __init__(self):
    self.func_time = {}
    self.last_time = 0.0

  def start(self):
    self.last_time = time.time()

  def end(self, func_name):
    ftime = time.time() - self.last_time
    if func_name in self.func_time:
      self.func_time[func_name] += ftime
    else:
      self.func_time[func_name] = ftime

  def report(self):
    dic = self.func_time
    for key in sorted(dic):
      print key, ':', dic[key]


timer = Timer()

def ceil(x, base):
  if x / base * base == x:
    return x / base
  else:
    return x / base + 1

def load(filename):
  with open(filename, 'rb') as f:
    model = cPickle.load(f)
  return model

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def isinteger(value):
  try:
    int(value)
    return True
  except ValueError:
    return False

def string_to_int_list(str):
  str = str.strip()
  if str.find('-'):
    f = int(str[0:str.find('-')])
    t  = int(str[str.find('-') + 1:-1])

    return range(f, t +1)
  else:
    elt = int(str)
    return [elt]
