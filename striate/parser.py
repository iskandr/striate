import sys
import os
import string

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


class Parser:
  def __init__(self, parsing_file):
    assert os.path.exists(parsing_file)
    self.parsing_file = parsing_file
    self.__parse()

  def get_result(self):
    return self.rst

  def __parse(self):
    with open(self.parsing_file) as f:
      for line in f:
        line = line.strip()
        if line.startswith('#'):
          continue
        elif line.startswith('['):
          name = line[1:line.find(']')]
          self.rst.append({'name':name})
        else:
          key = line[0:line.find('=')]
          value = line[line.find('=')+1, -1]

          if value.isdigit():
            value = int(value)
          elif isfloat(value):
            value = float(value)

          self.rst[-1][key] = value
