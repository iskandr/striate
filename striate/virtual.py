import numpy as np
import threading
from operator import attrgetter
from time import sleep
import util
from pycuda import gpuarray, driver, autoinit
from mpi4py import rc
#rc.initialize = False

from mpi4py import MPI
#MPI.Init_thread(required=MPI.THREAD_MULTIPLE)

COMM = MPI.COMM_WORLD
rank = COMM.Get_rank()

class Point:
  def __init__(self, first, *args):
    l = [first]
    for a in args:
      l.append(a)
    self.point = tuple(l)

  def __len__(self):
    return len(self.point)

  def __str__(self):
    return str(self.point)

  def __eq__(self, other):
    return all([a == b for a, b in zip(self.point, other.point)])

  def expand(self, padding):
    assert len(padding) == len(self.point)

    return Point(*[a + b for a, b in zip(self.point, padding)])


class Area:
  def __init__(self, f, t):
    if len(f) != len(t):
      return None
    else:
      self.dim = len(f)
      self._from = f
      self._to = t

  def __add__(self, other):
    _from = Point(*[ min(a, b) for a, b in zip(self._from.point, other._from.point)])
    _to = Point(* [ max(a, b) for a, b in zip(self._to.point, other._to.point)])
    return  Area(_from, _to)

  def to_slice(self):
    return tuple([slice(a, b + 1) for a, b in zip(self._from.point, self._to.point)])

  def __contains__(self, other):
    return all([ a <= b for a, b in zip(self._from.point, other._from.point)]) and all([ a >= b for a, b in zip(self._to.point, other._to.point)])

  def __and__(self, other):
    _from =  Point(*[ max(a, b) for a, b in zip(self._from.point, other._from.point)])
    _to = Point(*[ min(a, b) for a, b in zip(self._to.point, other._to.point)])

    if all([ a <= b for a, b in zip(_from.point, _to.point)]):
      return Area(_from, _to)
    else:
      return None
  def __str__(self):
    return str(self._from) + ' to ' + str(self._to)

  def move(self, point):
    _from = Point(*[a - b for a, b in zip(self._from.point, point.point)])
    _to = Point(*[ a - b for a, b in zip(self._to.point , point.point)])

    return Area(_from, _to)

  def get_shape(self):
    return tuple([ a - b + 1 for a, b in zip(self._to.point, self._from.point)])

class TwoDarea:
  def __init__(self, row_from, row_to, col_from, col_to):
    self.row_from = row_from
    self.row_to = row_to
    self.col_from = col_from
    self.col_to = col_to

  def __add__(self, other):
    row_from = min(self.row_from, other.row_from)
    row_to = max(self.row_to, other.row_to)
    col_from = min(self.col_from, other.col_from)
    col_to = max(self.col_to, other.col_to)

    return TwoDarea(row_from, row_to, col_from, col_to)

  def __str__(self):
    return 'Area(row:%d-%d, col:%d-%d)' %(self.row_from, self.row_to, self.col_from, self.col_to)

  def __contains__(self, item):
    return self.row_from <= item.row_from and self.row_to >= item.row_to and self.col_from <= item.col_from and self.col_to >= item.col_to

  def __and__(self, other):
    row_from = max(self.row_from, other.row_from)
    row_to = min(self.row_to, other.row_to)
    col_from = max(self.col_from, other.col_from)
    col_to = min(self.col_to, other.col_to)

    if row_to < row_from or col_to < col_from:
      return None
    else:
      return TwoDarea(row_from, row_to, col_from, col_to)

MPI_TAG = {'area':1, 'data':2}

class virtual_array(object):
  def __init__(self, rank, area = None, local = None):
    self.dic = {}
    self.rank = rank
    self._global = None
    if area is not None:
      self.local = local
      self.dic[self.rank] = area
      self.sync_dict()

    #self.thread = threading.Thread(target = self.run_back)
    #self.daemon = True
    #self.thread.start()

  def _log(self, fmt, *args):
    util.log('%s :: %s' % (self.rank, fmt % args))

  def get_local_area(self):
    return self.dic[self.rank]

  def get_local_shape(self):
    return self.dic[self.rank].get_shape()

  def get_global_shape(self):
    return self._global.shape

  def get_local(self):
    return self.local

  def run_back(self):
    status = MPI.Status()
    while(1):
      self._log('PROBE')
      COMM.Probe(source = MPI.ANY_SOURCE, tag = MPI_TAG['area'], status = status)
      source = status.Get_source()
      self._log('Respond (RECV) %s ', status.Get_source())

      area = COMM.recv(source = source, tag = MPI_TAG['area'])
      self._log('Respond (SEND) %s', source)
      data = self.fetch_local(area)
      COMM.send(data, dest = source, tag = MPI_TAG['data'])
      self._log('Respond (DONE) %s', source)


  def store(self, local, area):
    self.local = local
    self.dic[self.rank] = area
    self.sync_dict()

  def fetch_local(self, area):
    if area is None:
      return None
    a = self.dic[self.rank]
    area = area.move(a._from)
    slices = area.to_slice()
    if isinstance(self.local, gpuarray.GPUArray):
      local = self.local.get()
    else:
      local = self.local
    t = local.__getitem__(slices)
    return t

  def fetch_remote(self, reqs, subs):
    send_req_req = {}
    recv_req_req = {}
    recv_req = {}
    for rank in reqs:
      area = reqs[rank]
      send_req_req[rank] = COMM.isend(area, dest = rank)
      recv_req_req[rank] = COMM.irecv(dest = rank)

    for rank in send_req_req:
      send_req_req[rank].wait()
      recv_req[rank] = recv_req_req[rank].wait()

    send_data = {rank : self.fetch_local(area) for rank, area in recv_req.items()}
    send_data_req = {}
    recv_data_req = {}
    for rank in send_data:
      data = send_data[rank]
      send_data_req[rank] = COMM.isend(data, dest = rank)
      recv_data_req[rank] = COMM.irecv(dest = rank)

    for rank in send_data:
      send_data_req[rank].wait()
      subs[reqs[rank]] = recv_data_req[rank].wait()
    COMM.barrier()

  def fetch(self, area):
    a = self.dic[self.rank]
    subs = {}
    reqs = {}
    if area in a:
      subs[area] = self.fetch_local(area)
      for i in range(COMM.Get_size()):
        if i != self.rank:
          reqs[i] = None
    else:
      for rank, a in self.dic.iteritems():
        sub_area = a & area
        if rank == self.rank:
          sub_array = self.fetch_local(sub_area)
          subs[sub_area] = sub_array
        else:
          reqs[rank] = sub_area
    self.fetch_remote(reqs, subs)
    return virtual_array.merge(subs, area)

  def sync_dict(self):
    rev = COMM.allgather(self.dic[self.rank])
    self.area = self.dic[self.rank]
    for i in range(COMM.Get_size()):
      self.dic[i] = rev[i]
      self.area += rev[i]

  def get_cross(self, padding):
    if self.dic[self.rank]._from == self.area._from: # up left
      f = self.area._from
      t = self.dic[self.rank]._to.expand((0, padding, padding))
    elif self.dic[self.rank]._to == self.area._to:
      f = self.dic[self.rank]._from.expand((0, -padding, -padding)) #down right
      t = self.area._to
    elif self.rank == 1:
      f = self.dic[self.rank]._from.expand((0, 0, -padding))
      t = self.dic[self.rank]._to.expand((0, padding, 0))
    else:
      f = self.dic[self.rank]._from .expand((0, -padding, 0))
      t = self.dic[self.rank]._to.expand((0, 0, padding))
    area = Area(f, t)
    return self.fetch(area), area

  def gather(self):
    self._global = self.fetch(self.area)

  def reshape(self, shape):
    assert self._global is not None
    self._global = self._global.reshape(shape)
    self.area = Area(Point(*([0]*len(shape))), Point(*[a - 1 for a in shape]))

  def distribute(self, axis = 0):
    if axis == -1:
      self.local = self._global
      return

    shape = self._global.shape
    assert axis < len(shape)

    l = shape[axis]
    size = COMM.Get_size()
    shade = (l + l % size) / size

    if axis == 0:
      f = Point(rank * shade, *([0] * (len(shape) - 1)))
      t = Point(rank * shade + shade -1 , *shape[1:])
    elif axis == len(shape) - 1:
      f = Point(*([0]*(len(shape) - 1)).append(rank * shade))
      t = Point(*(shape[0:-1].append(rank * shade + shade - 1)))
    else:
      f = Point(*([0] * axis).append(rank * shade).extend([0] * (len(shape) - axis - 1)))
      t = Point(*shape[0:axis].append(rank * shade + shade -1).extend(shape[axis + 1:]))

    area = Area(f, t)
    self.dic[self.rank] = area
    self.sync_dict()
    self.local = self._global.__getitem__(area.to_slice())


  def distribute_square(self):
    assert len(self._global.shape) == 4
    shape = self._global.shape
    c, h, w, b = shape
    assert h == w

    h /= 2
    w /= 2

    f = Point(0, rank / 2 * h, rank % 2 * w, 0)
    t = Point(c - 1, rank / 2 * h + h - 1, rank % 2 * w + w - 1, b -1)
    area = Area(f, t)
    self.dic[self.rank] = area
    self.sync_dict()
    self.local = self._global.__getitem__(area.to_slice())

  def add_reduce(self):
      area = self.dic[self.rank]
      subs = {}
      reqs = {}
      for rank, a in self.dic.iteritems():
        if rank == self.rank:
          continue
        reqs[rank] = a & area
      self.fetch_remote(reqs, subs)

      local = self.local
      for sub_area, sub_array in subs:
        if sub_array is None:
          continue
        slices = sub_area.to_slice()
        local.__setitem__(slices, local.__getitem__(slices) + sub_array)
      self.gather()


  def __str__(self):
    rst = self.dic[self.rank]
    for a in self.dic:
      rst += self.dic[a]
    return str(rst)

  @staticmethod
  def merge(subs, area):
    subs = {sub_area: sub_array for sub_area, sub_array in subs.iteritems() if sub_array is not None}
    row_from = area._from.point[1]
    a = sorted([sub_area for sub_area in subs if sub_area._from.point[1] == row_from], key = lambda x: x._to.point[2])
    rst = np.concatenate(tuple([subs[sub] for sub in a]), axis = 2)
    while True:
      row_from = a[0]._to.point[1] + 1
      a = sorted([sub_area for sub_area in subs if sub_area._from.point[1] == row_from], key = lambda x: x._to.point[2])
      if not a:
        break;
      else:
        tmp = np.concatenate(tuple([subs[sub] for sub in a]), axis = 2)
        rst = np.concatenate((rst, tmp), axis = 1)
    return rst


if __name__ == '__main__':
  h = w = 4
  c , i = 3, 2
  row_from, row_to, col_from, col_to = rank / 2 * h, rank / 2 * h + h - 1, rank % 2 * w, rank % 2 * w + w -1

  '''
  area = TwoDarea(row_from, row_to, col_from, col_to)
  local = gpuarray.to_gpu(np.random.randn(h, w).astype(np.float32))
  va = virtual_array(rank, local = local, area = area)


  if rank in [0,1,2,3]:
    area = TwoDarea(1, 3, 1, 6)
    util.log('START: %s %s', rank, area)
    sub = va.fetch(area)
    util.log('FETCH DONE: %s', rank)
    util.log('RESULT: %s', sub)
  '''
  f = Point(0, row_from, col_from)
  t = Point(c-1, row_to, col_to)
  area = Area(f, t)

  local = gpuarray.to_gpu(np.random.randn(c, h, w).astype(np.float32))
  print local
  va = virtual_array(rank, local = local, area = area)

  va.gather()
  va.reshape((h* w * 4, c))
  va.distribute()
  print va.local
