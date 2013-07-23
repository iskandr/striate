#!/usr/bin/env python

import glob, re, cPickle
from os.path import basename, dirname

print __file__
DATADIR = '/ssd/nn-data/imagenet'
HERE = dirname(__file__) 

def synids():
  ids = glob.glob(DATADIR + '/n*')
  ids = [basename(x)[1:] for x in ids]
  ids = sorted(ids)
  return ids

def synid_to_name():
  syns = open(HERE + '/fall11_synsets.txt').read().split('\n')
  syns = dict([re.split(' ', x, maxsplit=1) for x in syns][:-1])
  for k, v in syns.items():
    syns[k] = v.split(',')[0]
  return syns
 
LABEL_TO_SYNID = synids()
SYNID_NAMES = synid_to_name()
LABEL_NAMES = [SYNID_NAMES[s] for s in LABEL_TO_SYNID]
SYNID_TO_LABEL = {}
for idx, synid in enumerate(LABEL_TO_SYNID):
  SYNID_TO_LABEL[synid] = idx    

meta = {}
meta['label_names'] = LABEL_NAMES
meta['synid_to_label'] = SYNID_TO_LABEL
meta['label_to_synid'] = LABEL_TO_SYNID

with open('./batches.meta', 'w') as m:
  m.write(cPickle.dumps(meta))

print 'Generated batches.meta'