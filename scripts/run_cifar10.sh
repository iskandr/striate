#!/bin/sh


CUDA_DEVICE=1 python  ./striate/trainer.py \
  --data_dir /hdfs/cifar/data/cifar-10-python/ \
  --param_file ./striate/cifar10.cfg \
  --data_provider cifar10 \
  --train_range 1-48 \
  --test_range 48 \
  --save_freq 10 \
  --test_freq 10 \
  --adjust_freq 100 \
  --learning_rate 0.0 \
  --batch_size 1000 \
  --checkpoint_dir /tmp/checkpoint/ \
  --trainer normal \
  --num_epoch 1 \
  $@
