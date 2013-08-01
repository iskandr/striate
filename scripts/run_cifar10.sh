#!/bin/sh


python ~/striate/striate/trainer.py \
  --data_dir /hdfs/cifar/data/cifar-10-python/ \
  --param_file ~/striate/striate/cifar10.cfg \
  --data_provider cifar10 \
  --train_range 1-2 \
  --test_range 41 \
  --save_freq 100 \
  --test_freq 100 \
  --adjust_freq 100 \
  --learning_rate 1.0 \
  --batch_size 128 \
  --checkpoint_dir ~/striate/striate/checkpoint/ \
  --trainer normal \
  --num_epoch 1 \
  $@
