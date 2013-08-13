#!/bin/sh


python ./striate/trainer.py \
  --data_dir /ssd/nn-data/cifar-10-py-colmajor \
  --param_file ./config/cifar-10-26pct.cfg \
  --data_provider cifar10 \
  --train_range 1-5 \
  --test_range 6 \
  --save_freq 10 \
  --test_freq 10 \
  --adjust_freq 100 \
  --learning_rate 0.1 \
  --batch_size 128 \
  --checkpoint_dir /tmp/checkpoint/ \
  --trainer normal \
  --num_epoch 1 \
  $@
