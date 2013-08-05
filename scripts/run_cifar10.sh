#!/bin/sh


CUDA_DEVICE=1 python  ~/striate/striate/trainer.py \
  --data_dir /hdfs/cifar/data/cifar-10-python/ \
  --param_file ~/striate/striate/fastcifar.cfg \
  --data_provider cifar10 \
  --train_range 1-40 \
  --test_range 1-48 \
  --save_freq 10 \
  --test_freq 10 \
  --adjust_freq 100 \
  --learning_rate 1.0 \
  --batch_size 128 \
  --checkpoint_dir ~/striate/striate/checkpoint/ \
  --trainer normal \
  --num_epoch 25 \
  --loading_file ~/striate/striate/checkpoint/test1000-26.27  \
  $@
