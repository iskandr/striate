#!/bin/sh


  #--param_file ~/striate/striate/fastcifar.cfg \
python ./striate/trainer.py \
  --data_dir /ssd/nn-data/cifar-10.old \
  --data_provider cifar10 \
  --param_file ./config/cifar-10-18pct.cfg \
  --train_range 1-40 \
  --test_range 40-48 \
  --save_freq 10 \
  --test_freq 10 \
  --adjust_freq 100 \
  --learning_rate 0.1 \
  --batch_size 128 \
  --checkpoint_dir /tmp/checkpoint/ \
  --trainer normal \
  --num_epoch 1 \
  $@
