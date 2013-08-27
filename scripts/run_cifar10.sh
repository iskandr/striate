#!/bin/sh


  #--param_file ~/striate/striate/fastcifar.cfg \
python ./striate/trainer.py \
  --data_dir /ssd/nn-data/cifar-10.old \
  --data_provider cifar10 \
  --param_file ./config/cifar_from_imagenet.cfg \
  --train_range 1-40 \
  --test_range 40-48 \
  --save_freq 50 \
  --test_freq 50 \
  --adjust_freq 100 \
  --learning_rate 1.0 \
  --batch_size 128 \
  --checkpoint_dir ~/striate/striate/checkpoint/ \
  --trainer normal \
  --num_epoch 50 \
  $@
