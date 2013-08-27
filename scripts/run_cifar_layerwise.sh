#!/bin/sh


python ~/striate/striate/trainer.py \
  --data_dir /ssd/nn-data/cifar-10.old/ \
  --param_file ~/striate/config/cifar_from_imagenet.cfg \
  --data_provider cifar10 \
  --train_range 1-40 \
  --test_range 41-48 \
  --save_freq 100 \
  --test_freq 100 \
  --adjust_freq 100 \
  --learning_rate 1.0 \
  --batch_size 128 \
  --checkpoint_dir ~/striate/striate/checkpoint/ \
  --trainer layerwise \
  --num_epoch 15 \
  $@
