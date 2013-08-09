#!/bin/sh

python ~/striate/striate/trainer.py  \
  --data_dir /ssd/nn-data/imagenet/ \
  --param_file ~/striate/striate/imagenet.cfg \
  --data_provider imagenet \
  --train_range 0-1200 \
  --test_range 1200-1300 \
  --save_freq 100 \
  --test_freq 100 \
  --adjust_freq 100 \
  --learning_rate "[0.1, 0.1]" \
  --batch_size 128 \
  --checkpoint_dir ~/striate/striate/checkpoint/ \
  --trainer catewise \
  --num_minibatch "[10000, 30000]" \
  --num_caterange_list "[100, 1000]" \
  $@
