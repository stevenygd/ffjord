#! /bin/bash

python train_3d_toy.py \
    --viz_freq 1 \
    --dims "128-128-128" \
    --num_blocks 5 \
    --data ${1}
