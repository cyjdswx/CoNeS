#!/bin/bash

python train.py --name brats2018_test --config_file ./configs/brats2018_t2t1ce.json \
    --label_nc 1 --output_nc 1 --batchSize 8 --L1_loss \
    --hr_depth 5 --hr_width 64 --latent_dim 256 --use_gan --latent_code_regularization
