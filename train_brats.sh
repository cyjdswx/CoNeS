#!/bin/bash

python train.py --name brats2018_2split_t1t2f_t1ce --config_file ./configs/brats2018_3m.json \
    --label_nc 3 --output_nc 1 --batchSize 8 --no_vgg_loss --L1_loss \
    --hr_depth 5 --hr_width 64 --latent_dim 256 --use_gan --latent_code_regularization
