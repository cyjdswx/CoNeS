#!/bin/bash

python train_unet.py --name brats18_unet3m --config_file ./configs/brats_4m.json \
    --label_nc 3 --output_nc 1 --batchSize 8 --no_vgg_loss --ds_factor 1 --L1_loss \
    --hr_depth 5 --hr_width 64 --latent_dim 256  --use_gan --latent_code_regularization
