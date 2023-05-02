#!/bin/bash

python test_bratssegrm.py --name brats_fullres --config_file ./configs/brats_4m.json \
    --label_nc 4 --output_nc 4 --hr_depth 5 --hr_width 64 --latent_dim 256 --batchSize 1 --ds_factor 1 --phase test \
    --results_dir /exports/lkeb-hpc/ychen/03_result/02_image_synthesis/02_translation_results/brats_fullres
