#!/bin/bash

python test_brats.py --name  --config_file ./configs/brats2018_3m.json \
    --label_nc 3 --output_nc 1 --hr_depth 5 --hr_width 64 --latent_dim 256 --batchSize 1 --ds_factor 1 --phase test \
     --results_dir /exports/lkeb-hpc/ychen/03_result/02_image_synthesis/02_translation_results/inr_t1t2f_t1ce_brats2018_2p/
