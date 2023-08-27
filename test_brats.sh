#!/bin/bash

python test_seg.py --name brats2018_nnunet_t1cet1t2f_sgd --config_file ./configs/brats2018.json \
    --label_nc 4 --output_nc 1 --deep_supervision --hr_depth 5 --hr_width 64 --latent_dim 256 --batchSize 1 --ds_factor 1 --phase test \
     --results_dir /exports/lkeb-hpc/ychen/03_result/00_seg/brats/brats2018_nnunet_t1cet1t2f_sgd_synt1ce_asap_8/
