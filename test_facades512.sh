#!/bin/bash
python3 test.py --name facades512 --dataroot ./datasets/facadesHR --dataset_mode facades --load_size 512 --crop_size 512 \
    --batchSize 1 --gpu_ids 0 --phase test --reflection_pad
