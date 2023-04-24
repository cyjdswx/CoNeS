"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os,sys
from collections import OrderedDict
import json
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
#from util.visualizer import Visualizer
from util import html
from torchvision.utils import save_image
import time
import torch
import math
import nibabel as nib
import numpy as np
from util.util import pad_nd_image, compute_steps_for_sliding_window, get_gaussian

def create_nonzero_mask(data):
    from scipy.ndimage import binary_fill_holes
    assert len(data.shape) == 4 or len(data.shape) == 3, "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        this_mask = data[c] != -1
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask

def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

def test(opt):
    save_dir = opt.results_dir
    #data_dir = '/exports/lkeb-hpc/ychen/01_data/03_preprocessed/02_bratsSyn/007_brats_3split_raw/valid_data/'
    data_dir = '/exports/lkeb-hpc/ychen/01_data/03_preprocessed/02_bratsSyn/004_rawdata/valid_data'
    #dataloader = data.create_dataloader(opt)
    device = torch.device('cpu' if opt.gpu_ids == -1 else 'cuda')
    patch_size = (128,160)
    model = Pix2PixModel(opt, patch_size,device=device)
    model.eval()
    #visualizer = Visualizer(opt)
    with open(os.path.join(data_dir,"patientlist.txt")) as f:
        patientlist = f.read().splitlines()
    #target_spacing = objects["spacing_after_resampling"]
    #print(target_spacing)
    device = torch.device('cuda')
    # create a webpage that summarizes the all results
    #                    (opt.name, opt.phase, opt.which_epoch))
    with open(opt.config_file) as json_data:
        try:
            configs = json.load(json_data)
            print(configs)
        except json.JSONDecodeError:
            print('invalid json format')
            sys.exit()
    

    dataset_dict = configs['dataset']

    input_modalities= dataset_dict['input_modalities']
    
    for patient in patientlist:
        print(patient)
        image_list = []
        img_affine = None
        for m in input_modalities:
            image_handle = nib.load(os.path.join(data_dir, patient, patient + '_' + m + '.nii.gz'))
            #print(os.path.join(data_dir, patient, patient + '_' + m + '.nii.gz'))
            image_np =  np.asarray(image_handle.dataobj) .astype(np.float32)
            imax = np.max(image_np)
            imin = np.min(image_np)
            image_np = (image_np.astype(np.float64)-imin)/(imax-imin)
            image_np = 2 * image_np - 1
            image_np = np.transpose(image_np, (2,0,1))
            image_np = np.expand_dims(image_np, axis=1)
            image_list.append(image_np)
            if img_affine is None:
                img_affine = image_handle.affine
        image_data = np.concatenate(image_list,axis=1)
        

        #prediction = -1 * np.ones((128, 160, image_data.shape[0]))
        prediction = -1 * np.ones((image_data.shape[2], image_data.shape[3], image_data.shape[0]))
        
        x = create_nonzero_mask(np.transpose(image_data,(1,0,2,3)))
        bbox = get_bbox_from_mask(x, 0)
        #for i in range(image_data.shape[0]):
        for i in range(bbox[0][0],bbox[0][1]):
            ## cropping
            data = image_data[i,:,bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
            ## padding
            data, slicer = pad_nd_image(data,new_shape=patch_size, mode='constant',kwargs={'constant_values':-1},return_slicer=True)
            image_size = (data.shape[1], data.shape[2])
            data = torch.from_numpy(data)
            data = data.to(device=device,dtype=torch.float32)
            step_size = 0.5
            steps = compute_steps_for_sliding_window(patch_size, image_size, step_size)
            #steps =[[0],[0]]
            num_tiles = len(steps[0]) * len(steps[1])
            
            add_for_nb_of_preds = torch.ones(patch_size, device=device)
            aggregated_results = torch.zeros([1] + list(data.shape[1:]), device=device)
            aggregated_nb_of_predictions = torch.zeros([1] + list(data.shape[1:]), device=device)
            for x in steps[0]:
                lb_x = x
                ub_x = x + patch_size[0]
                for y in steps[1]:
                    lb_y = y
                    ub_y = y + patch_size[1]
                    with torch.no_grad():
                        data_to_test = data[None, :, lb_x:ub_x, lb_y:ub_y]
                        fake_img = model(data_to_test, False, mode='my_inference')
                        probs = fake_img.squeeze(0)
                        aggregated_results[:,lb_x:ub_x,lb_y:ub_y] += probs
                        aggregated_nb_of_predictions[:,lb_x:ub_x,lb_y:ub_y] += add_for_nb_of_preds
            slicer_trans = tuple(
                [slice(0, aggregated_results.shape[0])] + slicer[1:])

            aggregated_results = aggregated_results[slicer_trans]
            aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer_trans]
            pred_slice = aggregated_results / aggregated_nb_of_predictions
            #prediction[:,:,i] = probs.cpu().numpy()
            prediction[bbox[1][0]:bbox[1][1],bbox[2][0]:bbox[2][1],i] = pred_slice.cpu().numpy()
            
        img_to_save = nib.Nifti1Image(prediction, img_affine)
        save_name = os.path.join(save_dir, patient + ".nii.gz")
        nib.save(img_to_save,save_name)

if __name__ == '__main__':
    opt = TestOptions().parse()
    test(opt)
