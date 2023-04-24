"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os,sys
from collections import OrderedDict

import data
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
    #dataset = data.find_dataset_using_name(opt.dataset_mode)
    save_dir = '/exports/lkeb-hpc/ychen/03_result/02_image_synthesis/02_translation_results/neuro_brats_resize/'
    data_dir = '/exports/lkeb-hpc/ychen/01_data/03_preprocessed/02_bratsSyn/004_rawdata/valid_data'
    #print("dataset [%s] of size %d was created" %
    #        (type(instance).__name__, len(instance)))
    #dataloader = torch.utils.data.DataLoader(
    #    instance,
    #    batch_size=opt.batchSize,
    #    shuffle=not opt.serial_batches,
    #    num_workers=int(opt.nThreads),
    #    drop_last=opt.isTrain
    #)

    #dataloader = data.create_dataloader(opt)
    device = torch.device('cpu' if opt.gpu_ids == -1 else 'cuda')
    patch_size = (240,240)
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
    time_total = 0
    # testa
    input_modalities = {'t2'}
    
    for patient in patientlist:
        print(patient)
        image_list = []
        
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
        image_data = np.concatenate(image_list,axis=1)
        

        prediction = -1 * np.ones((image_data.shape[2], image_data.shape[3], image_data.shape[0]))
        #prediction = np.zeros((128, 160, image_data.shape[0]))
        print(image_data.shape)
        #for i in range(image_data.shape[0]):
        for i in range(image_data.shape[0]):
            ## cropping
            data = image_data[i,:,:, :]
            ## padding
            data = torch.from_numpy(data)
            data = data.to(device=device,dtype=torch.float32)
            
            with torch.no_grad():
                data_to_test = data[None, :, :, :]
                fake_img = model(data_to_test, mode='my_inference')
                pred = fake_img.squeeze(0)
            
            prediction[:,:,i] = pred.cpu().numpy()
        img_to_save = nib.Nifti1Image(prediction,image_handle.affine)
        save_name = os.path.join(save_dir, patient + ".nii.gz")
        nib.save(img_to_save,save_name)

if __name__ == '__main__':
    opt = TestOptions().parse()
    test(opt)
