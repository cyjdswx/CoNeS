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
import SimpleITK as sitk
import time
import torch
import math
import nibabel as nib
import numpy as np
from util.util import pad_nd_image, compute_steps_for_sliding_window, get_gaussian
import pickle
from batchgenerators.augmentations.utils import resize_segmentation
from skimage.transform import resize
from scipy.ndimage.interpolation import map_coordinates

def create_nonzero_mask(data):
    from scipy.ndimage import binary_fill_holes
    assert len(data.shape) == 4 or len(data.shape) == 3, "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        this_mask = data[c] != -1
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask

def resample_data(data, new_shape, is_seg, axis=None, order=3, do_separate_z=False, order_z=0):
    """
    separate_z=True will resample with order 0 along z
    :param data:
    :param new_shape:
    :param is_seg:
    :param axis:
    :param order:
    :param do_separate_z:
    :param cval:
    :param order_z: only appliesif do_separate_z is True
    :return:
    """
    assert len(data.shape) == 4, "data must be (c, x, y, z)"
    if is_seg:
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
    else:
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
    dtype_data = data.dtype
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if np.any(shape != new_shape):
        data = data.astype(float)
        if do_separate_z:
            print("separate z, order in z is", order_z, "order inplane is", order)
            assert len(axis) == 1, "only one anisotropic axis supported"
            axis = axis[0]
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            reshaped_final_data = []
            for c in range(data.shape[0]):
                reshaped_data = []
                for slice_id in range(shape[axis]):
                    if axis == 0:
                        reshaped_data.append(resize_fn(data[c, slice_id], new_shape_2d, order, **kwargs))
                    elif axis == 1:
                        reshaped_data.append(resize_fn(data[c, :, slice_id], new_shape_2d, order, **kwargs))
                    else:
                        reshaped_data.append(resize_fn(data[c, :, :, slice_id], new_shape_2d, order,
                                                       **kwargs))
                reshaped_data = np.stack(reshaped_data, axis)
                if shape[axis] != new_shape[axis]:

                    # The following few lines are blatantly copied and modified from sklearn's resize()
                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_data.shape

                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5

                    coord_map = np.array([map_rows, map_cols, map_dims])
                    if not is_seg or order_z == 0:
                        reshaped_final_data.append(map_coordinates(reshaped_data, coord_map, order=order_z,
                                                                   mode='nearest')[None])
                    else:
                        unique_labels = np.unique(reshaped_data)
                        reshaped = np.zeros(new_shape, dtype=dtype_data)

                        for i, cl in enumerate(unique_labels):
                            reshaped_multihot = np.round(
                                map_coordinates((reshaped_data == cl).astype(float), coord_map, order=order_z,
                                                mode='nearest'))
                            reshaped[reshaped_multihot > 0.5] = cl
                        reshaped_final_data.append(reshaped[None])
                    
                else:
                    reshaped_final_data.append(reshaped_data[None])
            reshaped_final_data = np.vstack(reshaped_final_data)
        else:
            print("no separate z, order", order)
            reshaped = []
            for c in range(data.shape[0]):
                reshaped.append(resize_fn(data[c], new_shape, order, **kwargs)[None])
            reshaped_final_data = np.vstack(reshaped)
        return reshaped_final_data.astype(dtype_data)
    else:
        print("no resampling necessary")
        return data

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
    save_dir = '/exports/lkeb-hpc/ychen/03_result/02_image_synthesis/02_translation_results/neuro_vs_t2t1ce/'
    data_dir = '/exports/lkeb-hpc/ychen/01_data/03_preprocessed/03_vssyn/01_rawdata/valid_data/'
    #dataloader = data.create_dataloader(opt)
    traindata_dir="/exports/lkeb-hpc/ychen/01_data/03_preprocessed/03_vssyn/03_preprocessednew/1-1minmax"
    device = torch.device('cpu' if opt.gpu_ids == -1 else 'cuda')
    patch_size = (448,448)
    model = Pix2PixModel(opt, patch_size,device=device)
    model.eval()

    #visualizer = Visualizer(opt)
    with open(os.path.join(data_dir,"patientlist.txt")) as f:
        patientlist = f.read().splitlines()
    
    with open(os.path.join(traindata_dir, "dataset.pkl"),"rb") as f:
        while True:
            try:
                objects=pickle.load(f)
            except EOFError:
                break
    target_spacing = objects["spacing_after_resampling"]
    print(target_spacing)
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
        img_affine = None
        for m in input_modalities:
            image_handle = nib.load(os.path.join(data_dir, patient, patient + '_' + m + '.nii.gz'))
            #print(os.path.join(data_dir, patient, patient + '_' + m + '.nii.gz'))
            spacing_sitk = sitk.ReadImage(os.path.join(data_dir, patient, patient+'_'+m+'.nii.gz')).GetSpacing()
            original_spacing = np.array([spacing_sitk[2], spacing_sitk[0], spacing_sitk[1]]) 

            image_np =  np.asarray(image_handle.dataobj) .astype(np.float32)
            image_np = np.transpose(image_np, (2,0,1))
            origin_shape = image_np.shape
            image_np = np.expand_dims(image_np,axis=0)
            print('original_spacing', original_spacing)
            print('origin_size', origin_shape)
            ##resampling
            target_spacing[0] = original_spacing[0]
            new_shape = np.round(((np.array(original_spacing) / np.array(target_spacing)).astype(float) * origin_shape)).astype(int)
            image_reshaped = resample_data(image_np, new_shape, False, None, 3)
            image_reshaped = image_reshaped.squeeze()
            print('after', image_reshaped.shape)

            imax = np.max(image_np)
            imin = np.min(image_np)
            image_reshaped = (image_reshaped.astype(np.float64)-imin)/(imax-imin)
            image_reshaped = 2 * image_reshaped - 1
            image_reshaped = np.expand_dims(image_reshaped, axis=1)
            image_list.append(image_reshaped)
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
                        fake_img = model(data_to_test, mode='my_inference')
                        probs = fake_img.squeeze(0)
                        
                        aggregated_results[:,lb_x:ub_x,lb_y:ub_y] += probs
                        aggregated_nb_of_predictions[:,lb_x:ub_x,lb_y:ub_y] += add_for_nb_of_preds
            slicer_trans = tuple(
                [slice(0, aggregated_results.shape[0])] + slicer[1:])

            aggregated_results = aggregated_results[slicer_trans]
            aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer_trans]
            pred_slice = aggregated_results / aggregated_nb_of_predictions
            
            prediction[bbox[1][0]:bbox[1][1],bbox[2][0]:bbox[2][1],i] = pred_slice.cpu().numpy()
        
        prediction = np.transpose(prediction,(2,0,1))
        prediction = np.expand_dims(prediction,axis=0)
        prediction_reshape = resample_data(prediction, origin_shape, False, None, 3)
        prediction_reshape = prediction_reshape.squeeze()
        prediction_reshape = np.transpose(prediction_reshape,(1,2,0))           
        img_to_save = nib.Nifti1Image(prediction_reshape, img_affine)
        save_name = os.path.join(save_dir, patient + ".nii.gz")
        nib.save(img_to_save,save_name)

if __name__ == '__main__':
    opt = TestOptions().parse()
    test(opt)