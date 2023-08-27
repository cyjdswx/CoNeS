"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os,sys
from skimage.measure import label
import data
import json
import argparse
from models.unet_model import UnetModel
#from util.visualizer import Visualizer
import torch.nn.functional as F
import torch
import nibabel as nib
import numpy as np
from util.util import pad_nd_image, compute_steps_for_sliding_window, get_gaussian

def getlargestcc(segmentation):
    labels = label(segmentation)
    #assert(labels.max() != 0)
    if labels.max() == 0:
        print('no segment')
        return np.ones_like(labels)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC

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

if __name__=='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # experiment specifics
    parser.add_argument('--name', type=str, default='label2coco', help='name of the experiment')
    parser.add_argument('--config_file', type=str,default='./configs/brats.json')
    parser.add_argument('--nThreads', default=12, type=int, help='# threads for loading data')
    parser.add_argument('--load_from_opt_file', action='store_true', help='load the options from checkpoints and use that as default')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    
    parser.add_argument('--model', type=str, default='pix2pix', help='which model to use')
    parser.add_argument('--norm_G', type=str, default='instanceaffine', help='instance normalization or batch normalization')
    parser.add_argument('--norm_D', type=str, default='spectralinstance', help='instance normalization or batch normalization')
    parser.add_argument('--norm_E', type=str, default='spectralinstance', help='instance normalization or batch normalization')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

    # input/output sizes
    parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
    parser.add_argument('--aspect_ratio', type=float, default=1.0, help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')
    parser.add_argument('--label_nc', type=int, default=3, help='# of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.')
    parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')

    # Hyperparameters
    parser.add_argument('--learned_ds_factor', type=int, default=16, help='enables partial learned_ds (S2 in sec. 3.2)')
    parser.add_argument('--ds_factor', type=int, default=5, help='enables partial learned_ds (S2 in sec. 3.2)')
    parser.add_argument('--lowest_ds_factor', type=int, default=16, help='enables partial learned_ds (S2 in sec. 3.2)')
    parser.add_argument('--lr_width', type=int, default=64, help='low res stream strided conv number of channles')
    parser.add_argument('--lr_max_width', type=int, default=1024, help='low res stream conv number of channles')
    parser.add_argument('--lr_depth', type=int, default=7, help='low res stream number of conv layers')
    parser.add_argument('--hr_width', type=int, default=64, help='high res stream number of MLP channles')
    parser.add_argument('--hr_depth', type=int, default=5, help='high res stream number of MLP layers')
    parser.add_argument('--latent_dim', type=int, default=64, help='high res stream number of MLP layers')
    parser.add_argument('--reflection_pad', action='store_true', help='if specified, use reflection padding at lr stream')
    parser.add_argument('--replicate_pad', action='store_true', help='if specified, use replicate padding at lr stream')
    parser.add_argument('--netG', type=str, default='ASAPNets', help='selects model to use for netG')
    parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')
    parser.add_argument('--hr_coor', choices=('cosine', 'None','siren'), default='cosine')
    parser.add_argument('--deep_supervision', action='store_true')

    parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')

    parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        
    parser.set_defaults(serial_batches=True)
    parser.set_defaults(no_flip=True)
    parser.set_defaults(phase='test')
    
    opt = parser.parse_args()
    opt.isTrain = False   # train or test
    
    save_dir = opt.results_dir
    pred_dir = os.path.join(save_dir, 'synimg')
    if not os.path.isdir(pred_dir):
        os.mkdir(pred_dir)
    seg_dir = os.path.join(save_dir, 'segmentation')
    if not os.path.isdir(seg_dir):
        os.mkdir(seg_dir)
    #data_dir = '/exports/lkeb-hpc/ychen/01_data/03_preprocessed/02_bratsSyn/004_rawdata/train_data'
    #data_dir = '/exports/lkeb-hpc/ychen/01_data/03_preprocessed/02_bratsSyn/004_rawdata/valid_data'
    #data_dir = '/exports/lkeb-hpc/ychen/01_data/03_preprocessed/02_bratsSyn/009_brats18/valid_data'
    #data_dir = '/exports/lkeb-hpc/ychen/01_data/03_preprocessed/02_bratsSyn/009_brats18/valid_data_synt1ce_inr'
    #data_dir = '/exports/lkeb-hpc/ychen/01_data/03_preprocessed/02_bratsSyn/009_brats18/valid_data_synt1ce_pix2pix'
    #data_dir = '/exports/lkeb-hpc/ychen/01_data/03_preprocessed/02_bratsSyn/009_brats18/valid_data_synt1ce_resvit'
    #data_dir = '/exports/lkeb-hpc/ychen/01_data/03_preprocessed/02_bratsSyn/009_brats18/valid_data_synt1ce_pGAN'
    #data_dir = '/exports/lkeb-hpc/ychen/01_data/03_preprocessed/02_bratsSyn/009_brats18/valid_data_synt1ce_asap'
    data_dir = '/exports/lkeb-hpc/ychen/01_data/03_preprocessed/02_bratsSyn/009_brats18/valid_data_synt1ce_asap_8'
    #data_dir = '/exports/lkeb-hpc/ychen/01_data/03_preprocessed/02_bratsSyn/009_brats18/valid_data_synt1_asap'
    #data_dir = '/exports/lkeb-hpc/ychen/01_data/03_preprocessed/02_bratsSyn/009_brats18/valid_data_synt2_asap'
    #data_dir = '/exports/lkeb-hpc/ychen/01_data/03_preprocessed/02_bratsSyn/009_brats18/valid_data_synflair_asap'
    #data_dir = '/exports/lkeb-hpc/ychen/01_data/03_preprocessed/02_bratsSyn/009_brats18/valid_data_synt1_inr'
    #data_dir = '/exports/lkeb-hpc/ychen/01_data/03_preprocessed/02_bratsSyn/009_brats18/valid_data_synt1_pix2pix'
    #data_dir = '/exports/lkeb-hpc/ychen/01_data/03_preprocessed/02_bratsSyn/009_brats18/valid_data_synt1_pGAN'
    #data_dir = '/exports/lkeb-hpc/ychen/01_data/03_preprocessed/02_bratsSyn/009_brats18/valid_data_synt1_resvit'
    #data_dir = '/exports/lkeb-hpc/ychen/01_data/03_preprocessed/02_bratsSyn/009_brats18/valid_data_synt2_pix2pix'
    #data_dir = '/exports/lkeb-hpc/ychen/01_data/03_preprocessed/02_bratsSyn/009_brats18/valid_data_synt2_pGAN'
    #data_dir = '/exports/lkeb-hpc/ychen/01_data/03_preprocessed/02_bratsSyn/009_brats18/valid_data_synt2_resvit'
    #data_dir = '/exports/lkeb-hpc/ychen/01_data/03_preprocessed/02_bratsSyn/009_brats18/valid_data_synt2_inr'
    #data_dir = '/exports/lkeb-hpc/ychen/01_data/03_preprocessed/02_bratsSyn/009_brats18/valid_data_synflair_pix2pix'
    #data_dir = '/exports/lkeb-hpc/ychen/01_data/03_preprocessed/02_bratsSyn/009_brats18/valid_data_synflair_pGAN'
    #data_dir = '/exports/lkeb-hpc/ychen/01_data/03_preprocessed/02_bratsSyn/009_brats18/valid_data_synflair_resvit'
    #data_dir = '/exports/lkeb-hpc/ychen/01_data/03_preprocessed/02_bratsSyn/009_brats18/valid_data_synflair_inr'
    device = torch.device('cpu' if opt.gpu_ids == -1 else 'cuda')
    patch_size = (128,160)
    #patch_size = (160,192)
    model = UnetModel(opt, device=device)
    model.eval()

    #visualizer = Visualizer(opt)
    with open(os.path.join(data_dir,"patientlist.txt")) as f:
        patientlist = f.read().splitlines()
    
    #print(target_spacing)
    device = torch.device('cuda')
    # create a webpage that summarizes the all results
    #                    (opt.name, opt.phase, opt.which_epoch))
    # test
    with open(opt.config_file) as json_data:
        try:
            configs = json.load(json_data)
            print(configs)
        except json.JSONDecodeError:
            print('invalid json format')
            sys.exit()
    
    dataset_dict = configs['dataset']
    input_modalities= dataset_dict['input_modalities']
    print(input_modalities)
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
        

        prediction = -1 * np.ones((image_data.shape[2], image_data.shape[3], image_data.shape[0]))
        segmentation = np.zeros((image_data.shape[2], image_data.shape[3], image_data.shape[0]))
        #prediction = -1 * np.ones((128, 160, image_data.shape[0]))
        
        x = create_nonzero_mask(np.transpose(image_data,(1,0,2,3)))
        bbox = get_bbox_from_mask(x, 0)
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
            #aggregated_results = torch.zeros([1] + list(data.shape[1:]), device=device)
            #aggregated_nb_of_predictions = torch.zeros([1] + list(data.shape[1:]), device=device)
            
            aggregated_segmentation = torch.zeros([4] + list(data.shape[1:]), device=device)
            aggregated_nb_of_segmentation = torch.zeros([4] + list(data.shape[1:]), device=device)
            
            for x in steps[0]:
                lb_x = x
                ub_x = x + patch_size[0]
                for y in steps[1]:
                    lb_y = y
                    ub_y = y + patch_size[1]
                    with torch.no_grad():
                        data_to_test = data[None, :, lb_x:ub_x, lb_y:ub_y]
                        seg = model(data_to_test, mode='inference')
                        #aggregated_nb_of_predictions[:,lb_x:ub_x,lb_y:ub_y] += add_for_nb_of_preds
                        
                        prob = F.softmax(seg, dim=1)
                        prob = prob.squeeze(0)
                        aggregated_segmentation[:,lb_x:ub_x,lb_y:ub_y] += prob
                        aggregated_nb_of_segmentation[:,lb_x:ub_x,lb_y:ub_y] += add_for_nb_of_preds

            #slicer_trans = tuple(
            #    [slice(0, aggregated_results.shape[0])] + slicer[1:])
            slicer_seg_trans = tuple(
                [slice(0, aggregated_segmentation.shape[0])] + slicer[1:])
            
            #aggregated_results = aggregated_results[slicer_trans]
            #aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer_trans]
            #pred_slice = aggregated_results / aggregated_nb_of_predictions
            
            #prediction[bbox[1][0]:bbox[1][1],bbox[2][0]:bbox[2][1],i] = pred_slice.cpu().numpy()
            aggregated_segmentation = aggregated_segmentation[slicer_seg_trans]
            aggregated_nb_of_segmentation = aggregated_nb_of_segmentation[slicer_seg_trans]
            seg_slice = aggregated_segmentation / aggregated_nb_of_segmentation

            #if Segnet.n_classes > 1:
            mask = seg_slice.argmax(dim=0).cpu().numpy()
            #else:
            #mask = (segmentation > 0.5).cpu().numpy()
            segmentation[bbox[1][0]:bbox[1][1],bbox[2][0]:bbox[2][1],i] = mask

        #img_to_save = nib.Nifti1Image(prediction, img_affine)
        #save_name = os.path.join(pred_dir, patient + ".nii.gz")
        #nib.save(img_to_save,save_name)
        
        prediction_bin = (segmentation >= 1).astype(int)
        largestCC = getlargestcc(prediction_bin)
        segmentation = segmentation * largestCC
        seg_to_save = nib.Nifti1Image(segmentation,img_affine)
        save_name = os.path.join(seg_dir, patient + ".nii.gz")
        nib.save(seg_to_save,save_name)
