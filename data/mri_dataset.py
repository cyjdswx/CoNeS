import os
import numpy as np
from collections import OrderedDict
from scipy.ndimage import interpolation

import torch
from torch.utils.data import Dataset
import nibabel as nib
from torchvision.utils import Tuple, save_image
import torchvision.transforms.functional as F
from torch import Tensor
from typing import List, Union, Tuple
from torchvision import transforms
from batchgenerators.augmentations.utils import resize_segmentation

def randomFilp(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        if torch.rand(1) < self.p:
            for i in range(len(imgs)):
                imgs[i] = F.vflip(imgs[i])
        if torch.rand(1) < self.p:
            for i in range(len(imgs)):
                imgs[i] = F.hflip(imgs[i])
        return imgs

def downsample_seg_for_ds_transform2(seg, ds_scales=((1, 1, 1), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)), order=0, axes=None):
    if axes is None:
        axes = list(range(2, len(seg.shape)))
    output = []
    for s in ds_scales:
        if all([i == 1 for i in s]):
            output.append(seg.squeeze(0))
        else:
            new_shape = np.array(seg.shape).astype(float)
            for i, a in enumerate(axes):
                new_shape[a] *= s[i]
            new_shape = np.round(new_shape).astype(int)
            out_seg = np.zeros(new_shape, dtype=seg.dtype)
            for b in range(seg.shape[0]):
                for c in range(seg.shape[1]):
                    out_seg[b, c] = resize_segmentation(seg[b, c], new_shape[2:], order)
            out_seg = torch.from_numpy(out_seg).squeeze(0)
            output.append(out_seg)
    return output

class randomCrop_mask(object):
    def __init__(self, size, pad_if_needed=True, fill=0.0, padding_mode='constant'):
        self.size = size
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
    
    @staticmethod
    #def get_params(img : Tensor, patch_size : tuple[int,int]) -> Tuple[int,int,int,int]:
    def get_params(img : Tensor, patch_size):
        ##Get parameters for ``crop`` for a random crop.
        h = img.shape[0]
        w = img.shape[1]
        th, tw = patch_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger then input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return i, j, th, tw

    def __call__(self,imgs, seg):
        img_height = imgs[0].shape[0]
        img_width = imgs[0].shape[1]
        #imgs = torch.tensor(np.zeros(imgs.shape[0], self.size[0], self.size[1]))
        if self.pad_if_needed and img_height < self.size[0]:
            padding = [0, self.size[0] - img_height]
            #for i in range(imgs.shape[0]):
            imgs = F.pad(imgs,padding, self.fill, self.padding_mode)
            seg = F.pad(seg, padding, self.fill, self.padding_mode)

        if self.pad_if_needed and img_width < self.size[1]:
            padding = [self.size[1] - img_width, 0]
            #for i in range(len(imgs)):
            imgs = F.pad(imgs, padding, self.fill, self.padding_mode)
            seg = F.pad(seg, padding, self.fill, self.padding_mode)
        
        x,y,h,w = self.get_params(imgs[0], self.size)
        #for i in range(len(imgs)):
        imgs = F.crop(imgs, x, y, h, w)
        seg = F.crop(seg, x, y, h, w)
        return imgs, seg

class randomCrop(object):
    #def __init__(self, size: tuple[int,int], modalities : List[str], pad_if_needed : bool=True, fill : float=-1.0, padding_mode : str='constant'):
    def __init__(self, size, modalities, pad_if_needed=True, fill=-1.0, padding_mode='constant'):
        self.size = size
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
        self.modalities = modalities
    
    @staticmethod
    #def get_params(img : Tensor, patch_size : tuple[int,int]) -> Tuple[int,int,int,int]:
    def get_params(img : Tensor, patch_size):
        ##Get parameters for ``crop`` for a random crop.
        h = img.shape[0]
        w = img.shape[1]
        th, tw = patch_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger then input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return i, j, th, tw

    def __call__(self,imgs):
        img_height = imgs[0].shape[0]
        img_width = imgs[0].shape[1]
        if self.pad_if_needed and img_height < self.size[0]:
            padding = [0, self.size[0] - img_height]
            for i in range(len(imgs)):
                if self.modalities[i] == 'seg':
                    filling = 0
                else:
                    filling = self.fill
                imgs[i] = F.pad(imgs[i],padding, filling, self.padding_mode)
        if self.pad_if_needed and img_width < self.size[1]:
            padding = [self.size[1] - img_width, 0]
            for i in range(len(imgs)):
                if self.modalities[i] == 'seg':
                    filling = 0
                else:
                    filling = self.fill
                imgs[i] = F.pad(imgs[i],padding, filling, self.padding_mode)
        x,y,h,w = self.get_params(imgs[0], self.size)
        for i in range(len(imgs)):
            imgs[i] = F.crop(imgs[i],x,y,h,w)
        return imgs

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class MultiMRI2d(object):

    def __init__(self, inputfiles, num_modalities):
        assert isinstance(inputfiles, list),\
                "Class MultiMRI2d needs list of inputfiles"
        self.inputfiles = inputfiles
        self.images_handle = []
        self.num_modalities = num_modalities
        self.modal_dict = ['t2','t1ce','t1','flair', 'seg']
        #for i in range(len(self.inputfiles)):
        assert self.num_modalities == len(self.inputfiles),\
                "missing or too much inputfiles"

        #sitk_reader = sitk.ImageFileReader()
        for i in range(self.num_modalities):
            assert os.path.isfile(self.inputfiles[i]),\
                    "missing file"
            #sitk_reader.SetFileName(self.inputfiles[i])
            #image_handle = sitk_reader.Execute()
            image_handle = nib.load(self.inputfiles[i])
            self.images_handle.append(image_handle)

    def get_MRI_shape(self):
        #for i in range(self.num_modalities):
        #image_np = sitk.GetArrayFromImage(self.images_handle[0])
        shapes = self.images_handle[0].header.get_data_shape()
        return shapes

    def get_img_data(self,index):
        assert index >= 0 and index < self.num_modalities,\
                "incorrect index"
        image_data =  np.asarray(self.images_handle[index].dataobj) 
        return image_data

    def get_img_data_dict(self, modal):
        #assert index >= 0 and index < self.num_modalities,\
        #        "incorrect index"
        index = self.modal_dict.index(modal)
        image_data =  np.asarray(self.images_handle[index].dataobj)
        return image_data


class MriDataset_DA(Dataset):
    def __init__(self, dataroot , filelist, \
            input_modal, output_modal, patch_size, transform, is_train) -> None:
        super(MriDataset_DA, self).__init__()
        self.modal_dict = ['t2','t1ce','t1','flair', 'seg']
        self.dataroot = dataroot
        self.patch_size = patch_size
        self.transform = transform
        #self.transform_color = transform_color
        self.img_handlers = []
        self.slice_list = []
        self.filelist = filelist
        self.input_modal = input_modal
        self.output_modal = output_modal
        self.randcrop = randomCrop_mask(self.patch_size)
        self.istrain = is_train
        with open(os.path.join(self.dataroot, self.filelist)) as f:
            self.patientlist = f.read().splitlines()
        self.__load_images()
        self.__get_indexes()
        
    def __load_images(self):
        for patient in self.patientlist:
            img_list = []
            for i in range(len(self.modal_dict)):
                filename = os.path.join(self.dataroot, patient, patient + '_' + self.modal_dict[i] + '.nii.gz')
                assert os.path.isfile(filename), \
                        "missing file"
                img_list.append(filename)
            print(patient)
            MRI = MultiMRI2d(img_list, len(self.modal_dict))
            self.img_handlers.append(MRI)

    def __get_indexes(self):
        for handle in self.img_handlers:
            input_data_shape = handle.get_MRI_shape()
            for slice in range(input_data_shape[2]):
                sample_index = (handle,slice)
                self.slice_list.append(sample_index)

    def __getitem__(self, index):
        img_handle = self.slice_list[index][0]
        slice_index = self.slice_list[index][1]
        images = []
        for i in self.input_modal:
            image = img_handle.get_img_data_dict(i)
            image = image[...,int(slice_index)]
            image = torch.from_numpy(image)
            image = image.unsqueeze(0)
            images.append(image)
        for i in self.output_modal:
            image = img_handle.get_img_data_dict(i)
            image = image[...,int(slice_index)]
            image = torch.from_numpy(image)
            image = image.unsqueeze(0)
            images.append(image)
        images = torch.cat(images,dim=0)
        images = (images + 1) / 2
        

        segmentation = img_handle.get_img_data_dict('seg')
        segmentation = segmentation[...,int(slice_index)]
        segmentation = torch.from_numpy(segmentation)
        segmentation  = segmentation.unsqueeze(0)
        if self.istrain:
            ## randon flip ##
            if torch.rand(1) < 0.5:
                images = F.hflip(images)
                segmentation = F.hflip(segmentation)
            if torch.rand(1) < 0.5:
                images = F.vflip(images)
                segmentation = F.vflip(segmentation)
            ## random rotation ##
            d = transforms.RandomRotation.get_params([-30, 30])
            images = F.rotate(images,d, F.InterpolationMode.BILINEAR,fill=0)
            segmentation = F.rotate(segmentation, d, F.InterpolationMode.NEAREST, fill=0)
        ## random crop ##
        images, segmentation = self.randcrop(images, segmentation)

        inputs = images[0:len(self.input_modal),:,:]
        outputs = images[len(self.input_modal):len(self.input_modal) + len(self.output_modal),:,:]
        if self.istrain:
            for i in range(inputs.shape[0]):
                if torch.rand(1) < 0.2:
                    gamma = 0.8 * torch.rand(1) + 0.7
                    inputs[i,:,:] = F.adjust_gamma(inputs[i,:,:],gamma=gamma)
                inputs[i,:,:] = self.transform(inputs[i,:,:].unsqueeze(0))
        inputs[inputs<0] = 0
        inputs[inputs>1] = 1
        inputs = inputs  * 2 - 1
        outputs = outputs  * 2 - 1
        if True:
            ds_scales = [[1,1,1],[0.5,0.5],[0.25,0.25],[0.125,0.125],[0.0625,0.0625]]
            segmentation = segmentation.unsqueeze(0).numpy()
            segmentation = downsample_seg_for_ds_transform2(segmentation, ds_scales=ds_scales,order=0,axes=[2,3])
        
        input_dict = {'label': inputs,
                      'seg': segmentation,
                      'image': outputs
                      }
        return input_dict


    def __len__(self):
        return len(self.slice_list)

class MriDataset(Dataset):
    def __init__(self, dataroot , filelist, \
            input_modal, output_modal, patch_size) -> None:
        super(MriDataset, self).__init__()
        self.modal_dict = ['t2','t1ce','t1','flair', 'seg']
        #self.modal_dict = ['t2','t1ce', 'seg']
        self.dataroot = dataroot
        self.patch_size = patch_size
        self.randcrop = randomCrop(self.patch_size, self.modal_dict)
        self.transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.5),
                        transforms.RandomRotation(30, fill=-1.0)])

        self.img_handlers = []
        self.slice_list = []
        self.filelist = filelist
        self.input_modal = input_modal
        self.output_modal = output_modal
        with open(os.path.join(self.dataroot, self.filelist)) as f:
            self.patientlist = f.read().splitlines()
        self.__load_images()
        self.__get_indexes()
        
    def __load_images(self):
        for patient in self.patientlist:
            img_list = []
            for i in range(len(self.modal_dict)):
                filename = os.path.join(self.dataroot, patient, patient + '_' + self.modal_dict[i] + '.nii.gz')
                assert os.path.isfile(filename), \
                        "missing file"
                img_list.append(filename)
            print(patient)
            MRI = MultiMRI2d(img_list, len(self.modal_dict))
            self.img_handlers.append(MRI)

    def __get_indexes(self):
        for handle in self.img_handlers:
            input_data_shape = handle.get_MRI_shape()
            for slice in range(input_data_shape[2]):
                sample_index = (handle,slice)
                self.slice_list.append(sample_index)

    def __getitem__(self, index):
        img_handle = self.slice_list[index][0]
        slice_index = self.slice_list[index][1]
        images = []
        print(img_handle, slice_index)
        for i in range(len(self.modal_dict)):
            image = img_handle.get_img_data(i)
            
            image = image[...,int(slice_index)]
            image = torch.from_numpy(image)
            images.append(image)
        images = self.randcrop(images)
        img_tensors = {}
        for i in range(len(self.modal_dict)):
            if len(images[i].shape) == 2:
                images[i] = torch.unsqueeze(images[i], 0)
            img_tensors[self.modal_dict[i]] = images[i]
        input_list = []
        for i in range(len(self.input_modal)):
            input_list.append(img_tensors[self.input_modal[i]])
        inputs = torch.cat(input_list,dim=0)
        #input_dict = {'label': img_tensors[self.input_modal],
        input_dict = {'label': inputs,
                      'seg': img_tensors['seg'],
                      'image': img_tensors[self.output_modal],
                      }
        return input_dict


    def __len__(self):
        return len(self.slice_list)

if __name__=='__main__':
    modal_dict = ['t2','t1ce','seg']
    dataset_dir = '/exports/lkeb-hpc/ychen/01_data/03_preprocessed/02_bratsSyn/003_1-1norm/train_data/'
    #dataset = MRI2DTransferDataset('/exports/lkeb-hpc/ychen/01_data/03_preprocessed/02_bratsSyn/000_train/', 'patientlist_tmp.txt',modal_dict)

    #X = data["image"].type(torch.Tensor)
    #print(X.shape)
    input_modal = 't2'
    output_modal = 't1ce'
    img_height=128
    img_width=160
    train_instance = MriDataset(dataset_dir, 'patientlist.txt', input_modal, output_modal,(img_height, img_width))
    print("dataset [%s] of size %d was created" %
            (type(train_instance).__name__, len(train_instance)))
    save_dir = "/exports/lkeb-hpc/ychen/03_result/test4/"
    for i in range(500):
        data = train_instance[i]
        visuals = OrderedDict([('input_label', data['label']),('real_image', data['image'])])
        for label, image_numpy in visuals.items():
            img_path = os.path.join(save_dir, '%s_%d.png' % (label,i))
            if len(image_numpy.shape) >= 4:
                image_numpy = image_numpy[0]                    
            tmp = torch.tensor(image_numpy)                  
            save_image(tmp.float().squeeze(), img_path, normalize=True, range=(-1,1))


