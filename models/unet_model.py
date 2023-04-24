"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import models.networks as networks
from models.networks.segmentation import ASAPNetsSeg, ASAPNetsMultiSeg, InitWeights_He
from models.networks.segmentation import SegNet
import util.util as util
import torch.nn.functional as F
from scipy.ndimage.morphology import distance_transform_edt as DistTransform
from models.networks.loss import DC_and_CE_loss, MultipleOutputLoss2
import numpy as np
class UnetModel(torch.nn.Module):
    def __init__(self, opt, device):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            '''
            if opt.MSE_loss:
                self.criterionMSE = torch.nn.MSELoss()
            if opt.L1_loss:
                self.criterionL1 = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            '''
            self.criterionseg = torch.nn.CrossEntropyLoss()
            self.loss = DC_and_CE_loss({'batch_dice':True,'smooth':1e-5, 'do_bg':False},{})
            # we need to know the number of outputs of the network
            net_numpool = 5

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.segloss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode):
        if mode == 'inference':
            with torch.no_grad():
                seg = self.generate_seg(data)
            return seg
        else:
            input_semantics, real_image, mask = self.preprocess_input(data)
        
        if mode == 'generator':
            g_loss, generated, seg = self.compute_generator_loss(
                input_semantics, real_image, mask)
            return g_loss, generated, seg
        elif mode == 'segmentation':
            g_loss, seg = self.compute_segmentation_loss(
                input_semantics, mask)
            return g_loss, seg
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image)
            return d_loss
        else:
            raise ValueError("|mode| is invalid")

    def generate_seg(self, input):
        seg = self.netG(input)
        return seg[0]

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        '''
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2
        '''
        #optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_G = torch.optim.Adam(G_params, lr=opt.lr, betas=(beta1, beta2), weight_decay=1e-4)
        '''
        if D_params is not None:
            #optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2), weight_decay=1e-4)
            optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))
        else:
            optimizer_D = None
        '''
        return optimizer_G

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        #util.save_network(self.netD, 'D', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        #netG = ASAPNetsMultiSeg(opt,n_classes=4)
        norm_op_kwargs = {'eps':1e-05, 'affine':True}
        dropout_op_kwargs = {'p':0, 'inplace':True}
        net_nonline_kwargs = {'negative_slope': 0.01, 'inplace': True}
        num_pool_op_kernel_size = [[2,2],[2,2],[2,2],[2,2],[2,2]]
        net_conv_kernel_size = [[3,3],[3,3],[3,3],[3,3],[3,3],[3,3]]
        netG = SegNet(input_channels=3, base_num_features=32, num_classes=4, num_pool=5,num_conv_per_stage=2,\
                feat_map_mul_on_downscale=2,conv_op=torch.nn.Conv2d, norm_op=torch.nn.InstanceNorm2d, norm_op_kwargs=norm_op_kwargs,\
                dropout_op=torch.nn.Dropout2d,dropout_op_kwargs=dropout_op_kwargs, nonlin=torch.nn.LeakyReLU,\
                nonlin_kwargs=net_nonline_kwargs,deep_supervision=True, dropout_in_localization=False,final_nonlin=lambda x:x,\
               weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=num_pool_op_kernel_size, conv_kernel_sizes=net_conv_kernel_size,\
               upscale_logits=False,convolutional_pooling=True,convolutional_upsampling=True)
        if len(opt.gpu_ids) > 0:
            assert(torch.cuda.is_available())
            netG.cuda()
        #netG.init_weights(opt.init_type, opt.init_variance)
        print(netG)
        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)

        return netG#, netD

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # move to GPU and change data types
        #if not(self.opt.no_one_hot):
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            for i in range(len(data['seg'])):
                data['seg'][i] = data['seg'][i].cuda()
            data['image'] = data['image'].cuda()
        input_semantics = data['label']
        #data['seg'] = data['seg'].to(dtype=torch.long).squeeze(dim=1)
        return input_semantics, data['image'], data['seg']
    
    def compute_segmentation_loss(self, input_semantics,  mask):
        G_losses = {}
        seg = self.netG(input_semantics)

        ##segmentation
        #seg_los = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth':1e-5,'dp_bg':False},{})
        #G_losses['seg'] = (self.criterionseg(seg, mask) + \
        #            networks.dice_loss(F.softmax(seg, dim=1).float(), \
        #            F.one_hot(mask, 4).permute(0,3,1,2).float(), multiclass=True))
        G_losses['seg'] = self.segloss(seg, mask)
        return G_losses,  seg
    
    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image):
        #print(input_semantics.shape)
        #print(fake_image.shape)
        #print(real_image.shape)
        assert self.netD is not None
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def get_edges(self, t):
        #edge = self.ByteTensor(t.size()).zero_()
        edge = self.ByteTensor(t.size()).zero_().bool()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
