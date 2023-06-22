"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import models.networks as networks
from models.networks.generator import ASAPfunctaGeneratorV3
from models.networks.segmentation import InitWeights_He, SegNet, ASAPNetsMultiSeg_nnunet_feat, Synseg_vanillaunet, Synseg_vanillaunet_feat
import util.util as util
import torch.nn.functional as F
from scipy.ndimage.morphology import distance_transform_edt as DistTransform
from models.networks.loss import DC_and_CE_loss, MultipleOutputLoss2
import numpy as np

class PixSegModel(torch.nn.Module):
    def __init__(self, opt, device):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor
        self.deep_supervision = opt.deep_supervision
        self.netG, self.netD = self.initialize_networks(opt)
        #self.netG, self.netS, self.netD = self.initialize_networks_tri()(opt)

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if opt.MSE_loss:
                self.criterionMSE = torch.nn.MSELoss()
            if opt.L1_loss:
                self.criterionL1 = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            #self.criterionseg = torch.nn.CrossEntropyLoss()
            
            self.loss = DC_and_CE_loss({'batch_dice':True,'smooth':1e-5, 'do_bg':False},{})
            if opt.deep_supervision:
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
            else:
                self.segloss = self.loss
    
    def forward(self, data, mode):
        if mode == 'inference':
            with torch.no_grad():
                fake_image, seg, _ = self.generate_fake(data)
            return fake_image, seg
        else:
            data['label'] = data['label'].cuda()
            if self.deep_supervision:
                for i in range(len(data['seg'])):
                    data['seg'][i] = data['seg'][i].cuda()
            else:
                data['seg'] = data['seg'].cuda()
            data['image'] = data['image'].cuda()
        
        if mode == 'generator':
            #g_loss, generated, seg = self.compute_generator_loss(
            #    input_image, real_image, mask)
            g_loss, generated, seg = self.compute_generator_loss(
                data['label'], data['image'], data['seg'])
            return g_loss, generated, seg
        elif mode == 'discriminator':
            #d_loss = self.compute_discriminator_loss(
            #    input_image, real_image)
            d_loss = self.compute_discriminator_loss(
                data['label'], data['image'])
            return d_loss
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        #G_params = list(self.netG.parameters())
        #S_params = None
        G_params = list(self.netG.lowres_stream.parameters()) + \
                list(self.netG.latlayers.parameters()) + list(self.netG.highres_stream.parameters())
        S_params = list(self.netG.lowres_stream.parameters()) + \
                list(self.netG.latlayers.parameters()) + list(self.netG.highres_stream.parameters()) + list(self.netG.seg_stream.parameters())
        if opt.isTrain and self.netD is not None:
            D_params = list(self.netD.parameters())
        else:
            D_params = None

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2), weight_decay=3e-5)
        if S_params is not None:
            #optimizer_S = torch.optim.Adam(S_params, lr=opt.lr, betas=(beta1,beta2), weight_decay=3e-5)
            optimizer_S = torch.optim.SGD(S_params, lr=0.05, momentum=0.9, weight_decay=3e-5)
        else:
            optimizer_S = None
        if D_params is not None:
            optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))
        else:
            optimizer_D = None
        
        return optimizer_G, optimizer_D, optimizer_S

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        #util.save_network(self.netS, 'S', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        #netG = ASAPNetsMultiSeg_nnunet(opt,n_classes=4, use_dropout=True)
        #netG = ASAPNetsMultiSeg_vanillaunet_feat(opt,n_classes=4, dropout=False)
        #netG = Synseg_vanillaunet(opt,n_classes=4, dropout=False)
        netG = Synseg_vanillaunet_feat(opt,n_classes=4, dropout=False)
        netG.print_network()
        if len(opt.gpu_ids) > 0:
            assert(torch.cuda.is_available())
            netG.cuda()
        print(netG)
        netD = networks.define_D(opt) if opt.isTrain else None
        print(netD)
        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)

        if opt.pretrained is not None:
            print("load pretrained synthetic network from:", opt.pretrained)
            weights = torch.load(opt.pretrained)
            netG.load_state_dict(weights, strict=False)
        return netG, netD
    
    def initialize_networks_tri(self, opt):
        netG = ASAPNetsMultiSeg_nnunet_feat(opt,n_classes=4)
        #netG = ASAPfunctaGeneratorV3(opt)
        netG.print_network()
        if len(opt.gpu_ids) > 0:
            assert(torch.cuda.is_available())
            netG.cuda()
        print(netG)
        norm_op_kwargs = {'eps':1e-05, 'affine':True}
        dropout_op_kwargs = {'p':0, 'inplace':True}
        net_nonline_kwargs = {'negative_slope': 0.01, 'inplace': True}
        num_pool_op_kernel_size = [[2,2],[2,2],[2,2],[2,2],[2,2]]
        net_conv_kernel_size = [[3,3],[3,3],[3,3],[3,3],[3,3],[3,3]]
        netS = SegNet(input_channels=4+256, base_num_features=32, num_classes=4, num_pool=5,num_conv_per_stage=2,\
                feat_map_mul_on_downscale=2,conv_op=torch.nn.Conv2d, norm_op=torch.nn.InstanceNorm2d, norm_op_kwargs=norm_op_kwargs,\
                dropout_op=torch.nn.Dropout2d,dropout_op_kwargs=dropout_op_kwargs, nonlin=torch.nn.LeakyReLU,\
                nonlin_kwargs=net_nonline_kwargs,deep_supervision=True, dropout_in_localization=False,final_nonlin=lambda x:x,\
               weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=num_pool_op_kernel_size, conv_kernel_sizes=net_conv_kernel_size,\
               upscale_logits=False,convolutional_pooling=True,convolutional_upsampling=True)
        if len(opt.gpu_ids) > 0:
            assert(torch.cuda.is_available())
            netS.cuda()

        netD = networks.define_D(opt) if opt.isTrain else None
        print(netD)
        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)

        return netG, netS, netD
    '''
    def preprocess_input(self, data):
        # move to GPU and change data types
        #if not(self.opt.no_one_hot):
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            #for i in range(len(data['seg'])):
            #    data['seg'][i] = data['seg'][i].cuda()
            data['seg'] = data['seg'].cuda()
            data['image'] = data['image'].cuda()
        input_semantics = data['label']
        return input_semantics, data['image'], data['seg']
    '''
    def compute_generator_loss(self, input_images, real_image, mask):
        G_losses = {}
        fake_image, seg, lr_features = self.generate_fake(input_images)
        
        if self.opt.L1_loss:
            G_losses['L1'] = self.criterionL1(fake_image, real_image)
        
        if self.opt.use_gan:
            pred_fake, pred_real = self.discriminate(input_images, fake_image, real_image)

            if not self.opt.no_adv_loss:
                G_losses['GAN'] = self.criterionGAN(pred_fake, True, for_discriminator=False)

            if not self.opt.no_ganFeat_loss:
                num_D = len(pred_fake)
                GAN_Feat_loss = self.FloatTensor(1).fill_(0)
                for i in range(num_D):  # for each discriminator
                    # last output is the final prediction, so we exclude it
                    num_intermediate_outputs = len(pred_fake[i]) - 1
                    for j in range(num_intermediate_outputs):  # for each layer output
                        unweighted_loss = self.criterionFeat(
                            pred_fake[i][j], pred_real[i][j].detach())
                        GAN_Feat_loss += unweighted_loss / num_D
                G_losses['GAN_Feat'] = GAN_Feat_loss

        
        ##segmentation
        G_losses['seg'] = self.segloss(seg, mask) 
        
        if self.opt.latent_code_regularization:
            G_losses['latent_loss'] = torch.mean(lr_features ** 2)

        if self.opt.consistency_loss:
            G_losses['consistency_loss'] = 0
        
        return G_losses, fake_image, seg

    def compute_discriminator_loss(self, input_semantics, real_image):
        D_losses = {}
        with torch.no_grad():
            fake_image, _, _ = self.generate_fake(input_semantics)
            
            noise = torch.randn(fake_image.shape) * 0.01
            fake_image_noise = fake_image + noise.to(device=torch.device('cuda'))
            fake_image_noise = fake_image_noise.detach()
            fake_image_noise.requires_grad_()
            #fake_image = fake_image.detach()
            #fake_image.requires_grad_()
        
        noise = torch.randn(real_image.shape) * 0.01
        real_image_noise = real_image + noise.to(device=torch.device('cuda'))
        #pred_fake, pred_real = self.discriminate(
        #    input_semantics, fake_image, real_image)
        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image_noise, real_image_noise)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)

        return D_losses

    def generate_fake(self, input_semantics):
        fake_image, mask, lr_features = self.netG(input_semantics)
        #fake_image, lr_features = self.netG(input_semantics)
        #seg_input = torch.cat([input_semantics, lr_features], dim=1)
        #mask = self.netS(seg_input)
        return fake_image, mask, lr_features
    '''
    def generate_and_seg(self, input_semantics, real_image)
        fake_image, lr_features = self.netG(input_semantics)
        seg_input = torch.cat([input_semantics,])
    '''
    def my_generate_fake(self, data):
        fake_image, mask_pred, lr_features = self.netG(data)
        return fake_image, mask_pred
    
    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image):
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

    def get_distransform(self, t):
        #edge = self.ByteTensor(t.size()).zero_().cpu()
        edge = self.ByteTensor(t.size()).zero_().bool().cpu()
        device = t.device
        t = t.cpu()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        distransform = DistTransform(edge)
        distransform = torch.from_numpy(distransform).float().to(device)
        return distransform.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
