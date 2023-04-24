"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from dominate.tags import input_
import torch
import models.networks as networks
from models.networks.generator import ASAPNetsGenerator, ASAPFPNGenerator, ASAPFPNGeneratorV2, ASAPfunctaGeneratorV2, ASAPfunctaGeneratorV3
#from models.networks.generator_cat import CatGenerator, ConcatsirenGenerator,ConcatGenerator_fpn, ConcatGenerator_fpn2, ConcatGenerator_fpn_multi
from models.networks.generator_vit import TransGenerator, TransGenerator_onemlp, TransGeneratorV2, TransGeneratorV3, TransGeneratorV4, AttGenerator_fpn
from models.networks.sirengenerator import Siren
import util.util as util
from scipy.ndimage.morphology import distance_transform_edt as DistTransform
import numpy as np

class Pix2PixModel(torch.nn.Module):
    def __init__(self, opt, img_size, device):
        super().__init__()
        self.opt = opt
        self.device = device
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG, self.netD = self.initialize_networks(opt, img_size)

        # set loss functions
        if opt.isTrain:
            if opt.use_gan:
                self.criterionGAN = networks.GANLoss(opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
                self.criterionFeat = torch.nn.L1Loss()
            if opt.MSE_loss:
                self.criterionMSE = torch.nn.MSELoss()
            if opt.L1_loss:
                self.criterionL1 = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            #if opt.use_vae:
            #    self.KLDLoss = networks.KLDLoss()

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, use_gan, mode):
        if mode == 'my_inference':
            with torch.no_grad():
                fake_image = self.my_generate_fake(data)
            return fake_image
        
        if mode == 'inference_grad':
            fake_image, lr_features, coords = self.netG(data)
            return fake_image, coords

        input_semantics, real_image = self.preprocess_input(data)
        #print('input_semantics', input_semantics.shape)
        #print('real_image', real_image.shape)
        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(
                input_semantics, real_image, use_gan)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image)
            return d_loss
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(real_image)
            return mu, logvar
        elif mode == 'inference':
            with torch.no_grad():
                fake_image, _, _ = self.generate_fake(input_semantics, real_image)
            return fake_image
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.isTrain and self.netD is not None:
            D_params = list(self.netD.parameters())
        else:
            D_params = None
        
        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2
        
        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2), weight_decay=1e-4)
        #optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))

        if D_params is not None:
            optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))
        else:
            optimizer_D = None
        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        if self.opt.use_gan:
            util.save_network(self.netD, 'D', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt, img_size):
        #netG = networks.define_G(opt)
        #netG= ASAPFPNGenerator(opt)
        #netG= ASAPFPNGeneratorV2(opt)
        netG= ASAPfunctaGeneratorV2(opt)
        #netG= ASAPfunctaGeneratorV3(opt)
        netG.cuda()
        netG.init_weights(opt.init_type, opt.init_variance)
        netG.print_network()
        print(netG)
        netD = networks.define_D(opt) if opt.isTrain and opt.use_gan else None
        print(netD)
        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain and opt.use_gan:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)

        return netG, netD

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # move to GPU and change data types
        #if not(self.opt.no_one_hot):
        #    data['label'] = data['label'].long()
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['seg'] = data['seg'].cuda()
            data['image'] = data['image'].cuda()
        
        # create one-hot label map
        input_semantics = data['label']
        '''
        if self.opt.no_one_hot:
            input_semantics = data['label']
        else:
            label_map = data['label']
            bs, _, h, w = label_map.size()
            nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
                else self.opt.label_nc
            #input_label = torch.tensor(np.zeros((bs,nc,h,w)))
            #input_label = input_label.to(device=self.device, dtype=torch.float32)
            input_label = self.FloatTensor(bs, nc, h, w).zero_()
            input_semantics = input_label.scatter_(1, label_map, 1.0)
        '''
        return input_semantics, data['image']

    def compute_generator_loss(self, input_semantics, real_image, use_gan):
        G_losses = {}
        fake_image, lr_features, KLD_loss = self.generate_fake(
            input_semantics, real_image)
        #if self.opt.use_vae:
        #    G_losses['KLD'] = KLD_loss
    
        if use_gan:
            pred_fake, pred_real = self.discriminate(input_semantics, fake_image, real_image)

            if not self.opt.no_adv_loss:
                G_losses['GAN'] = self.criterionGAN(pred_fake, True, for_discriminator=False)

            if not self.opt.no_ganFeat_loss:
                num_D = len(pred_fake)
                #GAN_Feat_loss = torch.tensor(np.zeros((1)))
                #GAN_Feat_loss = GAN_Feat_loss.to(device=self.device, dtype=torch.float32)
                GAN_Feat_loss = self.FloatTensor(1).fill_(0)
                for i in range(num_D):  # for each discriminator
                    # last output is the final prediction, so we exclude it
                    num_intermediate_outputs = len(pred_fake[i]) - 1
                    for j in range(num_intermediate_outputs):  # for each layer output
                        unweighted_loss = self.criterionFeat(
                            pred_fake[i][j], pred_real[i][j].detach())
                        GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
                G_losses['GAN_Feat'] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) \
                * self.opt.lambda_vgg

        if self.opt.MSE_loss:
            G_losses['MSE'] = self.criterionMSE(fake_image, real_image) \
                * self.opt.lambda_MSE
        if self.opt.L1_loss:
            G_losses['L1'] = self.criterionL1(fake_image, real_image) \
                * self.opt.lambda_L1
        '''
        if self.opt.use_weight_decay:
            print(lr_features.shape)
            lr_features_l2 = lr_features.norm(p=2)
            print('l2',lr_features_l2)
            device = lr_features_l2.device
            zero = torch.zeros(lr_features_l2.shape).to(device)
            G_losses['WD'] = self.WDLoss(lr_features_l2, zero) \
                * self.opt.lambda_WD
        '''
        if self.opt.latent_code_regularization:
            G_losses['latent_loss'] = torch.mean(lr_features ** 2) * 10
        return G_losses, fake_image

    def compute_discriminator_loss(self, input_semantics, real_image):
        D_losses = {}
        with torch.no_grad():
            fake_image, _, _ = self.generate_fake(input_semantics, real_image)
            
            noise = torch.randn(fake_image.shape) * 0.05
            fake_image_noise = fake_image + noise.to(device=torch.device('cuda'))
            fake_image_noise = fake_image_noise.detach()
            fake_image_noise.requires_grad_()
            #fake_image = fake_image.detach()
            #fake_image.requires_grad_()
        
        noise = torch.randn(real_image.shape) * 0.05
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

    def encode_z(self, real_image):
        assert self.netE is not None
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, real_image, compute_kld_loss=False):
        KLD_loss = None
        fake_image, lr_features = self.netG(input_semantics)

        return fake_image, lr_features, KLD_loss
    
    def my_generate_fake(self, data):
        z = None
        KLD_loss = None
        fake_image, lr_features = self.netG(data)
        return fake_image
    
    def generate_fake_grad(self, data):
        fake_image, lr_features,coords = self.netG(data)
        return fake_image, coords
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
        #edge = torch.zeros(t.size())
        #edge = edge.to(device=self.device, dtype=torch.bool)
        edge = self.ByteTensor(t.size()).zero_().bool()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def get_distransform(self, t):
        #edge = self.ByteTensor(t.size()).zero_().cpu()
        #edge = torch.zeros(t.size())
        #edge = edge.to(device=self.device, dtype=torch.bool)
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
