import torch
import models.networks as networks
from models.networks.generator import ConesGenerator
import util.util as util

class ConesModel(torch.nn.Module):
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

    def forward(self, data, use_gan, mode):
        if mode == 'my_inference':
            with torch.no_grad():
                fake_image = self.my_generate_fake(data)
            return fake_image
        
        if mode == 'inference_grad':
            fake_image, lr_features, coords = self.netG(data)
            return fake_image, coords

        input_semantics, real_image = self.preprocess_input(data)
        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(
                input_semantics, real_image, use_gan)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image)
            return d_loss
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
        
        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2), weight_decay=3e-5)

        if D_params is not None:
            optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))
        else:
            optimizer_D = None
        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        if self.opt.use_gan:
            util.save_network(self.netD, 'D', epoch, self.opt)

    def initialize_networks(self, opt, img_size):
        netG= ConesGenerator(opt)
        netG.cuda()
        netG.print_network()
        print(netG)
        netD = networks.define_D(opt) if opt.isTrain and opt.use_gan else None
        print(netD)
        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain and opt.use_gan:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)

        return netG, netD

    def preprocess_input(self, data):
        # move to GPU and change data types
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['image'] = data['image'].cuda()
        
        return data['label'], data['image']

    def compute_generator_loss(self, input_semantics, real_image, use_gan):
        G_losses = {}
        fake_image, lr_features, KLD_loss = self.generate_fake(
            input_semantics, real_image)
        
        if self.opt.L1_loss:
            G_losses['L1'] = self.criterionL1(fake_image, real_image) 
    
        if use_gan:
            pred_fake, pred_real = self.discriminate(input_semantics, fake_image, real_image)

            if not self.opt.no_adv_loss:
                G_losses['GAN'] = self.criterionGAN(pred_fake, True, for_discriminator=False)

            if not self.opt.no_ganFeat_loss:
                num_D = len(pred_fake)
                GAN_Feat_loss = self.FloatTensor(1).fill_(0)
                for i in range(num_D):  
                    num_intermediate_outputs = len(pred_fake[i]) - 1
                    for j in range(num_intermediate_outputs):  
                        unweighted_loss = self.criterionFeat(
                            pred_fake[i][j], pred_real[i][j].detach())
                        GAN_Feat_loss += unweighted_loss / num_D
                G_losses['GAN_Feat'] = GAN_Feat_loss

        if self.opt.MSE_loss:
            G_losses['MSE'] = self.criterionMSE(fake_image, real_image) 
        
        if self.opt.latent_code_regularization:
            G_losses['latent_loss'] = torch.mean(lr_features ** 2) 
        
        return G_losses, fake_image

    def compute_discriminator_loss(self, input_semantics, real_image):
        D_losses = {}
        with torch.no_grad():
            fake_image, _, _ = self.generate_fake(input_semantics, real_image)
            noise = torch.randn(fake_image.shape) * 0.05   ## add noise to the images
            fake_image_noise = fake_image + noise.to(device=torch.device('cuda'))
            fake_image_noise = fake_image_noise.detach()
            fake_image_noise.requires_grad_()
        
        noise = torch.randn(real_image.shape) * 0.05
        real_image_noise = real_image + noise.to(device=torch.device('cuda'))
        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image_noise, real_image_noise)
        
        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)

        return D_losses

    def generate_fake(self, input_semantics, real_image, compute_kld_loss=False):
        fake_image, lr_features = self.netG(input_semantics)

        return fake_image, lr_features, None
    
    def my_generate_fake(self, data):
        fake_image, lr_features = self.netG(data)
        return fake_image

    def discriminate(self, input_semantics, fake_image, real_image):
        assert self.netD is not None
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)
        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    def divide_pred(self, pred):
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

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
