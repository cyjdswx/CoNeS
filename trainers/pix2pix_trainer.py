"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from torch.optim.lr_scheduler import LambdaLR
from models.networks.sync_batchnorm import DataParallelWithCallback
from models.pix2pix_model import Pix2PixModel
from torch.cuda.amp import GradScaler
import torch
from tqdm import tqdm
import os,sys


class LambdaLinear:
    def __init__(self, n_epochs, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "decay start before end of the training"
        self.n_epochs = n_epochs
        self.decay_start_epoch = decay_start_epoch
    
    def step(self, epoch):
        return 1.0 - max(0, epoch - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

class Pix2PixTrainer():
    def __init__(self, opt, img_size):
        self.opt = opt
        device = torch.device('cpu' if self.opt.gpu_ids == -1 else 'cuda')
        self.scaler = GradScaler()
        if opt.L1_loss:
            self.rec_loss = 'L1'
        elif opt.MSE_loss:
            self.rec_loss = 'MSE'
        self.pix2pix_model = Pix2PixModel(opt,img_size, device)
        self.generated = None

        if opt.isTrain:
            self.optimizer_G, self.optimizer_D = self.pix2pix_model.create_optimizers(opt)
            self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer_G,\
                    lr_lambda=LambdaLinear(self.opt.niter + self.opt.niter_decay, self.opt.niter).step)

            if self.optimizer_D is not None:
                self.lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer_D,\
                    lr_lambda=LambdaLinear(self.opt.niter + self.opt.niter_decay, self.opt.niter).step)
            else:
                self.lr_scheduler_D = None
        
    def run_generator_one_step(self, data, use_gan) -> None:
        self.optimizer_G.zero_grad()
        g_losses, generated = self.pix2pix_model(data, use_gan, mode='generator')
        print(g_losses)
        g_loss = self.opt.lambda_L1 * g_losses['L1'] + g_losses['GAN'] + self.opt.lambda_feat * g_losses['GAN_Feat'] + \
                self.opt.lambda_ll * g_losses['latent_loss']
        
        self.scaler.scale(g_loss).backward()
        self.scaler.step(self.optimizer_G)
        self.scaler.update()
        return g_losses, generated

    def run_discriminator_one_step(self, data) -> None:
        if self.optimizer_D is not None:
            self.optimizer_D.zero_grad()
            d_losses = self.pix2pix_model(data, True, mode='discriminator')
            d_loss = d_losses['D_Fake'] + d_losses['D_real']
            self.scaler.scale(d_loss).backward()
            self.scaler.step(self.optimizer_D)
            self.scaler.update()
    
    def run_evalutation_during_training(self, dataloader):
        num_val = len(dataloader)
        self.pix2pix_model.eval()
        val_rec_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                g_losses, generated = self.pix2pix_model(batch, self.opt.use_gan, mode='generator')
                val_rec_loss += g_losses[self.rec_loss]

        self.pix2pix_model.train()
        return val_rec_loss / num_val

    def save(self, epoch):
        self.pix2pix_model.save(epoch)
        state = {'epochs': epoch, 'n_epochs': self.opt.niter + self.opt.niter_decay}
        state['netG_opt']= self.optimizer_G.state_dict()
        if self.opt.use_gan and self.optimizer_D is not None:
            state['netD_opt'] = self.optimizer_D.state_dict()
        torch.save(state, os.path.join(self.opt.checkpoints_dir, self.opt.name, 'checkpoint.pth.tar'))

    def update_learning_rate(self, epoch):
        self.lr_scheduler_D.step()
        self.lr_scheduler_G.step()
