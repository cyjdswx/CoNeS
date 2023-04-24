"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from torch.optim.lr_scheduler import LambdaLR
from models.networks.sync_batchnorm import DataParallelWithCallback
from models.pix2pix_model import Pix2PixModel
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
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt, img_size):
        self.opt = opt
        device = torch.device('cpu' if self.opt.gpu_ids == -1 else 'cuda')
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
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()
        self.g_losses = g_losses
        self.generated = generated

    def run_discriminator_one_step(self, data) -> None:
        if self.optimizer_D is not None:
            self.optimizer_D.zero_grad()
            d_losses = self.pix2pix_model(data, True, mode='discriminator')
            d_loss = sum(d_losses.values()).mean()
            d_loss.backward()
            self.optimizer_D.step()
            self.d_losses = d_losses

    def get_latest_losses(self):
        if self.opt.use_gan:
            return {**self.g_losses, **self.d_losses}
        else:
            return {**self.g_losses}
    
    def run_evalutation_during_training(self, dataloader):
        num_val = len(dataloader)
        self.pix2pix_model.eval()
        val_gan_loss = 0
        val_ganfeat_loss = 0
        val_l1_loss = 0
        generated = None
        for batch in tqdm(dataloader,total=num_val,desc='Validation round', unit='batch'):
            with torch.no_grad():
                g_losses, generated = self.pix2pix_model(batch, self.opt.use_gan, mode='generator')
                #val_gan_loss += g_losses['GAN']
                #val_ganfeat_loss += g_losses['GAN_Feat']
                #val_l1_loss += g_losses['L1']
                val_l1_loss += g_losses[self.rec_loss]

        self.pix2pix_model.train()
        
        return val_l1_loss / num_val, generated
        

    def get_latest_generated(self):
        return self.generated

    def get_latest_lr(self):
        lr_G = self.lr_scheduler_G.get_last_lr()[0]
        if self.lr_scheduler_D is not None:
            lr_D = self.lr_scheduler_D.get_last_lr()[0]
        else:
            lr_D = 0
        return lr_G, lr_D 

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
        '''
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2
            if self.optimizer_D is not None:
                for param_group in self.optimizer_D.param_groups:
                    param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
        '''
