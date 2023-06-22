"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from models.networks.sync_batchnorm import DataParallelWithCallback
from models.unet_model import UnetModel
import torch
from tqdm import tqdm
from util.util import PolyLRScheduler

class LambdaLinear:
    def __init__(self, n_epochs, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "decay start before end of the training"
        self.n_epochs = n_epochs
        self.decay_start_epoch = decay_start_epoch
    
    def step(self, epoch):
        return 1.0 - max(0, epoch - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

class UnetTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt, patch_size):
        self.opt = opt
        device = torch.device('cpu' if self.opt.gpu_ids == -1 else 'cuda')
        self.model = UnetModel(opt, device)
        self.generated = None
        self.seg = None
        if opt.isTrain:
            self.optimizer_G= self.model.create_optimizers(opt)
            #self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer_G,\
            #        lr_lambda=LambdaLinear(self.opt.niter + self.opt.niter_decay, self.opt.niter).step)
            self.lr_scheduler_G = PolyLRScheduler(self.optimizer_G, self.opt.lr, self.opt.niter + self.opt.niter_decay)
            #self.old_lr = opt.lr

    def run_generator_one_step(self, data):
        self.optimizer_G.zero_grad()
        g_losses, generated, seg = self.model(data, mode='segmentation')
        print(g_losses)
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()
        self.g_losses = g_losses
        self.generated = generated
        self.seg = seg
    
    def run_training_one_step(self, data):
        self.optimizer_G.zero_grad()
        g_losses, seg = self.model(data, mode='segmentation')
        print(g_losses)
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()
        self.g_losses = g_losses
        self.seg = seg
        return g_losses, seg
    
    def run_discriminator_one_step(self, data):
        assert self.optimizer_D is not None
        self.optimizer_D.zero_grad()
        d_losses = self.model(data, mode='discriminator')
        d_loss = sum(d_losses.values()).mean()
        d_loss.backward()
        self.optimizer_D.step()
        self.d_losses = d_losses

    def get_latest_losses(self):
        return {**self.g_losses}
    
    def run_evalutation_during_training(self, dataloader):
        num_val = len(dataloader)
        fake_img = None
        val_seg_loss = 0
        
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
        #for batch in tqdm(dataloader,total=num_val,desc='Validation round', unit='batch'):
        #    with torch.no_grad():
                g_losses, seg = self.model(batch, mode='segmentation')
                val_seg_loss += g_losses['seg']
        self.model.train()
        
        return val_seg_loss/ num_val, fake_img
        

    def get_latest_generated(self):
        return self.seg

    def save(self, epoch):
        #self.pix2pix_model_on_one_gpu.save(epoch)
        self.model.save(epoch)

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
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
