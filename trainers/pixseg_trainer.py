"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from models.networks.sync_batchnorm import DataParallelWithCallback
from models.pixseg_model import PixSegModel
from torch.cuda.amp import GradScaler
import torch
from tqdm import tqdm

class LambdaLinear:
    def __init__(self, n_epochs, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "decay start before end of the training"
        self.n_epochs = n_epochs
        self.decay_start_epoch = decay_start_epoch
    
    def step(self, epoch):
        return 1.0 - max(0, epoch - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

class PixSegTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt, img_size):
        self.opt = opt
        device = torch.device('cpu' if self.opt.gpu_ids == -1 else 'cuda')
        self.model = PixSegModel(opt, device)
        self.scaler = GradScaler()
        self.generated = None
        self.seg = None
        if opt.isTrain:
            self.optimizer_G, self.optimizer_D, self.optimizer_S = self.model.create_optimizers(opt)
            #self.optimizer_G, self.optimizer_D = self.model.create_optimizers(opt)
            #self.optimizer_S = None
            self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer_G,\
                    lr_lambda=LambdaLinear(self.opt.niter + self.opt.niter_decay, self.opt.niter).step)

            if self.optimizer_S is not None:
                #self.lr_scheduler_S= torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer_S,\
                #    lr_lambda=LambdaLinear(self.opt.niter + self.opt.niter_decay, 0).step)

                self.lr_scheduler_S = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer_S,step_size=50, gamma=0.1)
            else:
                self.lr_scheduler_S = None
            if self.optimizer_D is not None:
                self.lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer_D,\
                    lr_lambda=LambdaLinear(self.opt.niter + self.opt.niter_decay, self.opt.niter).step)
            else:
                self.lr_scheduler_D = None

            print(self.optimizer_G, self.optimizer_S, self.optimizer_D)

    def run_generator_one_step(self, data):
        self.optimizer_G.zero_grad()
        if self.optimizer_S is not None:
            self.optimizer_S.zero_grad()
        g_losses, generated, seg = self.model(data, mode='generator')
        print(g_losses)
        g_loss = self.opt.lambda_L1 * g_losses['L1'] + g_losses['GAN'] + self.opt.lambda_feat * g_losses['GAN_Feat'] + \
                self.opt.lambda_seg * g_losses['seg'] + self.opt.lambda_ll * g_losses['latent_loss']
        #g_loss = self.opt.lambda_seg * g_losses['seg'] + self.opt.lambda_ll * g_losses['latent_loss']
        self.scaler.scale(g_loss).backward()
        self.scaler.step(self.optimizer_G)
        if self.optimizer_S is not None:
            self.scaler.step(self.optimizer_S)
        self.scaler.update()
        return g_losses, generated, seg

    def run_discriminator_one_step(self, data):
        assert self.optimizer_D is not None
        self.optimizer_D.zero_grad()
        d_losses = self.model(data, mode='discriminator')
        #d_loss = sum(d_losses.values()).mean()
        d_loss = d_losses['D_Fake'] + d_losses['D_real']
        self.scaler.scale(d_loss).backward()
        self.scaler.step(self.optimizer_D)
        self.scaler.update()
    
    def run_evalutation_during_training(self, dataloader):
        num_val = len(dataloader)
        fake_img = None
        val_seg_loss = 0
        val_rec_loss = 0
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                g_losses, fake_img, seg = self.model(batch, mode='generator')
                val_seg_loss += g_losses['seg']
                val_rec_loss += g_losses['L1']
        self.model.train()
        return val_seg_loss/ num_val, val_rec_loss/ num_val, fake_img

    def save(self, epoch):
        #self.pix2pix_model_on_one_gpu.save(epoch)
        self.model.save(epoch)

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        self.lr_scheduler_D.step()
        self.lr_scheduler_G.step()
        if self.lr_scheduler_S is not None:
            self.lr_scheduler_S.step()
