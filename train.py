import sys,os
import shutil

from data.mri_dataset import MriDataset_DA
#from trainers.pix2pix_trainer import Pix2PixTrainer
from trainers.cones_trainer import ConesTrainer
import torch
import time
from torch.backends import cudnn
from torch.utils.data import DataLoader
import json
import random
import numpy as np
from torchvision import transforms
import argparse
import wandb

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., p=0.3):
        self.std = std
        self.mean = mean
        self.p = p
        
    def __call__(self, tensor):
        if torch.rand(1) < self.p:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        else:
            return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

if __name__=='__main__':
    # parse options
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # experiment specifics
    parser.add_argument('--name', type=str, default='brats_t1ce', help='name of the experiment')
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
    parser.add_argument('--label_nc', type=int, default=3, help='# of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.')
    parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')

    # Hyperparameters
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

    parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')
    parser.add_argument('--use_gan', action='store_true', help='enable training with an image encoder.')

    # for training
    parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
    parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
    parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
    parser.add_argument('--no_TTUR', action='store_true', help='Use TTUR training scheme')

    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')

    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
    parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss')
    parser.add_argument('--lambda_MSE', type=float, default=10.0, help='weight for MSE loss')
    parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
    parser.add_argument('--lambda_ll', type=float, default=10.0, help='weight for L1 loss')
    parser.add_argument('--no_adv_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
    parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
    parser.add_argument('--MSE_loss', action='store_true', help='if specified, use MSE loss')
    parser.add_argument('--L1_loss', action='store_true', help='if specified, use L1 loss')
    parser.add_argument('--latent_code_regularization', action='store_true', help='if specified, use weight decay loss on the estimated parameters from LR')
    parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
    
    # for discriminators
    parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale|image)')
    parser.add_argument('--lambda_kld', type=float, default=0.05)
    parser.add_argument('--netD_subarch', type=str, default='n_layer', help='architecture of each discriminator')
    parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to be used in multiscale')
    parser.add_argument('--n_layers_D', type=int, default=4, help='# layers in each discriminator')
    parser.add_argument('--ndf_max', type=int, default=512, help='maximal number of discriminator filters')

    # print options to help debugging
    opt = parser.parse_args()
    opt.isTrain = True   # train or test
    setup_seed(100)
    
    #################### set gpu ids  ####################
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])

    assert len(opt.gpu_ids) == 0 or opt.batchSize % len(opt.gpu_ids) == 0, \
            "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
            % (opt.batchSize, len(opt.gpu_ids))
    #################### print configs ###################
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)
    print(' '.join(sys.argv))
    ################ load the dataset ###################
    with open(opt.config_file) as json_data:
        try:
            configs = json.load(json_data)
            print(configs)
        except json.JSONDecodeError:
            print('invalid json format')
            sys.exit()
    
    wandb.login()
    wandb.init(
            project="Image translation&segmentation",
            name=opt.name,
            config=vars(opt)
            )
    
    dataset_dict = configs['dataset']
    img_height = dataset_dict['img_height']
    img_width = dataset_dict['img_width']
    data_root = dataset_dict['dataset_dir']
    input_modal = dataset_dict['input_modalities']
    output_modal = dataset_dict['output_modality']
    modal_dict = dataset_dict['modal_dict']
    transform_tr = transforms.Compose([
                        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.3),
                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5,sigma=(0.5,1.0))], p=0.3),
                        AddGaussianNoise(0,0.02)])
    
    train_dataroot = os.path.join(data_root, 'train_data')

    train_instance = MriDataset_DA(train_dataroot, 'patientlist.txt',modal_dict, \
                    input_modal, output_modal,(img_height, img_width), False, transform_tr, True)
    
    print("dataset [%s] of size %d was created" %
            (type(train_instance).__name__, len(train_instance)))
    dataloader = DataLoader(
        train_instance,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.nThreads),
        drop_last=opt.isTrain
    )
    ''' 
    val_dataroot = os.path.join(data_root, 'valid_data')
    val_instance = MriDataset_noseg(val_dataroot, 'patientlist_valid.txt', modal_dict, \
                    input_modal, output_modal,(img_height, img_width), False, None,False)
    print("dataset [%s] of size %d was created" %
            (type(val_instance).__name__, len(val_instance)))

    val_dataloader = DataLoader(
        val_instance,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.nThreads),
        drop_last=opt.isTrain
    )
    '''
    #### create checkpoints directory ####
    if not os.path.isdir(opt.checkpoints_dir):
        os.mkdir(opt.checkpoints_dir)
    experiment_dir = os.path.join(opt.checkpoints_dir, opt.name)
    if not os.path.isdir(experiment_dir):
        os.mkdir(experiment_dir)

    datajson_dir = os.path.join(opt.checkpoints_dir, opt.name, 'data.json')
    shutil.copy(opt.config_file, datajson_dir)
    
    # initialize trainer
    trainer = ConesTrainer(opt, (img_height, img_width))
    batches_per_epoch = 500

    if opt.use_gan:
        use_gan = True
    else:
        use_gan = False
    
    dataiter = iter(dataloader)
    total_epochs = opt.niter + opt.niter_decay
    start_epoch = 1
    for epoch in range(1, total_epochs + 1):
        epoch_start_time = time.time()
        print("epoch:%d" % epoch)
        rec_loss = 0
        latent_loss = 0
        gan_loss = 0
        gan_feat_loss = 0
        for i in range(batches_per_epoch):
            try:
                data_i = next(dataiter)
            except StopIteration:
                dataiter = iter(dataloader)
                data_i = next(dataiter)
            # train generator
            g_losses, pred_img = trainer.run_generator_one_step(data_i, use_gan)
            rec_loss += g_losses['L1']
            gan_loss += g_losses['GAN']
            gan_feat_loss += g_losses['GAN_Feat']
            latent_loss += g_losses['latent_loss']
            # train discriminator
            if use_gan:
                trainer.run_discriminator_one_step(data_i)
        
        trainer.update_learning_rate(epoch)
        print('updating learning rate:lr_G:%f, lr_D:%f' % \
                (trainer.lr_scheduler_G.get_last_lr()[0], trainer.lr_scheduler_D.get_last_lr()[0]))
        
        logs = {}
        logs['rec_loss'] = rec_loss / batches_per_epoch
        logs['latent_loss'] = latent_loss / batches_per_epoch
        logs['GAN'] = gan_loss / batches_per_epoch
        logs['GAN_Feat'] = gan_feat_loss / batches_per_epoch

        #val_rec_loss = trainer.run_evalutation_during_training(val_dataloader)
        #logs['val_rec_loss'] = val_rec_loss
        
        logs['synthesized_image'] = wandb.Image(pred_img)
        logs['real_image'] = wandb.Image(data_i['image'])
        for i in range(data_i['label'].shape[1]):
            logs['input_img_'+str(i)] = wandb.Image(data_i['label'][:,i,:,:].unsqueeze(1))
        
        wandb.log(logs)
        wandb.log
        

        time_per_epoch = time.time() - epoch_start_time
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, total_epochs, time_per_epoch))
        trainer.save(epoch)
        print('saving the latest model (epoch %d)' % epoch)
        trainer.save('latest')
    print('Training was successfully finished.')
    
    wandb.finish()
