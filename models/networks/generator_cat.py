from dominate.tags import a
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ASAPNetsBlock as ASAPNetsBlock
from models.networks.architecture import MySeparableBilinearDownsample as BilinearDownsample
from models.networks.sirengenerator import Siren
import torch.nn.utils.spectral_norm as spectral_norm
import torch as th
from math import pi
from math import log2
import time
import numpy as np

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)


def _get_coords(bs, h, w, device, ds):
    """Creates the position encoding for the pixel-wise MLPs"""
    print('01global',ds)
    f0 = 0
    while f0 <= ds:
        f = pow(2, f0)
        x = th.arange(0, w).float()
        y = th.arange(0, h).float()
        xcos = th.cos(f * pi * x.float() / w)
        xsin = th.sin(f * pi * x.float() / w)
        ycos = th.cos(f * pi * y.float() / h)
        ysin = th.sin(f * pi * y.float() / h)
        xcos = xcos.view(1, 1, 1, w).repeat(bs, 1, h, 1)
        xsin = xsin.view(1, 1, 1, w).repeat(bs, 1, h, 1)
        ycos = ycos.view(1, 1, h, 1).repeat(bs, 1, 1, w)
        ysin = ysin.view(1, 1, h, 1).repeat(bs, 1, 1, w)
        coords_cur = th.cat([xcos, xsin, ycos, ysin], 1).to(device)
        if f0 == 0:
            coords = coords_cur
        else:
            coords = th.cat([coords, coords_cur], 1).to(device)    # pyright: ignore
        f0 = f0 + 1
    return coords.to(device)    # pyright:ignore

class CatGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='instanceaffine')
        parser.set_defaults(no_instance_dist=True)
        return parser

    def __init__(self, opt, img_size=(128,160), hr_stream=None, lr_stream=None):
        super(CatGenerator, self).__init__()
        if lr_stream is None or hr_stream is None:
            lr_stream = dict()
            hr_stream = dict()
        self.num_inputs = opt.label_nc
        self.lr_instance = opt.lr_instance
        self.num_params = opt.latent_dim
        self.learned_ds_factor = opt.learned_ds_factor #(S2 in sec. 3.2)
        self.gpu_ids = opt.gpu_ids

        self.img_size = img_size
        
        # calculates the total downsampling factor in order to get the final low-res grid of parameters (S=S1xS2 in sec. 3.2)
        #self.downsampling = opt.crop_size // (16 * opt.aspect_ratio)
        #self.downsampling = min(img_size) // opt.lowest_ds_factor
        self.downsampling = opt.ds_factor
        norm_layer = get_nonspade_norm_layer(opt,'batch')
        #norm_layer = nn.BatchNorm2d
        
        self.encoder = CatEncoderV2(self.downsampling, self.num_inputs, self.num_params, self.img_size, norm_layer)
        #self.decoder = CatDecoderV2(self.downsampling, num_inputs=self.num_inputs, latent_dim=self.num_params,
        #                                       num_outputs=opt.output_nc, width=opt.hr_width,
        #                                       depth=opt.hr_depth, coordinates=opt.hr_coor)
        self.decoder = CatDecoder(self.downsampling, img_size, num_inputs=self.num_inputs, latent_dim=self.num_params,
                                               num_outputs=opt.output_nc, width=opt.hr_width,
                                               depth=opt.hr_depth, coordinates=opt.hr_coor)
        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)
    

    def use_gpu(self):
        return len(self.gpu_ids) > 0

    def forward(self, input):
        features = self.encoder(input)
        output = self.decoder(input, features)
        return output, features#, lowres

class ConcatsirenGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='instanceaffine')
        parser.set_defaults(no_instance_dist=True)
        return parser

    def __init__(self, opt, img_size=(128,160), hr_stream=None, lr_stream=None):
        super(ConcatsirenGenerator, self).__init__()
        if lr_stream is None or hr_stream is None:
            lr_stream = dict()
            hr_stream = dict()
        self.num_inputs = opt.label_nc
        self.lr_instance = opt.lr_instance
        self.num_params = opt.latent_dim
        self.num_siren_in = self.num_inputs + self.num_params + 2 # input image  + latentcode + coordinates(2)
        #self.num_siren_in = self.num_inputs + 2 # input image  + latentcode + coordinates(2)
        self.learned_ds_factor = opt.learned_ds_factor #(S2 in sec. 3.2)
        self.gpu_ids = opt.gpu_ids

        self.img_size = img_size
        
        # calculates the total downsampling factor in order to get the final low-res grid of parameters (S=S1xS2 in sec. 3.2)
        #self.downsampling = opt.crop_size // (16 * opt.aspect_ratio)
        #self.downsampling = min(img_size) // opt.lowest_ds_factor
        self.downsampling = opt.ds_factor
        norm_layer = get_nonspade_norm_layer(opt,'batch')
        #norm_layer = nn.BatchNorm2d
        
        self.encoder = CatEncoderV2(self.downsampling, self.num_inputs, self.num_params, self.img_size, norm_layer)
        
        self.decoder = Siren(in_features=self.num_siren_in, out_features=opt.output_nc, hidden_features=opt.hr_width, hidden_layers=opt.hr_depth,
                            outermost_linear=False)
        self.encoder.apply(init_weights)
        #self.decoder.apply(init_weights)
    

    def use_gpu(self):
        return len(self.gpu_ids) > 0

    def forward(self, input):
        b,_,h,w = input.shape
        features = self.encoder(input)
        _,_,lr_h,lr_w = features.shape
        features_upsampled = features.repeat_interleave(int(h/lr_h), dim=2)
        features_upsampled = features_upsampled.repeat_interleave(int(w/lr_w), dim=3)
        output, coords = self.decoder(input, features_upsampled)
        return output, features#, coords

class CatGenerator_pyramid(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='instanceaffine')
        parser.set_defaults(no_instance_dist=True)
        return parser

    def __init__(self, opt, img_size=(128,160), hr_stream=None, lr_stream=None):
        super(CatGenerator_pyramid, self).__init__()
        if lr_stream is None or hr_stream is None:
            lr_stream = dict()
            hr_stream = dict()
        self.num_inputs = opt.label_nc
        self.lr_instance = opt.lr_instance
        self.num_params = opt.latent_dim
        self.learned_ds_factor = opt.learned_ds_factor #(S2 in sec. 3.2)
        self.gpu_ids = opt.gpu_ids

        self.img_size = img_size
        
        # calculates the total downsampling factor in order to get the final low-res grid of parameters (S=S1xS2 in sec. 3.2)
        #self.downsampling = opt.crop_size // (16 * opt.aspect_ratio)
        #self.downsampling = min(img_size) // opt.lowest_ds_factor
        self.downsampling = opt.ds_factor
        norm_layer = get_nonspade_norm_layer(opt,'batch')
        #norm_layer = nn.BatchNorm2d
        
        self.encoder = CatEncoder_muls(self.downsampling, self.num_inputs, self.num_params, self.img_size, norm_layer)
        #self.decoder = CatDecoderV2(self.downsampling, num_inputs=self.num_inputs, latent_dim=self.num_params,
        #                                       num_outputs=opt.output_nc, width=opt.hr_width,
        #                                       depth=opt.hr_depth, coordinates=opt.hr_coor)
        self.decoder = CatDecoder(self.downsampling, img_size, num_inputs=self.num_inputs, latent_dim=self.num_params,
                                               num_outputs=opt.output_nc, width=opt.hr_width,
                                               depth=opt.hr_depth, coordinates=opt.hr_coor)
        
        
        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)
    

    def use_gpu(self):
        return len(self.gpu_ids) > 0

    def forward(self, input):
        features = self.encoder(input)
        output = self.decoder(input, features)
        return output, features#, lowres

class ConcatGenerator_fpn2(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='instanceaffine')
        parser.set_defaults(no_instance_dist=True)
        return parser

    def __init__(self, opt, img_size=(128,160), hr_stream=None, lr_stream=None):
        super(ConcatGenerator_fpn2, self).__init__()
        if lr_stream is None or hr_stream is None:
            lr_stream = dict()
            hr_stream = dict()
        self.num_inputs = opt.label_nc
        self.lr_instance = opt.lr_instance
        self.num_params = opt.latent_dim
        self.learned_ds_factor = opt.learned_ds_factor #(S2 in sec. 3.2)
        self.gpu_ids = opt.gpu_ids

        self.img_size = img_size
        
        # calculates the total downsampling factor in order to get the final low-res grid of parameters (S=S1xS2 in sec. 3.2)
        #self.downsampling = opt.crop_size // (16 * opt.aspect_ratio)
        #self.downsampling = min(img_size) // opt.lowest_ds_factor
        self.downsampling = opt.ds_factor
        norm_layer = get_nonspade_norm_layer(opt,'batch')
        #norm_layer = nn.BatchNorm2d
        
        self.encoder = Encoder_fpn2(num_in=self.num_inputs, block=Bottleneck, num_blocks=[2,2,2,2])
        #self.decoder = CatDecoderV2(self.downsampling, num_inputs=self.num_inputs, latent_dim=self.num_params,
        #                                       num_outputs=opt.output_nc, width=opt.hr_width,
        #                                       depth=opt.hr_depth, coordinates=opt.hr_coor)
        self.decoder = CatDecoder_fpn(num_inputs=self.num_inputs, latent_dim=self.num_params,
                                               num_outputs=opt.output_nc, width=opt.hr_width,
                                               depth=opt.hr_depth, coordinates=opt.hr_coor)
        
        
        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)
    

    def use_gpu(self):
        return len(self.gpu_ids) > 0

    def forward(self, input):
        features = self.encoder(input)
        features = features.repeat_interleave(2, dim=2)
        features = features.repeat_interleave(2, dim=3)
        output = self.decoder(input, features)
        return output, features#, lowres

class ConcatGenerator_fpn_multi(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='instanceaffine')
        parser.set_defaults(no_instance_dist=True)
        return parser

    def __init__(self, opt, img_size=(128,160), hr_stream=None, lr_stream=None):
        super(ConcatGenerator_fpn_multi, self).__init__()
        if lr_stream is None or hr_stream is None:
            lr_stream = dict()
            hr_stream = dict()
        self.num_inputs = opt.label_nc
        self.lr_instance = opt.lr_instance
        self.num_params = opt.latent_dim
        self.learned_ds_factor = opt.learned_ds_factor #(S2 in sec. 3.2)
        self.gpu_ids = opt.gpu_ids

        self.img_size = img_size
        
        # calculates the total downsampling factor in order to get the final low-res grid of parameters (S=S1xS2 in sec. 3.2)
        #self.downsampling = opt.crop_size // (16 * opt.aspect_ratio)
        #self.downsampling = min(img_size) // opt.lowest_ds_factor
        self.downsampling = opt.ds_factor
        
        self.encoder = Encoder_fpn2(num_in=self.num_inputs, block=Bottleneck, num_blocks=[2,2,2,2])
        #self.decoder = CatDecoder_fpn(num_inputs=self.num_inputs, latent_dim=self.num_params,
        #                                       num_outputs=opt.output_nc, width=opt.hr_width,
        #                                       depth=opt.hr_depth, coordinates=opt.hr_coor)
        
        
        self.decoder = CatDecoderV3(self.downsampling, img_size, num_inputs=self.num_inputs, latent_dim=self.num_params,
                                               num_outputs=opt.output_nc, width=opt.hr_width,
                                               depth=opt.hr_depth, coordinates=opt.hr_coor)
        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)
    

    def use_gpu(self):
        return len(self.gpu_ids) > 0

    def forward(self, input):
        features = self.encoder(input)
        features = features.repeat_interleave(2, dim=2)
        features = features.repeat_interleave(2, dim=3)
        output = self.decoder(input, features)
        return output, features#, lowres

class ConcatGenerator_fpn(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='instanceaffine')
        parser.set_defaults(no_instance_dist=True)
        return parser

    def __init__(self, opt, img_size=(128,160), hr_stream=None, lr_stream=None):
        super(ConcatGenerator_fpn, self).__init__()
        if lr_stream is None or hr_stream is None:
            lr_stream = dict()
            hr_stream = dict()
        self.num_inputs = opt.label_nc
        self.lr_instance = opt.lr_instance
        self.num_params = opt.latent_dim
        self.learned_ds_factor = opt.learned_ds_factor #(S2 in sec. 3.2)
        self.gpu_ids = opt.gpu_ids

        self.img_size = img_size
        
        # calculates the total downsampling factor in order to get the final low-res grid of parameters (S=S1xS2 in sec. 3.2)
        #self.downsampling = opt.crop_size // (16 * opt.aspect_ratio)
        #self.downsampling = min(img_size) // opt.lowest_ds_factor
        self.downsampling = opt.ds_factor
        norm_layer = get_nonspade_norm_layer(opt,'batch')
        #norm_layer = nn.BatchNorm2d
        
        self.encoder = Encoder_fpn(num_in=self.num_inputs, block=Bottleneck, num_blocks=[2,2,2,2])
        #self.decoder = CatDecoderV2(self.downsampling, num_inputs=self.num_inputs, latent_dim=self.num_params,
        #                                       num_outputs=opt.output_nc, width=opt.hr_width,
        #                                       depth=opt.hr_depth, coordinates=opt.hr_coor)
        self.decoder = CatDecoder_fpn(num_inputs=self.num_inputs, latent_dim=self.num_params,
                                               num_outputs=opt.output_nc, width=opt.hr_width,
                                               depth=opt.hr_depth, coordinates=opt.hr_coor)
        
        
        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)
    

    def use_gpu(self):
        return len(self.gpu_ids) > 0

    def forward(self, input):
        features,p3,p4 = self.encoder(input)
        output = self.decoder(input, features)
        return output, features#, lowres

class CatDecoder_fpn(th.nn.Module):
    """Addaptive pixel-wise MLPs"""
    def __init__(self, num_inputs=13, num_outputs=3, latent_dim = 64, width=64, depth=3, coordinates="cosine"):
        super(CatDecoder_fpn, self).__init__()

        self.pe_factor = 6
        self.num_inputs = num_inputs + 4*(self.pe_factor+1) + 256
        self.num_outputs = num_outputs
        self.width = width
        self.depth = depth
        self.coordinates = coordinates
        self.xy_coords = None
        self.net = []
        self.net += [nn.Linear(self.num_inputs, width, bias=True),
                     nn.LeakyReLU(0.01, inplace=True)]
        
        for i in range(depth):
            self.net += [nn.Linear(width, width, bias=True),
                         nn.LeakyReLU(0.01, inplace=True)]

        self.net += [nn.Linear(width, num_outputs),
                    nn.Tanh()]
            
        #with torch.no_grad():
        #    final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
        #                                np.sqrt(6 / hidden_features) / hidden_omega_0)
        self.net = nn.Sequential(*self.net)

    def forward(self, input, latent_code):
        bs, _, h, w = input.shape
        # Spatial encoding
        if self.xy_coords == None:
            self.xy_coords = _get_coords(bs, h, w, input.device,self.pe_factor)
        features = th.cat([input, self.xy_coords,latent_code], 1)
        features = features.permute(0,2,3,1)
        features = features.reshape(bs*h*w,-1)
        output = self.net(features)
        output= output.reshape((bs,1,h,w))
        return output

class CatDecoder(th.nn.Module):
    """Addaptive pixel-wise MLPs"""
    def __init__(self, downsampling, img_size,num_inputs=13, num_outputs=3, latent_dim = 64, width=64, depth=3, coordinates="cosine"):
        super(CatDecoder, self).__init__()

        self.downsampling = downsampling
        self.pe_factor = 6
        self.num_inputs = num_inputs + 4*(self.pe_factor+1) + latent_dim
        self.num_outputs = num_outputs
        self.width = width
        self.depth = depth
        self.coordinates = coordinates
        self.xy_coords = None
        self.net = []
        self.net += [nn.Linear(self.num_inputs, width, bias=True),
                     nn.LeakyReLU(0.01, inplace=True)]
        
        self.cell_size = pow(2, downsampling)
        #self.lr_h = int(img_size[0] / self.cell_size)
        #self.lr_w = int(img_size[1] / self.cell_size)
        for i in range(depth):
            self.net += [nn.Linear(width, width, bias=True),
                         nn.LeakyReLU(0.01, inplace=True)]

        self.net += [nn.Linear(width, num_outputs),
                    nn.Tanh()]
            
        #with torch.no_grad():
        #    final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
        #                                np.sqrt(6 / hidden_features) / hidden_omega_0)
        self.net = nn.Sequential(*self.net)

    def forward(self, input, latent_code):

        # Fetch sizes
        k = int(self.downsampling)
        
        bs, _, h, w = input.shape
        # Spatial encoding
        if self.xy_coords == None:
            self.xy_coords = _get_coords(bs, h, w, input.device,self.pe_factor)
        #latent_code = latent_code.unsqueeze(2).unsqueeze(3).expand(-1,-1,h,w)
        latent_code = latent_code.repeat_interleave(self.cell_size, dim=2)
        latent_code = latent_code.repeat_interleave(self.cell_size, dim=3)
        features = th.cat([input, self.xy_coords,latent_code], 1)
        features = features.permute(0,2,3,1)
        features = features.reshape(bs*h*w,-1)
        output = self.net(features)
        output= output.reshape((bs,1,h,w))
        return output

class CatGeneratorV2(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='instanceaffine')
        parser.set_defaults(no_instance_dist=True)
        return parser

    def __init__(self, opt, img_size=(128,160), hr_stream=None, lr_stream=None):
        super(CatGeneratorV2, self).__init__()
        if lr_stream is None or hr_stream is None:
            lr_stream = dict()
            hr_stream = dict()
        self.num_inputs = opt.label_nc
        self.lr_instance = opt.lr_instance
        self.num_params = opt.latent_dim
        self.learned_ds_factor = opt.learned_ds_factor #(S2 in sec. 3.2)
        self.gpu_ids = opt.gpu_ids

        self.img_size = img_size
        
        # calculates the total downsampling factor in order to get the final low-res grid of parameters (S=S1xS2 in sec. 3.2)
        #self.downsampling = opt.crop_size // (16 * opt.aspect_ratio)
        #self.downsampling = min(img_size) // opt.lowest_ds_factor
        self.downsampling = opt.ds_factor
        norm_layer = get_nonspade_norm_layer(opt,'batch')
        #norm_layer = nn.BatchNorm2d
        self.encoder = CatEncoderV2(self.downsampling, self.num_inputs, self.num_params, self.img_size, norm_layer)
        
        self.decoder = CatDecoderV4(self.downsampling, img_size, num_inputs=self.num_inputs, latent_dim=self.num_params,
                                               num_outputs=opt.output_nc, width=opt.hr_width,
                                               depth=opt.hr_depth, coordinates=opt.hr_coor)
        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)
    

    def use_gpu(self):
        return len(self.gpu_ids) > 0

    def forward(self, input):
        features = self.encoder(input)
        output = self.decoder(input, features)
        return output, features#, lowres

class CatGeneratorV3(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='instanceaffine')
        parser.set_defaults(no_instance_dist=True)
        return parser

    def __init__(self, opt, img_size=(128,160), hr_stream=None, lr_stream=None):
        super(CatGeneratorV3, self).__init__()
        if lr_stream is None or hr_stream is None:
            lr_stream = dict()
            hr_stream = dict()
        self.num_inputs = opt.label_nc
        self.lr_instance = opt.lr_instance
        self.num_params = opt.latent_dim
        self.learned_ds_factor = opt.learned_ds_factor #(S2 in sec. 3.2)
        self.gpu_ids = opt.gpu_ids

        self.img_size = img_size
        
        # calculates the total downsampling factor in order to get the final low-res grid of parameters (S=S1xS2 in sec. 3.2)
        #self.downsampling = opt.crop_size // (16 * opt.aspect_ratio)
        #self.downsampling = min(img_size) // opt.lowest_ds_factor
        self.downsampling = opt.ds_factor
        norm_layer = get_nonspade_norm_layer(opt,'batch')
        #norm_layer = nn.BatchNorm2d
        self.encoder = CatEncoderV2(self.downsampling, self.num_inputs, self.num_params, self.img_size, norm_layer)
        
        self.decoder = CatDecoderV4(self.downsampling, img_size, num_inputs=self.num_inputs, latent_dim=self.num_params,
                                               num_outputs=opt.output_nc, width=opt.hr_width,
                                               depth=opt.hr_depth, coordinates=opt.hr_coor)
        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)
    

    def use_gpu(self):
        return len(self.gpu_ids) > 0

    def forward(self, input):
        features = self.encoder(input)
        noise = torch.randn(features.shape) * 0.1
        features = features + noise.to(device=torch.device('cuda'))
        output = self.decoder(input, features)
        return output, features#, lowres

class CatGeneratorV4(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='instanceaffine')
        parser.set_defaults(no_instance_dist=True)
        return parser

    def __init__(self, opt, img_size=(128,160), hr_stream=None, lr_stream=None):
        super(CatGeneratorV4, self).__init__()
        if lr_stream is None or hr_stream is None:
            lr_stream = dict()
            hr_stream = dict()
        self.num_inputs = opt.label_nc
        self.lr_instance = opt.lr_instance
        self.num_params = opt.latent_dim
        self.learned_ds_factor = opt.learned_ds_factor #(S2 in sec. 3.2)
        self.gpu_ids = opt.gpu_ids

        self.img_size = img_size
        
        # calculates the total downsampling factor in order to get the final low-res grid of parameters (S=S1xS2 in sec. 3.2)
        #self.downsampling = opt.crop_size // (16 * opt.aspect_ratio)
        #self.downsampling = min(img_size) // opt.lowest_ds_factor
        self.downsampling = opt.ds_factor
        norm_layer = get_nonspade_norm_layer(opt,'batch')
        #norm_layer = nn.BatchNorm2d
        self.encoder = CatEncoderV2(self.downsampling, self.num_inputs, self.num_params, self.img_size, norm_layer)
        
        self.decoder = CatDecoderV3(self.downsampling, img_size, num_inputs=self.num_inputs, latent_dim=self.num_params,
                                               num_outputs=opt.output_nc, width=opt.hr_width,
                                               depth=opt.hr_depth, coordinates=opt.hr_coor)
        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)
    

    def use_gpu(self):
        return len(self.gpu_ids) > 0

    def forward(self, input):
        features = self.encoder(input)
        output = self.decoder(input, features)
        return output, input#, lowres

class CatEncoderV2(th.nn.Module):
    """ Same architecture as the image discriminator """

    def __init__(self, downsampling, num_in, num_out, crop_size, norm_layer, ndf=64, kernel_size=3):
        super(CatEncoderV2, self).__init__()
        self.max_width = 1024
        dslayers = []
        dslayers += [norm_layer(nn.Conv2d(num_in,ndf,kernel_size,stride=1,padding=1)),
                    nn.LeakyReLU(0.2,False)]
        for i in range(downsampling):
            dslayers += norm_layer(nn.Conv2d(ndf, ndf, kernel_size, stride=2, padding=1))
            dslayers += [nn.LeakyReLU(0.2, False)]
        convlayers = []
        conv_in = ndf
        conv_out = ndf
        for i in range(7):
            conv_in = conv_out
            conv_out = min(self.max_width, conv_in * 2)
            convlayers += norm_layer(nn.Conv2d(conv_in, conv_out, kernel_size, padding=1))
            convlayers += [nn.LeakyReLU(0.2, False)]
        self.dslayers = nn.Sequential(*dslayers)
        self.convlayers = nn.Sequential(*convlayers)
        self.final = nn.Conv2d(conv_out, num_out, kernel_size,stride=1,padding=1)
        self.actvn = nn.LeakyReLU(0.2, False)

    def forward(self, x):
        x = self.dslayers(x)
        x = self.convlayers(self.actvn(x))
        output = self.final(x)
        return output

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneckdropout(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneckdropout, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.dropout1 = nn.Dropout2d(p=0.2, inplace=True)
        self.bn1 = nn.InstanceNorm2d(planes, affine=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dropout2 = nn.Dropout2d(p=0.2, inplace=True)
        self.bn2 = nn.InstanceNorm2d(planes, affine=True)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.dropout3 = nn.Dropout2d(p=0.2, inplace=True)
        self.bn3 = nn.InstanceNorm2d(self.expansion*planes, affine=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(self.expansion*planes, affine=True)
            )

    def forward(self, x):
        out = self.conv1(x)
        if self.dropout1 is not None:
            out = self.dropout1(out)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        if self.dropout2 is not None:
            out = self.dropout2(out)
        out = F.relu(self.bn2(out))
        out = self.conv3(out)
        if self.dropout3 is not None:
            out = self.dropout3(out)
        out = self.bn3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck_in(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_in, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(planes, affine=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(planes, affine=True)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.InstanceNorm2d(self.expansion*planes, affine=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(self.expansion*planes, affine=True)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Encoder_fpn_att(th.nn.Module):
    "multi scale encoder inspired by fpn"
    def __init__(self, num_in,  block, num_blocks ,ndf=64, kernel_size=3):
        super(Encoder_fpn_att, self).__init__()
        self.in_planes = ndf
        self.block_expansion = 4
        net = []
        self.conv1 = nn.Conv2d(num_in, ndf, kernel_size=7,stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(ndf)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
        ## Bottom-up layers
        self.layer1 = self._make_layer(block, ndf, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, ndf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, ndf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, ndf * 8, num_blocks[3], stride=2)

        self.toplayer = nn.Conv2d(ndf * 8 * self.block_expansion, 256, kernel_size=1, stride=1,padding=0)   #Reduce channels
        
        # Smooth layers
        #self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        #self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        #self.net = nn.Sequential(*net)
        ## lateral layers
        self.latlayers1 = nn.Conv2d(ndf * 16, 256, kernel_size=1, stride=1,padding=0)
        self.latlayers2 = nn.Conv2d(ndf * 8, 256, kernel_size=1,stride=1,padding=0)
        self.latlayers3 = nn.Conv2d(ndf * 4, 256, kernel_size=1,stride=1,padding=0)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        ''' upsample and add two feature maps'''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self,x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayers1(c4))
        p3 = self._upsample_add(p4, self.latlayers2(c3))
        p2 = self._upsample_add(p3, self.latlayers3(c2))
        # Smooth
        #p4 = self.smooth1(p4)
        #p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        #print(p2.shape, p3.shape, p4.shape)
        return p2

class Encoder_fpn4(th.nn.Module):
    "multi scale encoder inspired by fpn fully resolution"
    def __init__(self, num_in,  block, num_blocks ,ndf=64, kernel_size=3):
        super(Encoder_fpn4, self).__init__()
        self.in_planes = ndf
        self.block_expansion = 4
        self.conv1 = nn.Conv2d(num_in, ndf, kernel_size=7, stride=1, padding=3, bias=False)
        #self.bn1 = nn.BatchNorm2d(ndf)
        self.bn1 = nn.InstanceNorm2d(ndf, affine=True)
        #self.bn1 = nn.InstanceNorm2d(ndf)

        ## Bottom-up layers
        self.layer1 = self._make_layer(block, ndf, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, ndf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, ndf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, ndf * 8, num_blocks[3], stride=2)

        self.toplayer = nn.Conv2d(ndf * 8 * self.block_expansion, 256, kernel_size=1, stride=1,padding=0)   #Reduce channels
        
        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        ## lateral layers
        self.latlayers1 = nn.Conv2d(ndf * 16, 256, kernel_size=1, stride=1,padding=0)
        self.latlayers2 = nn.Conv2d(ndf * 8, 256, kernel_size=1,stride=1,padding=0)
        self.latlayers3 = nn.Conv2d(ndf * 4, 256, kernel_size=1,stride=1,padding=0)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        ''' upsample and add two feature maps'''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self,x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Top-down & smooth
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayers1(c4))
        p4 = self.smooth1(p4)
        p3 = self._upsample_add(p4, self.latlayers2(c3))
        p3 = self.smooth2(p3)
        p2 = self._upsample_add(p3, self.latlayers3(c2))
        p2 = self.smooth3(p2)
        return p2

class Encoder_fpn3(th.nn.Module):
    "multi scale encoder inspired by fpn"
    def __init__(self, num_in,  block, num_blocks ,ndf=64, kernel_size=3):
        super(Encoder_fpn3, self).__init__()
        self.in_planes = ndf
        self.block_expansion = 4
        self.conv1 = nn.Conv2d(num_in, ndf, kernel_size=7, stride=2, padding=3, bias=False)
        #self.bn1 = nn.BatchNorm2d(ndf)
        self.bn1 = nn.InstanceNorm2d(ndf, affine=True)
        #self.bn1 = nn.InstanceNorm2d(ndf)

        ## Bottom-up layers
        self.layer1 = self._make_layer(block, ndf, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, ndf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, ndf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, ndf * 8, num_blocks[3], stride=2)

        self.toplayer = nn.Conv2d(ndf * 8 * self.block_expansion, 256, kernel_size=1, stride=1,padding=0)   #Reduce channels
        
        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        ## lateral layers
        self.latlayers1 = nn.Conv2d(ndf * 16, 256, kernel_size=1, stride=1,padding=0)
        self.latlayers2 = nn.Conv2d(ndf * 8, 256, kernel_size=1,stride=1,padding=0)
        self.latlayers3 = nn.Conv2d(ndf * 4, 256, kernel_size=1,stride=1,padding=0)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        ''' upsample and add two feature maps'''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self,x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Top-down & smooth
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayers1(c4))
        p4 = self.smooth1(p4)
        p3 = self._upsample_add(p4, self.latlayers2(c3))
        p3 = self.smooth2(p3)
        p2 = self._upsample_add(p3, self.latlayers3(c2))
        p2 = self.smooth3(p2)
        return p2

class Encoder_fpn2(th.nn.Module):
    "multi scale encoder inspired by fpn"
    def __init__(self, num_in,  block, num_blocks ,ndf=64, kernel_size=3):
        super(Encoder_fpn2, self).__init__()
        self.in_planes = ndf
        self.block_expansion = 4
        self.conv1 = nn.Conv2d(num_in, ndf, kernel_size=7, stride=2, padding=3, bias=False)
        #self.bn1 = nn.BatchNorm2d(ndf)
        #self.bn1 = nn.InstanceNorm2d(ndf, affine=True)
        self.bn1 = nn.InstanceNorm2d(ndf)

        ## Bottom-up layers
        self.layer1 = self._make_layer(block, ndf, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, ndf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, ndf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, ndf * 8, num_blocks[3], stride=2)

        self.toplayer = nn.Conv2d(ndf * 8 * self.block_expansion, 256, kernel_size=1, stride=1,padding=0)   #Reduce channels
        
        # Smooth layers
        self.smooth = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        ## lateral layers
        self.latlayers1 = nn.Conv2d(ndf * 16, 256, kernel_size=1, stride=1,padding=0)
        self.latlayers2 = nn.Conv2d(ndf * 8, 256, kernel_size=1,stride=1,padding=0)
        self.latlayers3 = nn.Conv2d(ndf * 4, 256, kernel_size=1,stride=1,padding=0)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        ''' upsample and add two feature maps'''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self,x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayers1(c4))
        p3 = self._upsample_add(p4, self.latlayers2(c3))
        p2 = self._upsample_add(p3, self.latlayers3(c2))
        # Smooth
        p2 = self.smooth(p2)
        return p2

class Encoder_fpn(th.nn.Module):
    "multi scale encoder inspired by fpn"
    def __init__(self, num_in,  block, num_blocks ,ndf=64, kernel_size=3):
        super(Encoder_fpn, self).__init__()
        self.in_planes = ndf
        self.block_expansion = 4
        net = []
        self.conv1 = nn.Conv2d(num_in, ndf, kernel_size=7,stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(ndf)

        ## Bottom-up layers
        self.layer1 = self._make_layer(block, ndf, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, ndf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, ndf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, ndf * 8, num_blocks[3], stride=2)

        self.toplayer = nn.Conv2d(ndf * 8 * self.block_expansion, 256, kernel_size=1, stride=1,padding=0)   #Reduce channels
        
        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        #self.net = nn.Sequential(*net)
        ## lateral layers
        self.latlayers1 = nn.Conv2d(ndf * 16, 256, kernel_size=1, stride=1,padding=0)
        self.latlayers2 = nn.Conv2d(ndf * 8, 256, kernel_size=1,stride=1,padding=0)
        self.latlayers3 = nn.Conv2d(ndf * 4, 256, kernel_size=1,stride=1,padding=0)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        ''' upsample and add two feature maps'''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self,x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayers1(c4))
        p3 = self._upsample_add(p4, self.latlayers2(c3))
        p2 = self._upsample_add(p3, self.latlayers3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        return p2, p3, p4

class CatEncoder_muls(th.nn.Module):
    """ Same architecture as the image discriminator multi scale"""

    def __init__(self, downsampling, num_in, num_out, crop_size, norm_layer, ndf=64, kernel_size=3):
        super(CatEncoder_muls, self).__init__()
        self.scales = 5
        self.max_width = 1024
        #dslayers = []
        #dslayers += [norm_layer(nn.Conv2d(num_in,ndf,kernel_size,stride=1,padding=1)),
        #            nn.LeakyReLU(0.2,False)]
        #self.conv1 = nn.Conv2d(num_in,64,kernel_size=7,stride=2,padding=3,bias=False)
        conv_in = num_in
        conv_out = ndf
        dslayers_1 = [norm_layer(nn.Conv2d(conv_in, conv_out, 7, stride=1, padding=3)),
                    nn.LeakyReLU(0.2,False),
                    norm_layer(nn.Conv2d(conv_out,conv_out,kernel_size,padding=1)),
                    nn.LeakyReLU(0.2,False)]

        conv_in = conv_out
        conv_out = conv_in * 2
        dslayers_2 = [norm_layer(nn.Conv2d(conv_in, conv_out, kernel_size, stride=2, padding=1)),
                    nn.LeakyReLU(0.2,False),
                    norm_layer(nn.Conv2d(conv_out,conv_out,kernel_size,padding=1)),
                    nn.LeakyReLU(0.2,False),
                    norm_layer(nn.Conv2d(conv_out,conv_out,kernel_size,padding=1)),
                    nn.LeakyReLU(0.2,False)]

        conv_in = conv_out
        conv_out = conv_in * 2
        dslayers_3 = [norm_layer(nn.Conv2d(conv_in, conv_out, kernel_size, stride=2, padding=1)),
                    nn.LeakyReLU(0.2,False),
                    norm_layer(nn.Conv2d(conv_out,conv_out,kernel_size,padding=1)),
                    nn.LeakyReLU(0.2,False),
                    norm_layer(nn.Conv2d(conv_out,conv_out,kernel_size,padding=1)),
                    nn.LeakyReLU(0.2,False)]
        conv_in = conv_out
        conv_out = conv_in * 2
        dslayers_4 = [norm_layer(nn.Conv2d(conv_in, conv_out, kernel_size, stride=2, padding=1)),
                    nn.LeakyReLU(0.2,False),
                    norm_layer(nn.Conv2d(conv_out,conv_out,kernel_size,padding=1)),
                    nn.LeakyReLU(0.2,False),
                    norm_layer(nn.Conv2d(conv_out,conv_out,kernel_size,padding=1)),
                    nn.LeakyReLU(0.2,False)]
        
        self.dslayers_1 = nn.Sequential(*dslayers_1)
        self.dslayers_2 = nn.Sequential(*dslayers_2)
        self.dslayers_3 = nn.Sequential(*dslayers_3)
        self.dslayers_4 = nn.Sequential(*dslayers_4)
        #self.final = nn.Conv2d(conv_out, num_out, kernel_size,stride=1,padding=1)
        #self.actvn = nn.LeakyReLU(0.2, False)

    def forward(self, x):
        x = self.dslayers_1(x)
        latent_1 = x
        x = self.dslayers_2(x)
        latent_2 = x.repeat_interleave(2, dim=2)
        latent_2 = latent_2.repeat_interleave(2, dim=3)
        x = self.dslayers_3(x)
        latent_3 = x.repeat_interleave(4, dim=2)
        latent_3 = latent_3.repeat_interleave(4, dim=3)
        x = self.dslayers_4(x)
        latent_4 = x.repeat_interleave(8, dim=2)
        latent_4 = latent_4.repeat_interleave(8, dim=3)
        output = torch.concat([latent_1, latent_2, latent_3, latent_4],dim=1)
        return output

class locallinear(th.nn.Module):
    def __init__(self, lr_h, lr_w, in_ch, out_ch):
        super(locallinear, self).__init__()

        self.fc_w = nn.Parameter(torch.zeros(lr_h, lr_w, in_ch, out_ch))
        self.fc_b = nn.Parameter(torch.zeros(lr_h, lr_w, 1, out_ch))
        nn.init.normal_(self.fc_w, mean=0.0, std=0.02)
        nn.init.constant_(self.fc_b, 0)
    
    def forward(self, input):
        out = torch.matmul(input, self.fc_w) + self.fc_b
        return out

class CatDecoderV3(th.nn.Module):
    """Addaptive pixel-wise MLPs"""
    def __init__(self, downsampling, img_size, num_inputs=13, num_outputs=3, latent_dim = 64, width=64, depth=3, coordinates="cosine"):
        super(CatDecoderV3, self).__init__()

        self.downsampling = downsampling
        self.pe_factor = 6
        self.num_inputs = num_inputs + 4*(self.pe_factor+1) + latent_dim
        self.num_outputs = num_outputs
        self.width = width
        self.depth = depth
        self.coordinates = coordinates
        self.xy_coords = None
        self.cell_size = pow(2, downsampling)
        self.lr_h = int(img_size[0] / self.cell_size)
        self.lr_w = int(img_size[1] / self.cell_size)
        print(self.lr_h, self.lr_w)
        self.fc_w = []
        self.fc_b = []
        in_ch = self.num_inputs
        out_ch = latent_dim
        
        net = [locallinear(self.lr_h,self.lr_w,self.num_inputs,latent_dim),
                nn.LeakyReLU(0.01, inplace=True)]
        for i in range(self.depth):
            net += [locallinear(self.lr_h,self.lr_w, latent_dim, latent_dim),
                nn.LeakyReLU(0.01, inplace=True)]
        
        net += [locallinear(self.lr_h,self.lr_w, latent_dim, self.num_outputs),
                nn.Tanh()]
        self.net = nn.Sequential(*net)
 
    def forward(self, input, latent_code):
        # Fetch sizes
        k = int(self.downsampling)
        bs, _, h, w = input.shape
        # Spatial encoding
        if self.xy_coords == None:
            self.xy_coords = _get_coords(bs, h, w, input.device,self.pe_factor)
        #latent_code = latent_code.unsqueeze(2).unsqueeze(3).expand(-1,-1,h,w)
        #latent_code = latent_code.repeat_interleave(self.cell_size, dim=2)
        #latent_code = latent_code.repeat_interleave(self.cell_size, dim=3)
        features = th.cat([input, self.xy_coords,latent_code], 1)
        nci = features.shape[1]
        tiles = features.unfold(2, self.cell_size, self.cell_size).unfold(3, self.cell_size, self.cell_size)
        out = rearrange(tiles,'b c h w p1 p2 -> b h w (p1 p2) c')
        out = self.net(out)
        #output = torch.cat(tile_list,dim=2)
        out = rearrange(out,'b h w (p1 p2) c -> b c (h p1) (w p2)',p1=self.cell_size,p2=self.cell_size)
        return out

class CatDecoderV4(th.nn.Module):
    """Adaptive pixel-wise MLPs old implement"""
    def __init__(self, downsampling, img_size, num_inputs=13, num_outputs=3, latent_dim = 64, width=64, depth=3, coordinates="cosine"):
        super(CatDecoderV4, self).__init__()

        self.downsampling = downsampling
        self.pe_factor = 6
        self.latent_dim = latent_dim
        self.num_inputs = num_inputs + 4*(self.pe_factor+1) + latent_dim
        self.num_outputs = num_outputs
        self.width = width
        self.depth = depth
        self.coordinates = coordinates
        self.xy_coords = None
        self.cell_size = pow(2, downsampling)
        self.lr_h = int(img_size[0] / self.cell_size)
        self.lr_w = int(img_size[1] / self.cell_size)
        for i in range(self.lr_h * self.lr_w):
            subnet = SingleDecoder(self.downsampling, num_inputs=self.num_inputs,
                                               num_outputs=self.num_outputs, width=self.width,
                                               depth=self.depth, coordinates=self.coordinates)
            self.add_module('MLP_%d' % i, subnet)
        #with torch.no_grad():
        #    final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
        #                                np.sqrt(6 / hidden_features) / hidden_omega_0)

    def forward(self, input, latent_code):

        # Fetch sizes
        k = int(self.downsampling)
        bs, _, h, w = input.shape
        # Spatial encoding
        if self.xy_coords == None:
            self.xy_coords = _get_coords(bs, h, w, input.device,self.pe_factor)
        #latent_code = latent_code.unsqueeze(2).unsqueeze(3).expand(-1,-1,h,w)
        latent_code = latent_code.repeat_interleave(self.cell_size, dim=2)
        latent_code = latent_code.repeat_interleave(self.cell_size, dim=3)
        features = th.cat([input, self.xy_coords,latent_code], 1)
        nci = features.shape[1]
        tiles = features.unfold(2, self.cell_size, self.cell_size).unfold(3, self.cell_size, self.cell_size)
        tiles = tiles.permute(0, 2, 3, 4, 5, 1).contiguous().view(
            bs, self.lr_h, self.lr_w, int(self.cell_size * self.cell_size), nci)
        #features = features.permute(0,2,3,1)
        #features = features.reshape(bs*h*w,-1)
        generators = list(dict(self.named_children()).values())
        tile_list = []
        for i in range(self.lr_h):
            row_list = []
            for j in range(self.lr_w):
                tile = tiles[:,i,j,:,:].reshape(bs*self.cell_size * self.cell_size,-1)
                #tile = features[:,:,i*32:(i+1) * 32,j*32:(j+1)*32].reshape(bs*32*32,-1)
                sg = generators[self.lr_w*i+j]
                out = sg(tile)
                out = out.reshape(bs,1,self.cell_size, self.cell_size)
                row_list.append(out)
            row = torch.cat(row_list,dim=3)
            tile_list.append(row)
        output = torch.cat(tile_list,dim=2)
        output= output.reshape((bs,1,h,w))
        return output

class SingleDecoder(th.nn.Module):
    """Addaptive pixel-wise MLPs"""
    def __init__(self, downsampling, num_inputs=13, num_outputs=3,width=64, depth=3, coordinates="cosine"):
        super(SingleDecoder, self).__init__()

        self.downsampling = downsampling
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.width = width
        self.depth = depth
        self.coordinates = coordinates
        self.xy_coords = None
        self.net = []
        self.net += [nn.Linear(self.num_inputs, width, bias=True),
                     nn.LeakyReLU(0.01, inplace=True)]
        
        for i in range(depth):
            self.net += [nn.Linear(width, width, bias=True),
                         nn.LeakyReLU(0.01, inplace=True)]

        self.net += [nn.Linear(width, num_outputs),
                    nn.Tanh()]
            
        #with torch.no_grad():
        #    final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
        #                                np.sqrt(6 / hidden_features) / hidden_omega_0)
        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        output = self.net(input)
        return output
