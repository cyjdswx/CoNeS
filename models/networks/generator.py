from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ASAPNetsBlock as ASAPNetsBlock
from models.networks.generator_cat import Encoder_fpn, Bottleneck, Bottleneck_in, Bottleneckdropout, Encoder_fpn2, Encoder_fpn3, Encoder_fpn4
from models.networks.architecture import MySeparableBilinearDownsample as BilinearDownsample
import torch.nn.utils.spectral_norm as spectral_norm
import torch as th
from math import pi
from math import log2
import time
import numpy as np

from util.util import OrderedDict

class InitWeights_me(object):
    def __init__(self, init_type='normal', gain=0.02,neg_slope=1e-2):
        self.init_type = init_type
        self.gain = gain
        self.neg_slope = neg_slope

    def __call__(self,module):
        if isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, 1.0, self.gain)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear):
            if self.init_type == 'normal':
                nn.init.normal_(module.weight, 0.0, self.gain)
            elif self.init_type == 'xavier':
                nn.init.xavier_normal_(module.weight, self.gain)
            elif self.init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(module.weight, self.gain)
            elif self.init_type == 'kaiming':
                nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in')
            elif self.init_type =='orthogonal':
                nn.init.orthogonal_(module.weight, gain=self.gain)
            elif self.init_type == 'None':
                module.reset_parameters()
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % self.init_type)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)

class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(30 * input)
        #return torch.sin(input)

class ASAPNetsGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='instanceaffine')
        parser.set_defaults(no_instance_dist=True)
        return parser

    def __init__(self, opt, img_size=(128,160), hr_stream=None, lr_stream=None, fast=False):
        super(ASAPNetsGenerator, self).__init__()
        if lr_stream is None or hr_stream is None:
            lr_stream = dict()
            hr_stream = dict()
        self.num_inputs = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if (opt.no_instance_edge & opt.no_instance_dist) else 1)
        self.num_inputs = opt.label_nc
        self.lr_instance = opt.lr_instance
        self.learned_ds_factor = opt.learned_ds_factor #(S2 in sec. 3.2)
        self.gpu_ids = opt.gpu_ids
        # calculates the total downsampling factor in order to get the final low-res grid of parameters (S=S1xS2 in sec. 3.2)
        self.downsampling = pow(2, opt.ds_factor)

        self.highres_stream = ASAPNetsHRStream(self.downsampling, num_inputs=self.num_inputs,
                                               num_outputs=opt.output_nc, width=opt.hr_width,
                                               depth=opt.hr_depth, coordinates=opt.hr_coor,
                                               no_one_hot=opt.no_one_hot, lr_instance=opt.lr_instance,
                                               **hr_stream)

        num_params = self.highres_stream.num_params
        num_inputs_lr = self.highres_stream.num_inputs + (1 if opt.lr_instance else 0)
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        self.lowres_stream = ASAPNetsLRStream(num_inputs_lr, num_params, norm_layer, width=opt.lr_width,
                                              max_width=opt.lr_max_width, depth=opt.lr_depth,
                                              #learned_ds_factor=opt.learned_ds_factor,
                                              learned_ds_factor=self.downsampling,
                                              reflection_pad=opt.reflection_pad, **lr_stream)

    def use_gpu(self):
        return len(self.gpu_ids) > 0
    
    def forward(self, img, z=None):
        #lowres = self.get_lowres(highres)
        features = self.lowres_stream(img)
        output = self.highres_stream(img, features)
        return output, features#, lowres

class ASAPfunctaGeneratorV3(BaseNetwork):
    def __init__(self, opt, img_size=(128,160), hr_stream=None, lr_stream=None, fast=False):
        super(ASAPfunctaGeneratorV3, self).__init__()
        if lr_stream is None or hr_stream is None:
            lr_stream = dict()
            hr_stream = dict()
        self.num_inputs = opt.label_nc
        self.gpu_ids = opt.gpu_ids

        self.highres_stream = ASAPfunctaHRStreamfulres(num_inputs=self.num_inputs,
                                               num_outputs=opt.output_nc, width=opt.hr_width,
                                               depth=opt.hr_depth, coordinates=opt.hr_coor,**hr_stream)

        num_params = self.highres_stream.num_params
        
        self.latlayers = nn.Conv2d(256, num_params, kernel_size=1, stride=1,padding=0)
        self.lowres_stream = Encoder_fpn4(num_in=self.num_inputs, block=Bottleneck_in, num_blocks=[2,4,23,3])
        self.init = InitWeights_me(opt.init_type, opt.init_variance)
        self.lowres_stream.apply(self.init)
        self.latlayers.apply(self.init)
        self.highres_stream.apply(self.init)
    
    def use_gpu(self):
        return len(self.gpu_ids) > 0
    
    def forward(self, input):
        features = self.lowres_stream(input)
        features = self.latlayers(features)
        output = self.highres_stream(input, features)
        return output, features

class ASAPfunctaGeneratorV2(BaseNetwork):
    def __init__(self, opt, img_size=(128,160), hr_stream=None, lr_stream=None, fast=False):
        super(ASAPfunctaGeneratorV2, self).__init__()
        if lr_stream is None or hr_stream is None:
            lr_stream = dict()
            hr_stream = dict()
        #self.num_inputs = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if (opt.no_instance_edge & opt.no_instance_dist) else 1)
        self.num_inputs = opt.label_nc
        self.learned_ds_factor = opt.learned_ds_factor #(S2 in sec. 3.2)
        self.gpu_ids = opt.gpu_ids
        img_size = (128,160)
        #img_size = (448,448)
        # calculates the total downsampling factor in order to get the final low-res grid of parameters (S=S1xS2 in sec. 3.2)

        self.highres_stream = ASAPfunctaHRStream(2, num_inputs=self.num_inputs,
                                               num_outputs=opt.output_nc, width=opt.hr_width,
                                               depth=opt.hr_depth, coordinates=opt.hr_coor,**hr_stream)

        num_params = self.highres_stream.num_params
        
        self.latlayers = nn.Conv2d(256, num_params, kernel_size=1, stride=1,padding=0)
        #self.lowres_stream = Encoder_fpn2(num_in=self.num_inputs, block=Bottleneck_in, num_blocks=[2,4,23,3])
        self.lowres_stream = Encoder_fpn3(num_in=self.num_inputs, block=Bottleneck_in, num_blocks=[2,4,23,3])
    
    def use_gpu(self):
        return len(self.gpu_ids) > 0
    
    def forward(self, input):
        features = self.lowres_stream(input)
        features = self.latlayers(features)
        output = self.highres_stream(input, features)
        return output, features

class ASAPFPNGeneratorV2(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='instanceaffine')
        parser.set_defaults(no_instance_dist=True)
        #parser.set_defaults(hr_coor="cosine")
        return parser

    def __init__(self, opt, img_size=(128,160), hr_stream=None, lr_stream=None, fast=False):
        super(ASAPFPNGeneratorV2, self).__init__()
        if lr_stream is None or hr_stream is None:
            lr_stream = dict()
            hr_stream = dict()
        #self.num_inputs = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if (opt.no_instance_edge & opt.no_instance_dist) else 1)
        self.num_inputs = opt.label_nc
        self.lr_instance = opt.lr_instance
        self.learned_ds_factor = opt.learned_ds_factor #(S2 in sec. 3.2)
        self.gpu_ids = opt.gpu_ids
        img_size = (128,160)
        #img_size = (448,448)
        # calculates the total downsampling factor in order to get the final low-res grid of parameters (S=S1xS2 in sec. 3.2)

        self.highres_stream = ASAPFPNHRStream(2, num_inputs=self.num_inputs,
                                               num_outputs=opt.output_nc, width=opt.hr_width,
                                               depth=opt.hr_depth, coordinates=opt.hr_coor,**hr_stream)

        num_params = self.highres_stream.num_params
        
        self.latlayers = nn.Conv2d(256, num_params, kernel_size=1, stride=1,padding=0)
        #self.lowres_stream = Encoder_fpn2(num_in=self.num_inputs, block=Bottleneck, num_blocks=[2,2,2,2])
        self.lowres_stream = Encoder_fpn2(num_in=self.num_inputs, block=Bottleneck_in, num_blocks=[2,4,23,3])
        #self.lowres_stream = Encoder_fpn2(num_in=self.num_inputs, block=Bottleneck, num_blocks=[2,6,21,11])
    
    def use_gpu(self):
        return len(self.gpu_ids) > 0
    
    def forward(self, input):
        features = self.lowres_stream(input)
        features = self.latlayers(features)
        output = self.highres_stream(input, features)
        return output, features#, lowres

class ASAPFPNGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='instanceaffine')
        parser.set_defaults(no_instance_dist=True)
        #parser.set_defaults(hr_coor="cosine")
        return parser

    def __init__(self, opt, img_size=(128,160), hr_stream=None, lr_stream=None, fast=False):
        super(ASAPFPNGenerator, self).__init__()
        if lr_stream is None or hr_stream is None:
            lr_stream = dict()
            hr_stream = dict()
        #self.num_inputs = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if (opt.no_instance_edge & opt.no_instance_dist) else 1)
        self.num_inputs = opt.label_nc
        self.lr_instance = opt.lr_instance
        self.learned_ds_factor = opt.learned_ds_factor #(S2 in sec. 3.2)
        self.gpu_ids = opt.gpu_ids
        img_size = (128,160)
        #img_size = (448,448)
        # calculates the total downsampling factor in order to get the final low-res grid of parameters (S=S1xS2 in sec. 3.2)
        #self.downsampling = opt.crop_size // (16 * opt.aspect_ratio)
        self.downsampling = min(img_size) // opt.lowest_ds_factor

        self.highres_stream = ASAPFPNHRStream(num_inputs=self.num_inputs,
                                               num_outputs=opt.output_nc, width=opt.hr_width,
                                               depth=opt.hr_depth, coordinates=opt.hr_coor,
                                               **hr_stream)

        num_params = self.highres_stream.num_params
        self.latlayers = nn.Conv2d(256, num_params, kernel_size=1, stride=1,padding=0)
        self.lowres_stream = Encoder_fpn(num_in=self.num_inputs, block=Bottleneck, num_blocks=[2,2,2,2])
    
    def use_gpu(self):
        return len(self.gpu_ids) > 0
    '''
    def get_lowres(self, im):
        """Creates a lowres version of the input."""
        device = self.use_gpu()
        if(self.learned_ds_factor != self.downsampling):
            myds = BilinearDownsample(int(self.downsampling//self.learned_ds_factor), self.num_inputs,device)
            return myds(im)
        else:
            return im
    '''
    def forward(self, input, z=None):
        features = self.lowres_stream(input)
        features = self.latlayers(features)
        output = self.highres_stream(input, features)
        return output, features#, lowres

def _get_coords(bs, h, w, device, ds, coords_type):
    """Creates the position encoding for the pixel-wise MLPs"""
    if coords_type == 'cosine':
        f0 = ds
        f = f0
        while f > 1:
            x = th.arange(0, w).float()
            y = th.arange(0, h).float()
            xcos = th.cos((2 * pi * th.remainder(x, f).float() / f).float())
            xsin = th.sin((2 * pi * th.remainder(x, f).float() / f).float())
            ycos = th.cos((2 * pi * th.remainder(y, f).float() / f).float())
            ysin = th.sin((2 * pi * th.remainder(y, f).float() / f).float())
            xcos = xcos.view(1, 1, 1, w).repeat(bs, 1, h, 1)
            xsin = xsin.view(1, 1, 1, w).repeat(bs, 1, h, 1)
            ycos = ycos.view(1, 1, h, 1).repeat(bs, 1, 1, w)
            ysin = ysin.view(1, 1, h, 1).repeat(bs, 1, 1, w)
            coords_cur = th.cat([xcos, xsin, ycos, ysin], 1).to(device)
            if f < f0:
                coords = th.cat([coords, coords_cur], 1).to(device)    # pyright: ignore
            else:
                coords = coords_cur
            f = f//2
        return coords.to(device)    # pyright:ignore
    if coords_type == "siren":
        f = ds
        print("siren",f)
        x  = th.arange(0, w).float()
        y = th.arange(0, h).float()
        xn = x.float() * 2 / w -1
        yn = y.float() * 2 / h -1
        xn = xn.view(1, 1, 1, w).repeat(bs, 1, h, 1)
        yn = yn.view(1, 1, h, 1).repeat(bs, 1, 1, w)
        coords_cur = th.cat([xn, yn], 1).to(device)
        print(coords_cur)
        coords = coords_cur    # pyright: ignore
        return coords.to(device)
    else:
        raise NotImplementedError()

def _get_coords_global(bs, h, w, device, ds):
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

class ASAPNetsLRStream(th.nn.Sequential):
    """Convolutional LR stream to estimate the pixel-wise MLPs parameters"""
    def __init__(self, num_in, num_out, norm_layer, width=64, max_width=1024, depth=7, learned_ds_factor=16,
                 reflection_pad=False, replicate_pad=False):
        super(ASAPNetsLRStream, self).__init__()

        model = []

        self.num_out = num_out
        padw = 1
        if reflection_pad:
            padw = 0
            model += [th.nn.ReflectionPad2d(1)]
        if replicate_pad:
            padw = 0
            model += [th.nn.ReplicationPad2d(1)]

        count_ly = 0

        model += [norm_layer(th.nn.Conv2d(num_in, width, 3, stride=1, padding=padw)),
                  th.nn.ReLU(inplace=True)]

        num_ds_layers = int(log2(learned_ds_factor))

        # strided conv layers for learning downsampled representation of the input"
        for i in range(num_ds_layers):
            if reflection_pad:
                model += [th.nn.ReflectionPad2d(1)]
            if replicate_pad:
                model += [th.nn.ReplicationPad2d(1)]
            if i == num_ds_layers-1:
                last_width = max_width
                model += [norm_layer(th.nn.Conv2d(width, last_width, 3, stride=2, padding=padw)),
                          th.nn.ReLU(inplace=True)]
                width = last_width
            else:
                model += [norm_layer(th.nn.Conv2d(width, width, 3, stride=2, padding=padw)),
                      th.nn.ReLU(inplace=True)]

        # ConvNet to estimate the MLPs parameters"
        for i in range(count_ly, count_ly+depth):
            model += [ASAPNetsBlock(width, norm_layer, reflection_pad=reflection_pad, replicate_pad=replicate_pad)]

        # Final parameter prediction layer, transfer conv channels into the per-pixel number of MLP parameters
        model += [th.nn.Conv2d(width, self.num_out, 1)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class ASAPfunctaHRStreamfulres(th.nn.Module):
    """Addaptive pixel-wise MLPs"""
    def __init__(self, num_inputs=13, num_outputs=3, width=64, depth=5, coordinates="cosine"):
        super(ASAPfunctaHRStreamfulres, self).__init__()

        self.pe_factor = 6
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.width = width
        self.depth = depth
        self.coordinates = coordinates
        self.xy_coords = None
        self.channels = []
        self._set_channels()
        self.num_params = 0
        self.biases = []
        self._set_num_params()
        self.net = nn.ModuleList()
        for i in range(len(self.channels) - 1):
            #layer_name = 'fc'+str(i)
            self.net.append(nn.Linear(self.channels[i], self.channels[i+1], bias=True))

    @property  # for backward compatibility
    def ds(self):
        return self.downsampling


    def _set_channels(self):
        """Compute and store the hr-stream layer dimensions."""
        in_ch = self.num_inputs
        if self.coordinates == "cosine":
            in_ch += 4*(self.pe_factor+1)
        self.channels = [in_ch]

        for _ in range(self.depth - 1):  # intermediate layer -> cste size
            self.channels.append(self.width)
        # output layer
        self.channels.append(self.num_outputs)
    
    def _set_num_params(self):
        nparams = 0
        # go over input/output channels for each layer
        idx = 0
        for layer, nci in enumerate(self.channels[:-1]):
            nco = self.channels[layer + 1] 
            nparams += nco  # FC biases
            self.biases.append((idx, idx + nco))
            idx += nco

        self.num_params = nparams

    def _get_bias_indices(self, idx):
        return self.biases[idx]

    def forward(self, highres, lr_params):
        assert lr_params.shape[1] == self.num_params, "incorrect input params"
        # Fetch sizes
        bs, _, h, w = highres.shape
        bs, _, h_lr, w_lr = lr_params.shape
        # Spatial encoding
        if not(self.coordinates is None):
            if self.xy_coords is None:
                self.xy_coords = _get_coords_global(bs, h, w, highres.device, self.pe_factor)
            highres = th.cat([highres, self.xy_coords], 1)
        
        # Split input in tiles of size kxk according to the NN interp factor (the total downsampling factor),
        # with channels last (for matmul)
        # all pixels within a tile of kxk are processed by the same MLPs parameters
        nci = highres.shape[1]
        out = rearrange(highres,'b c h w -> b h w c')
        num_layers = len(self.channels) - 1
        for idx, nci in enumerate(self.channels[:-1]):
            nco = self.channels[idx + 1]

            # Select params in lowres buffer
            bstart, bstop = self._get_bias_indices(idx)
            b_ = lr_params[:, bstart:bstop]
            out = self.net[idx](out)
            b_ = b_.permute(0, 2, 3, 1)
            out = out + b_
            # Apply RelU non-linearity in all but the last layer, and tanh in the last
            if idx < num_layers - 1:
                out = th.nn.functional.leaky_relu(out, 0.01, inplace=True)
            else:
                out = F.tanh(out)
        out = rearrange(out,'b h w c-> b c h w')
        return out

class ASAPfunctaHRStream(th.nn.Module):
    """Addaptive pixel-wise MLPs"""
    def __init__(self, cell_size, num_inputs=13, num_outputs=3, width=64, depth=5, coordinates="cosine"):
        super(ASAPfunctaHRStream, self).__init__()

        self.pe_factor = 6
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.width = width
        self.depth = depth
        self.coordinates = coordinates
        self.xy_coords = None
        self.channels = []
        self._set_channels()
        self.num_params = 0
        self.biases = []
        self._set_num_params()
        self.cell_size = cell_size
        self.net = nn.ModuleList()
        for i in range(len(self.channels) - 1):
            #layer_name = 'fc'+str(i)
            self.net.append(nn.Linear(self.channels[i], self.channels[i+1], bias=True))
        
        #self.net = nn.Sequential(self.net)

    @property  # for backward compatibility
    def ds(self):
        return self.downsampling


    def _set_channels(self):
        """Compute and store the hr-stream layer dimensions."""
        in_ch = self.num_inputs
        if self.coordinates == "cosine":
            in_ch += 4*(self.pe_factor+1)
        self.channels = [in_ch]

        for _ in range(self.depth - 1):  # intermediate layer -> cste size
            self.channels.append(self.width)
        # output layer
        self.channels.append(self.num_outputs)
    
    def _set_num_params(self):
        nparams = 0
        # go over input/output channels for each layer
        idx = 0
        for layer, nci in enumerate(self.channels[:-1]):
            nco = self.channels[layer + 1] 
            nparams += nco  # FC biases
            self.biases.append((idx, idx + nco))
            idx += nco

        self.num_params = nparams

    def _get_bias_indices(self, idx):
        return self.biases[idx]

    def forward(self, highres, lr_params):
        assert lr_params.shape[1] == self.num_params, "incorrect input params"
        # Fetch sizes
        k = self.cell_size
        bs, _, h, w = highres.shape
        bs, _, h_lr, w_lr = lr_params.shape
        # Spatial encoding
        if not(self.coordinates is None):
            if self.xy_coords is None:
                self.xy_coords = _get_coords_global(bs, h, w, highres.device, self.pe_factor)
            highres = th.cat([highres, self.xy_coords], 1)
        
        # Split input in tiles of size kxk according to the NN interp factor (the total downsampling factor),
        # with channels last (for matmul)
        # all pixels within a tile of kxk are processed by the same MLPs parameters
        nci = highres.shape[1]
        out = rearrange(highres,'b c h w -> b h w c')
        num_layers = len(self.channels) - 1
        for idx, nci in enumerate(self.channels[:-1]):
            nco = self.channels[idx + 1]

            # Select params in lowres buffer
            bstart, bstop = self._get_bias_indices(idx)
            b_ = lr_params[:, bstart:bstop]
            out = self.net[idx](out)
            b_ = b_.permute(0, 2, 3, 1)
            b_ = b_.repeat_interleave(2, dim=1)
            b_ = b_.repeat_interleave(2, dim=2)
            out = out + b_
            # Apply RelU non-linearity in all but the last layer, and tanh in the last
            if idx < num_layers - 1:
                out = th.nn.functional.leaky_relu(out, 0.01, inplace=True)
            else:
                out = F.tanh(out)
        out = rearrange(out,'b h w c-> b c h w')
        return out

class ASAPFPNHRStream(th.nn.Module):
    """Addaptive pixel-wise MLPs"""
    def __init__(self, cell_size, num_inputs=13, num_outputs=3, width=64, depth=5, coordinates="cosine"):
        super(ASAPFPNHRStream, self).__init__()

        self.pe_factor = 6
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.width = width
        self.depth = depth
        self.coordinates = coordinates
        self.xy_coords = None
        self.channels = []
        self._set_channels()
        self.num_params = 0
        self.splits = {}
        self._set_num_params()
        self.cell_size = cell_size

    @property  # for backward compatibility
    def ds(self):
        return self.downsampling


    def _set_channels(self):
        """Compute and store the hr-stream layer dimensions."""
        in_ch = self.num_inputs
        if self.coordinates == "cosine":
            in_ch += 4*(self.pe_factor+1)
        self.channels = [in_ch]

        for _ in range(self.depth - 1):  # intermediate layer -> cste size
            self.channels.append(self.width)
        # output layer
        self.channels.append(self.num_outputs)
    
    def _set_num_params(self):
        nparams = 0
        self.splits = {
            "biases": [],
            "weights": [],
        }

        # go over input/output channels for each layer
        idx = 0
        for layer, nci in enumerate(self.channels[:-1]):
            nco = self.channels[layer + 1]
            nparams += nco  # FC biases
            self.splits["biases"].append((idx, idx + nco))
            idx += nco

            nparams += nci * nco  # FC weights
            self.splits["weights"].append((idx, idx + nco * nci))
            idx += nco * nci

        self.num_params = nparams

    def _get_weight_indices(self, idx):
        return self.splits["weights"][idx]

    def _get_bias_indices(self, idx):
        return self.splits["biases"][idx]

    def forward(self, highres, lr_params):
        assert lr_params.shape[1] == self.num_params, "incorrect input params"
        # Fetch sizes
        k = self.cell_size
        bs, _, h, w = highres.shape
        bs, _, h_lr, w_lr = lr_params.shape

        # Spatial encoding
        if not(self.coordinates is None):
            if self.xy_coords is None:
                #self.xy_coords = _get_coords(bs, h, w, highres.device, self.ds, self.coordinates)
                self.xy_coords = _get_coords_global(bs, h, w, highres.device, self.pe_factor)
            highres = th.cat([highres, self.xy_coords], 1)
        
        # Split input in tiles of size kxk according to the NN interp factor (the total downsampling factor),
        # with channels last (for matmul)
        # all pixels within a tile of kxk are processed by the same MLPs parameters
        nci = highres.shape[1]
        # bs, 5 rgbxy, h//k=h_lr, w//k=w_lr, k, k
        tiles = highres.unfold(2, k, k).unfold(3, k, k)
        tiles = tiles.permute(0, 2, 3, 4, 5, 1).contiguous().view(
            bs, h_lr, w_lr, int(k * k), nci)
        out = tiles
        num_layers = len(self.channels) - 1
        for idx, nci in enumerate(self.channels[:-1]):
            nco = self.channels[idx + 1]

            # Select params in lowres buffer
            bstart, bstop = self._get_bias_indices(idx)
            wstart, wstop = self._get_weight_indices(idx)

            w_ = lr_params[:, wstart:wstop]
            b_ = lr_params[:, bstart:bstop]

            w_ = w_.permute(0, 2, 3, 1).view(bs, h_lr, w_lr, nci, nco)
            b_ = b_.permute(0, 2, 3, 1).view(bs, h_lr, w_lr, 1, nco)
            out = th.matmul(out, w_) + b_

            # Apply RelU non-linearity in all but the last layer, and tanh in the last
            if idx < num_layers - 1:
                out = th.nn.functional.leaky_relu(out, 0.01, inplace=True)
            else:
                out = F.tanh(out)
        # reorder the tiles in their correct position, and put channels first
        out = out.view(bs, h_lr, w_lr, k, k, self.num_outputs).permute(
            0, 5, 1, 3, 2, 4)
        out = out.contiguous().view(bs, self.num_outputs, h, w)

        return out

class ASAPNetsHRStream(th.nn.Module):
    """Addaptive pixel-wise MLPs"""
    def __init__(self, downsampling,
                 num_inputs=13, num_outputs=3, width=64, depth=5, coordinates="cosine",
                 no_one_hot=False, lr_instance=False):
        super(ASAPNetsHRStream, self).__init__()

        self.lr_instance = lr_instance
        self.downsampling = downsampling
        self.pe_factor = 6
        self.num_inputs = num_inputs - (1 if self.lr_instance else 0)
        self.num_outputs = num_outputs
        self.width = width
        self.depth = depth
        self.coordinates = coordinates
        self.xy_coords = None
        self.no_one_hot = no_one_hot
        self.channels = []
        self._set_channels()
        self.num_params = 0
        self.splits = {}
        self._set_num_params()

    @property  # for backward compatibility
    def ds(self):
        return self.downsampling


    def _set_channels(self):
        """Compute and store the hr-stream layer dimensions."""
        in_ch = self.num_inputs
        #if self.coordinates == "cosine":
        #    in_ch += int(4*log2(self.downsampling))
        #elif self.coordinates == "siren":
        #    in_ch += 2
        ### globle coord test
        if self.coordinates == "cosine":
            in_ch += 4*(self.pe_factor+1)
        self.channels = [in_ch]

        for _ in range(self.depth - 1):  # intermediate layer -> cste size
            self.channels.append(self.width)
        # output layer
        self.channels.append(self.num_outputs)
    
    def _set_num_params(self):
        nparams = 0
        self.splits = {
            "biases": [],
            "weights": [],
        }

        # go over input/output channels for each layer
        idx = 0
        for layer, nci in enumerate(self.channels[:-1]):
            nco = self.channels[layer + 1]
            nparams += nco  # FC biases
            self.splits["biases"].append((idx, idx + nco))
            idx += nco

            nparams += nci * nco  # FC weights
            self.splits["weights"].append((idx, idx + nco * nci))
            idx += nco * nci

        self.num_params = nparams

    def _get_weight_indices(self, idx):
        return self.splits["weights"][idx]

    def _get_bias_indices(self, idx):
        return self.splits["biases"][idx]

    def forward(self, highres, lr_params):
        assert lr_params.shape[1] == self.num_params, "incorrect input params"
        if self.lr_instance:
            highres = highres[:, :-1, :, :]
        # Fetch sizes
        k = int(self.downsampling)
        bs, _, h, w = highres.shape
        bs, _, h_lr, w_lr = lr_params.shape
       
       # Spatial encoding
        if not(self.coordinates is None):
            if self.xy_coords is None:
                #self.xy_coords = _get_coords(bs, h, w, highres.device, self.ds, self.coordinates)
                self.xy_coords = _get_coords_global(bs, h, w, highres.device, self.pe_factor)
            highres = th.cat([highres, self.xy_coords], 1)
        
        # Split input in tiles of size kxk according to the NN interp factor (the total downsampling factor),
        # with channels last (for matmul)
        # all pixels within a tile of kxk are processed by the same MLPs parameters
        nci = highres.shape[1]
        # bs, 5 rgbxy, h//k=h_lr, w//k=w_lr, k, k
        tiles = highres.unfold(2, k, k).unfold(3, k, k)
        tiles = tiles.permute(0, 2, 3, 4, 5, 1).contiguous().view(
            bs, h_lr, w_lr, int(k * k), nci)
        out = tiles
        num_layers = len(self.channels) - 1
        for idx, nci in enumerate(self.channels[:-1]):
            nco = self.channels[idx + 1]

            # Select params in lowres buffer
            bstart, bstop = self._get_bias_indices(idx)
            wstart, wstop = self._get_weight_indices(idx)

            w_ = lr_params[:, wstart:wstop]
            b_ = lr_params[:, bstart:bstop]

            w_ = w_.permute(0, 2, 3, 1).view(bs, h_lr, w_lr, nci, nco)
            b_ = b_.permute(0, 2, 3, 1).view(bs, h_lr, w_lr, 1, nco)
            out = th.matmul(out, w_) + b_

            # Apply RelU non-linearity in all but the last layer, and tanh in the last
            if idx < num_layers - 1:
                out = th.nn.functional.leaky_relu(out, 0.01, inplace=True)
            else:
                out = F.tanh(out)
        # reorder the tiles in their correct position, and put channels first
        out = out.view(bs, h_lr, w_lr, k, k, self.num_outputs).permute(
            0, 5, 1, 3, 2, 4)
        out = out.contiguous().view(bs, self.num_outputs, h, w)

        return out

class ASAPNetsHRSirenStream(th.nn.Module):
    """Addaptive pixel-wise MLPs"""
    def __init__(self, downsampling,
                 num_inputs=13, num_outputs=3, width=64, depth=5, coordinates="cosine",
                 no_one_hot=False, lr_instance=False):
        super(ASAPNetsHRSirenStream, self).__init__()
        print("siren")
        self.lr_instance = lr_instance
        self.downsampling = downsampling
        self.num_inputs = num_inputs - (1 if self.lr_instance else 0)
        self.num_outputs = num_outputs
        self.pe_factor = 6
        self.width = width
        self.depth = depth
        self.coordinates = coordinates
        self.xy_coords = None
        self.no_one_hot = no_one_hot
        self.channels = []
        self._set_channels()
        self.nl = Sine()

        self.num_params = 0
        self.splits = {}
        self._set_num_params()

    @property  # for backward compatibility
    def ds(self):
        return self.downsampling


    def _set_channels(self):
        """Compute and store the hr-stream layer dimensions."""
        in_ch = self.num_inputs
        #if self.coordinates == "cosine":
        #    in_ch += int(4*log2(self.downsampling))
        #elif self.coordinates == "siren":
        #    in_ch += 2
        ### globle coord test
        if self.coordinates == "cosine":
            #in_ch += int(4*(log2(self.pe_factor)+1))
            in_ch += 4*(self.pe_factor+1)
        
        self.channels = [in_ch]

        print(in_ch)
        for _ in range(self.depth - 1):  # intermediate layer -> cste size
            self.channels.append(self.width)
        # output layer
        self.channels.append(self.num_outputs)
    
    def _set_num_params(self):
        nparams = 0
        self.splits = {
            "biases": [],
            "weights": [],
        }

        # go over input/output channels for each layer
        idx = 0
        for layer, nci in enumerate(self.channels[:-1]):
            nco = self.channels[layer + 1]
            nparams += nco  # FC biases
            self.splits["biases"].append((idx, idx + nco))
            idx += nco

            nparams += nci * nco  # FC weights
            self.splits["weights"].append((idx, idx + nco * nci))
            idx += nco * nci

        self.num_params = nparams

    def _get_weight_indices(self, idx):
        return self.splits["weights"][idx]

    def _get_bias_indices(self, idx):
        return self.splits["biases"][idx]

    def forward(self, highres, lr_params):
        assert lr_params.shape[1] == self.num_params, "incorrect input params"

        if self.lr_instance:
            highres = highres[:, :-1, :, :]

        # Fetch sizes
        k = int(self.downsampling)
        bs, _, h, w = highres.shape
        bs, _, h_lr, w_lr = lr_params.shape
        
        # Spatial encoding
        if not(self.coordinates is None):
            if self.xy_coords is None:
                #self.xy_coords = _get_coords_global(bs, h, w, highres.device, 6)
                self.xy_coords = _get_coords_global(bs, h, w, highres.device, self.pe_factor)
            highres = th.cat([highres, self.xy_coords], 1)
        # Split input in tiles of size kxk according to the NN interp factor (the total downsampling factor),
        # with channels last (for matmul)
        # all pixels within a tile of kxk are processed by the same MLPs parameters
        nci = highres.shape[1]
        # bs, 5 rgbxy, h//k=h_lr, w//k=w_lr, k, k
        tiles = highres.unfold(2, k, k).unfold(3, k, k)
        tiles = tiles.permute(0, 2, 3, 4, 5, 1).contiguous().view(
            bs, h_lr, w_lr, int(k * k), nci)
        out = tiles
        num_layers = len(self.channels) - 1
        for idx, nci in enumerate(self.channels[:-1]):
            nco = self.channels[idx + 1]

            # Select params in lowres buffer
            bstart, bstop = self._get_bias_indices(idx)
            wstart, wstop = self._get_weight_indices(idx)

            w_ = lr_params[:, wstart:wstop]
            b_ = lr_params[:, bstart:bstop]

            w_ = w_.permute(0, 2, 3, 1).view(bs, h_lr, w_lr, nci, nco)
            b_ = b_.permute(0, 2, 3, 1).view(bs, h_lr, w_lr, 1, nco)
            
            out = th.matmul(out, w_) + b_

            # Apply RelU non-linearity in all but the last layer, and tanh in the last
            #if idx ==0:
            if idx < num_layers - 1:
                #out = th.nn.functional.leaky_relu(out, 0.01, inplace=True)
                out = self.nl(out)
            else:
                out = F.tanh(out)
                #out = self.nl(out)
        # reorder the tiles in their correct position, and put channels first
        out = out.view(bs, h_lr, w_lr, k, k, self.num_outputs).permute(
            0, 5, 1, 3, 2, 4)
        out = out.contiguous().view(bs, self.num_outputs, h, w)

        return out
