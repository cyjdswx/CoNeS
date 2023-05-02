import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ASAPNetsBlock as ASAPNetsBlock
from models.networks.generator import ASAPNetsLRStream,ASAPNetsHRStream, _get_coords, ASAPfunctaHRStream, ASAPfunctaHRStreamfulres, Encoder_fpn3, Bottleneck_in, Bottleneckdropout, Encoder_fpn4
from models.networks.architecture import MySeparableBilinearDownsample as BilinearDownsample
import torch.nn.utils.spectral_norm as spectral_norm
import torch as th
from einops import rearrange
from math import pi
from math import log2
from copy import deepcopy
import time
import numpy as np

from torch.nn import init

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
        #return torch.sin(30 * input)
        return torch.sin(input)

class ASAPNetsMultiSeg_nnunet_feat(BaseNetwork):
    def __init__(self, opt, n_classes, dropout=False):
        super(ASAPNetsMultiSeg_nnunet_feat, self).__init__()
        #if lr_stream is None or hr_stream is None:
        #    lr_stream = dict()
        #    hr_stream = dict()
        self.num_inputs = opt.label_nc
        self.num_outputs = opt.output_nc
        self.n_classes = n_classes
        self.learned_ds_factor = opt.learned_ds_factor #(S2 in sec. 3.2)
        self.gpu_ids = opt.gpu_ids
        # calculates the total downsampling factor in order to get the final low-res grid of parameters (S=S1xS2 in sec. 3.2)
        #self.downsampling = pow(2, opt.ds_factor)
        if dropout:
            #dropout_op_kwargs = {'p':0.2, 'inplace':True}
            dropout_op_kwargs = {'p':0, 'inplace':True}
            self.block = Bottleneckdropout
        else:
            dropout_op_kwargs = {'p':0, 'inplace':True}
            self.block = Bottleneck_in
        norm_op_kwargs = {'eps':1e-05, 'affine':True}
        net_nonline_kwargs = {'negative_slope': 0.01, 'inplace': True}
        num_pool_op_kernel_size = [[2,2],[2,2],[2,2],[2,2],[2,2]]
        net_conv_kernel_size = [[3,3],[3,3],[3,3],[3,3],[3,3],[3,3]]
        
        self.lowres_stream = Encoder_fpn4(num_in=self.num_inputs, block=self.block, num_blocks=[2,4,23,3])
        self.highres_stream = ASAPfunctaHRStreamfulres( num_inputs=self.num_inputs,
                                               num_outputs=opt.output_nc, width=opt.hr_width,
                                               depth=opt.hr_depth, coordinates=opt.hr_coor)
        
        num_params = self.highres_stream.num_params
        self.latlayers = nn.Conv2d(256, num_params, kernel_size=1, stride=1,padding=0)
        self.seg_stream = SegNet(input_channels=4+256, base_num_features=32, num_classes=4, num_pool=5,num_conv_per_stage=2,\
                feat_map_mul_on_downscale=2,conv_op=torch.nn.Conv2d, norm_op=torch.nn.InstanceNorm2d, norm_op_kwargs=norm_op_kwargs,\
                dropout_op=torch.nn.Dropout2d,dropout_op_kwargs=dropout_op_kwargs, nonlin=torch.nn.LeakyReLU,\
                nonlin_kwargs=net_nonline_kwargs,deep_supervision=True, dropout_in_localization=False,final_nonlin=lambda x:x,\
                #nonlin_kwargs=net_nonline_kwargs,deep_supervision=True, dropout_in_localization=True,final_nonlin=lambda x:x,\
               weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=num_pool_op_kernel_size, conv_kernel_sizes=net_conv_kernel_size,\
               upscale_logits=False,convolutional_pooling=True,convolutional_upsampling=True)
        
        self.init = InitWeights_me(opt.init_type, opt.init_variance)
        self.lowres_stream.apply(self.init)
        self.latlayers.apply(self.init)
        self.highres_stream.apply(self.init)

    def use_gpu(self):
        return len(self.gpu_ids) > 0

    def forward(self, highres):
        features = self.lowres_stream(highres)
        features = self.latlayers(features)
        output = self.highres_stream(highres, features)
        seg_input = torch.cat([highres, features], dim=1)
        seg = self.seg_stream(seg_input)
        return output, seg, features#, lowres

class ASAPNetsMultiSeg_nnunetonly_feat(BaseNetwork):
    def __init__(self, opt, n_classes, dropout=False):
        super(ASAPNetsMultiSeg_nnunetonly_feat, self).__init__()
        self.num_inputs = opt.label_nc
        self.num_outputs = opt.output_nc
        self.n_classes = n_classes
        self.learned_ds_factor = opt.learned_ds_factor #(S2 in sec. 3.2)
        self.gpu_ids = opt.gpu_ids
        # calculates the total downsampling factor in order to get the final low-res grid of parameters (S=S1xS2 in sec. 3.2)
        #self.downsampling = pow(2, opt.ds_factor)
        if dropout:
            dropout_op_kwargs = {'p':0.2, 'inplace':True}
            #dropout_op_kwargs = {'p':0, 'inplace':True}
            self.block = Bottleneckdropout
        else:
            dropout_op_kwargs = {'p':0, 'inplace':True}
            self.block = Bottleneck_in
        norm_op_kwargs = {'eps':1e-05, 'affine':True}
        net_nonline_kwargs = {'negative_slope': 0.01, 'inplace': True}
        num_pool_op_kernel_size = [[2,2],[2,2],[2,2],[2,2],[2,2]]
        net_conv_kernel_size = [[3,3],[3,3],[3,3],[3,3],[3,3],[3,3]]
        
        self.lowres_stream = Encoder_fpn4(num_in=self.num_inputs, block=self.block, num_blocks=[2,4,23,3])
        self.highres_stream = ASAPfunctaHRStreamfulres( num_inputs=self.num_inputs,
                                               num_outputs=opt.output_nc, width=opt.hr_width,
                                               depth=opt.hr_depth, coordinates=opt.hr_coor)
        
        num_params = self.highres_stream.num_params
        self.latlayers = nn.Conv2d(256, num_params, kernel_size=1, stride=1,padding=0)
        self.seg_stream = SegNet(input_channels=4+256, base_num_features=32, num_classes=4, num_pool=5,num_conv_per_stage=2,\
                feat_map_mul_on_downscale=2,conv_op=torch.nn.Conv2d, norm_op=torch.nn.InstanceNorm2d, norm_op_kwargs=norm_op_kwargs,\
                dropout_op=torch.nn.Dropout2d,dropout_op_kwargs=dropout_op_kwargs, nonlin=torch.nn.LeakyReLU,\
                nonlin_kwargs=net_nonline_kwargs,deep_supervision=True, dropout_in_localization=False,final_nonlin=lambda x:x,\
                #nonlin_kwargs=net_nonline_kwargs,deep_supervision=True, dropout_in_localization=True,final_nonlin=lambda x:x,\
               weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=num_pool_op_kernel_size, conv_kernel_sizes=net_conv_kernel_size,\
               upscale_logits=False,convolutional_pooling=True,convolutional_upsampling=True)
        
        self.init = InitWeights_me(opt.init_type, opt.init_variance)
        self.lowres_stream.apply(self.init)
        self.latlayers.apply(self.init)
        self.highres_stream.apply(self.init)

    def use_gpu(self):
        return len(self.gpu_ids) > 0

    def forward(self, highres):
        features = self.lowres_stream(highres)
        features = self.latlayers(features)
        seg_input = torch.cat([highres, features], dim=1)
        seg = self.seg_stream(seg_input)
        return None, seg, features

class ASAPNetsMultiSeg_nnunet_fullres(BaseNetwork):
    def __init__(self, opt, n_classes, use_dropout=False):
        super(ASAPNetsMultiSeg_nnunet_fullres, self).__init__()
        self.num_inputs = opt.label_nc
        self.num_outputs = opt.output_nc
        self.n_classes = n_classes
        self.use_dropout = use_dropout
        self.learned_ds_factor = opt.learned_ds_factor #(S2 in sec. 3.2)
        self.gpu_ids = opt.gpu_ids
        if self.use_dropout:
            dropout_op_kwargs = {'p':0, 'inplace':True}
            self.block = Bottleneckdropout
        else:
            dropout_op_kwargs = {'p':0, 'inplace':True}
            self.block = Bottleneck_in

        self.lowres_stream = Encoder_fpn4(num_in=self.num_inputs, block=self.block, num_blocks=[2,4,23,3])
        
        self.highres_stream = ASAPfunctaHRStreamfulres( num_inputs=self.num_inputs,
                                               num_outputs=opt.output_nc, width=opt.hr_width,
                                               depth=opt.hr_depth, coordinates=opt.hr_coor)
        num_params = self.highres_stream.num_params
        self.latlayers = nn.Conv2d(256, num_params, kernel_size=1, stride=1,padding=0)
        norm_op_kwargs = {'eps':1e-05, 'affine':True}
        net_nonline_kwargs = {'negative_slope': 0.01, 'inplace': True}
        num_pool_op_kernel_size = [[2,2],[2,2],[2,2],[2,2],[2,2]]
        net_conv_kernel_size = [[3,3],[3,3],[3,3],[3,3],[3,3],[3,3]]
        self.seg_stream = SegNet(input_channels=4, base_num_features=32, num_classes=4, num_pool=5,num_conv_per_stage=2,\
                feat_map_mul_on_downscale=2,conv_op=torch.nn.Conv2d, norm_op=torch.nn.InstanceNorm2d, norm_op_kwargs=norm_op_kwargs,\
                dropout_op=torch.nn.Dropout2d,dropout_op_kwargs=dropout_op_kwargs, nonlin=torch.nn.LeakyReLU,\
                nonlin_kwargs=net_nonline_kwargs,deep_supervision=True, dropout_in_localization=False,final_nonlin=lambda x:x,\
               weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=num_pool_op_kernel_size, conv_kernel_sizes=net_conv_kernel_size,\
               upscale_logits=False,convolutional_pooling=True,convolutional_upsampling=True)
        
        self.init = InitWeights_me(opt.init_type, opt.init_variance)
        self.lowres_stream.apply(self.init)
        self.latlayers.apply(self.init)
        self.highres_stream.apply(self.init)

    def use_gpu(self):
        return len(self.gpu_ids) > 0

    def forward(self, highres):
        features = self.lowres_stream(highres)
        features = self.latlayers(features)
        output = self.highres_stream(highres, features)
        seg = self.seg_stream(output)
        return output, seg, features#, lowres

class ASAPNetsMultiSeg_nnunet(BaseNetwork):
    def __init__(self, opt, n_classes, use_dropout=False):
        super(ASAPNetsMultiSeg_nnunet, self).__init__()
        self.num_inputs = opt.label_nc
        self.num_outputs = opt.output_nc
        self.n_classes = n_classes
        self.use_dropout = use_dropout
        self.learned_ds_factor = opt.learned_ds_factor #(S2 in sec. 3.2)
        self.gpu_ids = opt.gpu_ids
        if self.use_dropout:
            dropout_op_kwargs = {'p':0, 'inplace':True}
            self.block = Bottleneckdropout
        else:
            dropout_op_kwargs = {'p':0, 'inplace':True}
            self.block = Bottleneck_in

        self.lowres_stream = Encoder_fpn4(num_in=self.num_inputs, block=self.block, num_blocks=[2,4,23,3])
        
        self.highres_stream = ASAPfunctaHRStreamfulres( num_inputs=self.num_inputs,
                                               num_outputs=opt.output_nc, width=opt.hr_width,
                                               depth=opt.hr_depth, coordinates=opt.hr_coor)
        num_params = self.highres_stream.num_params
        self.latlayers = nn.Conv2d(256, num_params, kernel_size=1, stride=1,padding=0)
        norm_op_kwargs = {'eps':1e-05, 'affine':True}
        net_nonline_kwargs = {'negative_slope': 0.01, 'inplace': True}
        num_pool_op_kernel_size = [[2,2],[2,2],[2,2],[2,2],[2,2]]
        net_conv_kernel_size = [[3,3],[3,3],[3,3],[3,3],[3,3],[3,3]]
        self.seg_stream = SegNet(input_channels=4, base_num_features=32, num_classes=4, num_pool=5,num_conv_per_stage=2,\
                feat_map_mul_on_downscale=2,conv_op=torch.nn.Conv2d, norm_op=torch.nn.InstanceNorm2d, norm_op_kwargs=norm_op_kwargs,\
                dropout_op=torch.nn.Dropout2d,dropout_op_kwargs=dropout_op_kwargs, nonlin=torch.nn.LeakyReLU,\
                nonlin_kwargs=net_nonline_kwargs,deep_supervision=True, dropout_in_localization=False,final_nonlin=lambda x:x,\
               weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=num_pool_op_kernel_size, conv_kernel_sizes=net_conv_kernel_size,\
               upscale_logits=False,convolutional_pooling=True,convolutional_upsampling=True)
        
        self.init = InitWeights_me(opt.init_type, opt.init_variance)
        self.lowres_stream.apply(self.init)
        self.latlayers.apply(self.init)
        self.highres_stream.apply(self.init)

    def use_gpu(self):
        return len(self.gpu_ids) > 0

    def forward(self, highres):
        features = self.lowres_stream(highres)
        features = self.latlayers(features)
        output = self.highres_stream(highres, features)
        seg_input = torch.cat([highres, output], dim=1)
        seg = self.seg_stream(seg_input)
        return output, seg, features#, lowres

class ASAPNetsMultiSeg(BaseNetwork):
    def __init__(self, opt, n_classes, hr_stream=None, lr_stream=None, fast=False):
        super(ASAPNetsMultiSeg, self).__init__()
        if lr_stream is None or hr_stream is None:
            lr_stream = dict()
            hr_stream = dict()
        self.num_inputs = opt.label_nc
        self.num_outputs = opt.output_nc
        self.n_classes = n_classes
        self.learned_ds_factor = opt.learned_ds_factor #(S2 in sec. 3.2)
        self.gpu_ids = opt.gpu_ids
        # calculates the total downsampling factor in order to get the final low-res grid of parameters (S=S1xS2 in sec. 3.2)
        #self.downsampling = pow(2, opt.ds_factor)
        self.lowres_stream = Encoder_fpn3(num_in=self.num_inputs, block=Bottleneck_in, num_blocks=[2,4,23,3])
        
        self.highres_stream = ASAPfunctaHRStream(2, num_inputs=self.num_inputs,
                                               num_outputs=opt.output_nc, width=opt.hr_width,
                                               depth=opt.hr_depth, coordinates=opt.hr_coor,**hr_stream)
        num_params = self.highres_stream.num_params
        self.latlayers = nn.Conv2d(256, num_params, kernel_size=1, stride=1,padding=0)
        self.seg_stream = UNet(n_channels=self.num_inputs + self.num_outputs, n_classes=n_classes, bilinear=True)

    def use_gpu(self):
        return len(self.gpu_ids) > 0

    def forward(self, highres):
        features = self.lowres_stream(highres)
        features = self.latlayers(features)
        output = self.highres_stream(highres, features)
        seg_input = torch.cat([highres, output], dim=1)
        seg = self.seg_stream(seg_input)
        return output, seg, features#, lowres

class ASAPNetsSeg(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='instanceaffine')
        parser.set_defaults(no_instance_dist=True)
        #parser.set_defaults(hr_coor="cosine")
        return parser

    def __init__(self, opt, n_classes, hr_stream=None, lr_stream=None, fast=False):
        super(ASAPNetsSeg, self).__init__()
        if lr_stream is None or hr_stream is None:
            lr_stream = dict()
            hr_stream = dict()
        self.num_inputs = opt.label_nc
        self.num_outputs = opt.output_nc
        self.n_classes = n_classes
        self.learned_ds_factor = opt.learned_ds_factor #(S2 in sec. 3.2)
        self.gpu_ids = opt.gpu_ids
        # calculates the total downsampling factor in order to get the final low-res grid of parameters (S=S1xS2 in sec. 3.2)
        self.downsampling = pow(2, opt.ds_factor)
        self.lowres_stream = Encoder_fpn3(num_in=self.num_inputs, block=Bottleneck_in, num_blocks=[2,4,23,3])
        
        self.highres_stream = ASAPNetsHRSegStream(self.downsampling, num_inputs=self.num_inputs,
                                               num_outputs=self.num_outputs, n_classes=self.n_classes, width=opt.hr_width,
                                               depth=opt.hr_depth, coordinates=opt.hr_coor,
                                               **hr_stream)
        num_params = self.highres_stream.num_params
        self.latlayers = nn.Conv2d(256, num_params, kernel_size=1, stride=1,padding=0)

    def use_gpu(self):
        return len(self.gpu_ids) > 0

    def forward(self, highres):
        features = self.lowres_stream(highres)
        features = self.latlayers(features)
        output, seg = self.highres_stream(highres, features)
        return output, seg, features#, lowres

class ASAPNetsSegGenerator_v2(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='instanceaffine')
        parser.set_defaults(no_instance_dist=True)
        return parser

    def __init__(self, opt, n_classes, hr_stream=None, lr_stream=None, fast=False):
        super(ASAPNetsSegGenerator_v2, self).__init__()
        if lr_stream is None or hr_stream is None:
            lr_stream = dict()
            hr_stream = dict()
        self.num_inputs = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if (opt.no_instance_edge & opt.no_instance_dist) else 1)
        self.num_outputs = opt.output_nc
        self.n_classes = n_classes
        self.lr_instance = opt.lr_instance
        self.learned_ds_factor = opt.learned_ds_factor #(S2 in sec. 3.2)
        self.gpu_ids = opt.gpu_ids
        img_size = (128,160)
        # calculates the total downsampling factor in order to get the final low-res grid of parameters (S=S1xS2 in sec. 3.2)
        #self.downsampling = opt.crop_size // (16 * opt.aspect_ratio)
        self.downsampling = min(img_size) // opt.lowest_ds_factor
        print('down', self.downsampling)

        self.highres_stream = ASAPNetsHRStream(self.downsampling, num_inputs=self.num_inputs,
                                               num_outputs=self.num_outputs, width=opt.hr_width,
                                               depth=opt.hr_depth, coordinates=opt.hr_coor,
                                               no_one_hot=opt.no_one_hot, lr_instance=opt.lr_instance,
                                               **hr_stream)

        num_params = self.highres_stream.num_params
        num_inputs_lr = self.highres_stream.num_inputs + (1 if opt.lr_instance else 0)
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        self.lowres_stream = ASAPNetsLRStream(num_inputs_lr, num_params, norm_layer, width=opt.lr_width,
                                              max_width=opt.lr_max_width, depth=opt.lr_depth,
                                              learned_ds_factor=opt.learned_ds_factor,
                                              reflection_pad=opt.reflection_pad, **lr_stream)
        self.seg_stream = UNet(n_channels=self.num_inputs + self.num_outputs, n_classes=n_classes, bilinear=True)
    
    def use_gpu(self):
        return len(self.gpu_ids) > 0

    def get_lowres(self, im):
        """Creates a lowres version of the input."""
        device = self.use_gpu()
        if(self.learned_ds_factor != self.downsampling):
            myds = BilinearDownsample(int(self.downsampling//self.learned_ds_factor), self.num_inputs,device)
            return myds(im)
        else:
            return im

    def forward(self, highres):
        lowres = self.get_lowres(highres)
        lr_features = self.lowres_stream(lowres)
        output  = self.highres_stream(highres, lr_features)
        seg_input = torch.cat([highres,output], dim=1)
        seg = self.seg_stream(seg_input)
        return output, seg, lr_features

class ASAPNetsHRSegStream(th.nn.Module):
    """Addaptive pixel-wise MLPs"""
    def __init__(self, downsampling,
                 num_inputs=13, num_outputs=3,n_classes = 4, width=64, depth=5, coordinates="cosine"):
        super(ASAPNetsHRSegStream, self).__init__()

        self.downsampling = downsampling
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs 
        self.n_classes = n_classes
        self.width = width
        self.depth = depth
        self.coordinates = coordinates
        self.xy_coords = None
        self.channels = []
        self._set_channels()

        self.num_params = 0
        self.splits = {}
        self._set_num_params()
        self.net = nn.ModuleList()
        for i in range(len(self.channels) - 1):
            self.net.append(nn.Linear(self.channels[i], self.channels[i+1], bias=True))

    @property  # for backward compatibility
    def ds(self):
        return self.downsampling


    def _set_channels(self):
        """Compute and store the hr-stream layer dimensions."""
        in_ch = self.num_inputs
        if self.coordinates == "cosine":
            in_ch += int(4*log2(self.downsampling))
        #elif self.coordinates == "siren":
        #    in_ch += 2
        ### globle coord test
        #if self.coordinates == "cosine":
        #    in_ch += int(4*(log2(8)+1))
        self.channels = [in_ch]

        for _ in range(self.depth - 1):  # intermediate layer -> cste size
            self.channels.append(self.width)
        # output layer
        self.channels.append(self.num_outputs+ self.n_classes)
    
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
        bs, _, h, w = highres.shape
        bs, _, h_lr, w_lr = lr_params.shape
        
        # Spatial encoding
        if not(self.coordinates is None):
            if self.xy_coords is None:
                self.xy_coords = _get_coords(bs, h, w, highres.device, self.ds, self.coordinates)
                #self.xy_coords = _get_coords_global(bs, h, w, highres.device, 8, '01')
            highres = th.cat([highres, self.xy_coords], 1)
        
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
                output, mask = torch.split(out, [self.num_outputs,self.n_classes], dim=3)
                output = F.tanh(output)
        
        output = rearrange(output, 'b h w c-> b c h w')
        mask = rearrange(mask, 'b h w c -> b c h w')
        return output, mask

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = in_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 256)
        self.up1 = Up(512, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

softmax_helper = lambda x: F.softmax(x, 1) 

class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope
   
    def __call__(self, module): 
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) \
                or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):     
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope) 
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0) 

    #def init_weights(self, init_type='normal', gain=0.02):
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

class ConvDropoutNormNonlin(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))

class StackedConvLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin):
        '''
        stacks ConvDropoutNormLReLU layers. initial_stride will only be applied to first layer in the stack. The other parameters affect all layers
        :param input_feature_channels:
        :param output_feature_channels:
        :param num_convs:
        :param dilation:
        :param kernel_size:
        :param padding:
        :param dropout:
        :param initial_stride:
        :param conv_op:
        :param norm_op:
        :param dropout_op:
        :param inplace:
        :param neg_slope:
        :param norm_affine:
        :param conv_bias:
        '''
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        super(StackedConvLayers, self).__init__()
        self.blocks = nn.Sequential(
            *([basic_block(input_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs_first_conv,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs)] +
              [basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 1)]))

    def forward(self, x):
        return self.blocks(x)

class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)

class SegNet(nn.Module):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440

    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False):
        super(SegNet, self).__init__()
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

        if conv_op == nn.Conv2d:
            upsample_mode = 'bilinear'
            pool_op = nn.MaxPool2d
            transpconv = nn.ConvTranspose2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        elif conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            transpconv = nn.ConvTranspose3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features

        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.td = []
        self.tu = []
        self.seg_outputs = []

        output_features = base_num_features
        input_features = input_channels

        for d in range(num_pool):
            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))

            output_features = min(output_features, self.max_num_features)

        # now the bottleneck.
        # determine the first stride
        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1]
        else:
            first_stride = None

        # the output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv
        if self.convolutional_upsampling:
            final_num_features = output_features
        else:
            final_num_features = self.conv_blocks_context[-1].output_channels

        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block)))

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        # now lets build the localization pathway
        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[
                -(2 + u)].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            if not self.convolutional_upsampling:
                self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            else:
                self.tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u + 1)],
                                          pool_op_kernel_sizes[-(u + 1)], bias=False))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            self.conv_blocks_localization.append(nn.Sequential(
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
            ))

        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(conv_op(self.conv_blocks_localization[ds][-1].output_channels, num_classes,
                                            1, 1, 0, 1, 1, seg_output_use_bias))

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(lambda x: x)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(
                self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)
            # self.apply(print_module_training_status)

    def forward(self, x):
        skips = []
        seg_outputs = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)

        x = self.conv_blocks_context[-1](x)

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]

