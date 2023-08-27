from json import decoder
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.generator_cat import Encoder_fpn4, Encoder_fpn4_ds
import torch as th
from math import pi

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

def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def _get_coords_global(bs, h, w, device, ds):
    """Creates the position encoding for the pixel-wise MLPs"""
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
        print(self.channels)
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
        #in_ch = 0
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
        bs, _, h, w = highres.shape
        bs, _, h_lr, w_lr = lr_params.shape
        # positional encoding
        if not(self.coordinates is None):
            if self.xy_coords is None:
                self.xy_coords = _get_coords_global(bs, h, w, highres.device, self.pe_factor)
            ##for ablation study
            highres = th.cat([highres, self.xy_coords], 1)
            #highres = self.xy_coords
        
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

class ASAPfunctaHRStreamfulres_ds(th.nn.Module):
    """Addaptive pixel-wise MLPs"""
    def __init__(self, num_inputs=13, num_outputs=3, width=64, depth=5, coordinates="cosine"):
        super(ASAPfunctaHRStreamfulres_ds, self).__init__()

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
        print(self.channels)
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
        #in_ch = 0
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
        bs, _, h, w = highres.shape
        bs, _, h_lr, w_lr = lr_params.shape
        # positional encoding
        if not(self.coordinates is None):
            if self.xy_coords is None:
                self.xy_coords = _get_coords_global(bs, h, w, highres.device, self.pe_factor)
            ##for ablation study
            highres = th.cat([highres, self.xy_coords], 1)
            #highres = self.xy_coords
        
        # Split input in tiles of size kxk according to the NN interp factor (the total downsampling factor),
        # with channels last (for matmul)
        # all pixels within a tile of kxk are processed by the same MLPs parameters
        nci = highres.shape[1]
        out = rearrange(highres,'b c h w -> b h w c')
        num_layers = len(self.channels) - 1
        lr_params = torch.repeat_interleave(lr_params,repeats=2, dim=2)
        lr_params = torch.repeat_interleave(lr_params,repeats=2, dim=3)
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

class ASAPfunctaHRStreamfulres_noint(th.nn.Module):
    """Addaptive pixel-wise MLPs"""
    def __init__(self, num_inputs=13, num_outputs=3, width=64, depth=5, coordinates="cosine"):
        super(ASAPfunctaHRStreamfulres_noint, self).__init__()

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
        print(self.channels)
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
        #in_ch = self.num_inputs
        in_ch = 0
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
        bs, _, h, w = highres.shape
        bs, _, h_lr, w_lr = lr_params.shape
        # positional encoding
        if not(self.coordinates is None):
            if self.xy_coords is None:
                self.xy_coords = _get_coords_global(bs, h, w, highres.device, self.pe_factor)
            ##for ablation study
            #highres = th.cat([highres, self.xy_coords], 1)
            highres = self.xy_coords
        
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

class decoderfullres(th.nn.Module):
    """Addaptive pixel-wise MLPs"""
    def __init__(self, num_inputs=13, num_outputs=3, width=64, depth=5, coordinates="cosine"):
        super(decoderfullres, self).__init__()

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
        #self.biases = []
        print(self.channels)
        self._set_num_params()
        #self.net = nn.ModuleList()
        #for i in range(len(self.channels) - 1):
        #    self.net.append(nn.Linear(self.channels[i], self.channels[i+1], bias=True))

    @property  # for backward compatibility
    def ds(self):
        return self.downsampling

    def _set_channels(self):
        """Compute and store the hr-stream layer dimensions."""
        in_ch = 0
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

    def _get_bias_indices(self, idx):
        return self.splits["biases"][idx]
        #return self.biases[idx]
    def _get_weight_indices(self, idx):
        return self.splits["weights"][idx]
    def forward(self, highres, lr_params):
        assert lr_params.shape[1] == self.num_params, "incorrect input params"
        bs, _, h, w = highres.shape
        bs, _, h_lr, w_lr = lr_params.shape
        # positional encoding
        if not(self.coordinates is None):
            if self.xy_coords is None:
                self.xy_coords = _get_coords_global(bs, h, w, highres.device, self.pe_factor)
            ##for ablation study
            #highres = th.cat([highres, self.xy_coords], 1)
            #highres = self.xy_coords
        
        # Split input in tiles of size kxk according to the NN interp factor (the total downsampling factor),
        # with channels last (for matmul)
        # all pixels within a tile of kxk are processed by the same MLPs parameters
        #nci = highres.shape[1]
        nci = self.xy_coords.shape[1]
        #out = rearrange(highres,'b c h w -> b h w c')
        out = rearrange(self.xy_coords,'b c h w -> b h w c')
        num_layers = len(self.channels) - 1
        out = torch.unsqueeze(out,dim=3)
        for idx, nci in enumerate(self.channels[:-1]):
            nco = self.channels[idx + 1]
            # Select params in lowres buffer
            bstart, bstop = self._get_bias_indices(idx)
            wstart, wstop = self._get_weight_indices(idx)
            w_ = lr_params[:, wstart:wstop]
            b_ = lr_params[:, bstart:bstop]
            #out = self.net[idx](out)
            w_ = w_.permute(0, 2, 3, 1).view(bs, h_lr, w_lr, nci, nco)
            b_ = b_.permute(0, 2, 3, 1).view(bs, h_lr, w_lr, 1, nco)
            out = th.matmul(out, w_) + b_
            #out = out + b_
            # Apply RelU non-linearity in all but the last layer, and tanh in the last
            if idx < num_layers - 1:
                out = th.nn.functional.leaky_relu(out, 0.01, inplace=True)
            else:
                out = F.tanh(out)
        out = torch.squeeze(out,dim=3)
        out = rearrange(out,'b h w c-> b c h w')
        return out

class decoderfullres_int(th.nn.Module):
    """Addaptive pixel-wise MLPs"""
    def __init__(self, num_inputs=13, num_outputs=3, width=64, depth=5, coordinates="cosine"):
        super(decoderfullres_int, self).__init__()

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
        #self.biases = []
        print(self.channels)
        self._set_num_params()

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

    def _get_bias_indices(self, idx):
        return self.splits["biases"][idx]
        #return self.biases[idx]
    def _get_weight_indices(self, idx):
        return self.splits["weights"][idx]
    def forward(self, highres, lr_params):
        assert lr_params.shape[1] == self.num_params, "incorrect input params"
        bs, _, h, w = highres.shape
        bs, _, h_lr, w_lr = lr_params.shape
        # positional encoding
        if not(self.coordinates is None):
            if self.xy_coords is None:
                self.xy_coords = _get_coords_global(bs, h, w, highres.device, self.pe_factor)
            ##for ablation study
            highres = th.cat([highres, self.xy_coords], 1)
            #highres = self.xy_coords
        
        # Split input in tiles of size kxk according to the NN interp factor (the total downsampling factor),
        # with channels last (for matmul)
        # all pixels within a tile of kxk are processed by the same MLPs parameters
        nci = highres.shape[1]
        #nci = self.xy_coords.shape[1]
        out = rearrange(highres,'b c h w -> b h w c')
        #out = rearrange(self.xy_coords,'b c h w -> b h w c')
        num_layers = len(self.channels) - 1
        out = torch.unsqueeze(out,dim=3)
        for idx, nci in enumerate(self.channels[:-1]):
            nco = self.channels[idx + 1]
            # Select params in lowres buffer
            bstart, bstop = self._get_bias_indices(idx)
            wstart, wstop = self._get_weight_indices(idx)
            w_ = lr_params[:, wstart:wstop]
            b_ = lr_params[:, bstart:bstop]
            #out = self.net[idx](out)
            w_ = w_.permute(0, 2, 3, 1).view(bs, h_lr, w_lr, nci, nco)
            b_ = b_.permute(0, 2, 3, 1).view(bs, h_lr, w_lr, 1, nco)
            out = th.matmul(out, w_) + b_
            #out = out + b_
            # Apply RelU non-linearity in all but the last layer, and tanh in the last
            if idx < num_layers - 1:
                out = th.nn.functional.leaky_relu(out, 0.01, inplace=True)
            else:
                out = F.tanh(out)
        out = torch.squeeze(out,dim=3)
        out = rearrange(out,'b h w c-> b c h w')
        return out

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

class ASAPfunctaGeneratorV3_ds(BaseNetwork):
    def __init__(self, opt, img_size=(128,160), hr_stream=None, lr_stream=None, fast=False):
        super(ASAPfunctaGeneratorV3_ds, self).__init__()
        if lr_stream is None or hr_stream is None:
            lr_stream = dict()
            hr_stream = dict()
        self.num_inputs = opt.label_nc
        self.gpu_ids = opt.gpu_ids

        self.highres_stream = ASAPfunctaHRStreamfulres_ds(num_inputs=self.num_inputs,
                                               num_outputs=opt.output_nc, width=opt.hr_width,
                                               depth=opt.hr_depth, coordinates=opt.hr_coor,**hr_stream)

        num_params = self.highres_stream.num_params
        
        self.latlayers = nn.Conv2d(256, num_params, kernel_size=1, stride=1,padding=0)
        self.lowres_stream = Encoder_fpn4_ds(num_in=self.num_inputs, block=Bottleneck_in, num_blocks=[2,4,23,3])
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


class ASAPfunctaGeneratorV3_noint(BaseNetwork):
    def __init__(self, opt, img_size=(128,160), hr_stream=None, lr_stream=None, fast=False):
        super(ASAPfunctaGeneratorV3_noint, self).__init__()
        if lr_stream is None or hr_stream is None:
            lr_stream = dict()
            hr_stream = dict()
        self.num_inputs = opt.label_nc
        self.gpu_ids = opt.gpu_ids

        self.highres_stream = ASAPfunctaHRStreamfulres_noint(num_inputs=self.num_inputs,
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

class PPGenerator(BaseNetwork):
    'without shift modulation/without intensity'
    def __init__(self, opt, img_size=(128,160), hr_stream=None, lr_stream=None, fast=False):
        super(PPGenerator, self).__init__()
        if lr_stream is None or hr_stream is None:
            lr_stream = dict()
            hr_stream = dict()
        self.num_inputs = opt.label_nc
        self.gpu_ids = opt.gpu_ids

        self.highres_stream = decoderfullres(num_inputs=self.num_inputs,
                                               num_outputs=opt.output_nc, width=opt.hr_width,
                                               depth=opt.hr_depth, coordinates=opt.hr_coor,**hr_stream)

        num_params = self.highres_stream.num_params
        print('num of params', num_params)
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

class PPGenerator_int(BaseNetwork):
    'without shift modulation'
    def __init__(self, opt, img_size=(128,160), hr_stream=None, lr_stream=None, fast=False):
        super(PPGenerator_int, self).__init__()
        if lr_stream is None or hr_stream is None:
            lr_stream = dict()
            hr_stream = dict()
        self.num_inputs = opt.label_nc
        self.gpu_ids = opt.gpu_ids

        self.highres_stream = decoderfullres_int(num_inputs=self.num_inputs,
                                               num_outputs=opt.output_nc, width=opt.hr_width,
                                               depth=opt.hr_depth, coordinates=opt.hr_coor,**hr_stream)

        num_params = self.highres_stream.num_params
        print('num of params', num_params)
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
