import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules import ReLU
from models.networks.base_network import BaseNetwork
from models.networks.generator_cat import Encoder_fpn_att, Bottleneck
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ASAPNetsBlock as ASAPNetsBlock
import torch as th
from math import pi
from math import log2
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

# classes
class AttGenerator_fpn(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='instanceaffine')
        parser.set_defaults(no_instance_dist=True)
        return parser

    def __init__(self, opt, img_size=(128,160), hr_stream=None, lr_stream=None):
        super(AttGenerator_fpn, self).__init__()
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
        
        self.encoder = Encoder_fpn_att(num_in=self.num_inputs, block=Bottleneck, num_blocks=[2,2,2,2])
        #self.decoder = CatDecoderV2(self.downsampling, num_inputs=self.num_inputs, latent_dim=self.num_params,
        #                                       num_outputs=opt.output_nc, width=opt.hr_width,
        #                                       depth=opt.hr_depth, coordinates=opt.hr_coor)
        #self.decoder = CatDecoder_fpn(num_inputs=self.num_inputs, latent_dim=self.num_params,
        #                                       num_outputs=opt.output_nc, width=opt.hr_width,
        #                                       depth=opt.hr_depth, coordinates=opt.hr_coor)
        self.latlayers = nn.Conv2d(256, 768, kernel_size=7, stride=3,padding=0)
        self.transdecoder = TransDecoder(self.img_size, self.downsampling, dim_head=64)
        
        self.encoder.apply(init_weights)
        self.transdecoder.apply(init_weights)
    

    def use_gpu(self):
        return len(self.gpu_ids) > 0

    def forward(self, input):
        features = self.encoder(input)
        features = self.latlayers(features)
        features = rearrange(features,'b c h w -> b (h w) c')
        output = self.transdecoder(input, features)
        output = rearrange(output, 'b (h w) c -> b c h w',h=self.img_size[0],w=self.img_size[1])
        return output, features#, lowres

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class FeedForward_mul(nn.Module):
    def __init__(self, patches, dim, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.fc1_w = nn.Parameter(torch.randn(patches,dim,hidden_dim))
        self.fc1_b = nn.Parameter(torch.randn(patches,1,hidden_dim))
        self.fc2_w = nn.Parameter(torch.randn(patches,hidden_dim,dim))
        self.fc2_b = nn.Parameter(torch.randn(patches,1,dim))
        
        nn.init.normal_(self.fc1_w, mean=0.0, std=0.02)
        nn.init.constant_(self.fc1_b, 0)
        nn.init.normal_(self.fc2_w, mean=0.0, std=0.02)
        nn.init.constant_(self.fc2_b, 0)
    
    def forward(self, x):
        x = self.norm(x)
        x = torch.matmul(x, self.fc1_w) + self.fc1_b
        x = self.act(x)
        x = torch.matmul(x, self.fc2_w) + self.fc2_b
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, self_att = True, kv_dim=None):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.self_att = self_att
        if not self.self_att and kv_dim is not None:
            self.to_q = nn.Linear(dim, inner_dim, bias=False)
            self.to_kv = nn.Linear(kv_dim, inner_dim * 2, bias=False)
        else:
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)
        
    def forward(self, x, z=None):
        x = self.norm(x)
        if not self.self_att and z is not None:
            q = self.to_q(x)
            k,v = self.to_kv(z).chunk(2, dim=-1)
            qkv = (q, k, v)
        else:
            qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, self_att=True, kv_dim=None):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head,self_att=self_att,kv_dim=kv_dim),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x, z=None):
        for attn, ff in self.layers:
            x = attn(x, z=z) + x
            x = ff(x) + x
        return x

class Transformer_img(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, self_att=True, kv_dim=None):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head,self_att=self_att,kv_dim=kv_dim),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self,  x, z=None):
        for attn, ff in self.layers:
            x = attn(x, z=z) + x
            x = ff(x) + x
        return x

class Transformer_multiMLP(nn.Module):
    def __init__(self, patches, dim, depth, heads, dim_head, mlp_dim, self_att=True, kv_dim=None):
        super().__init__()
        self.patches = patches
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head,self_att=self_att,kv_dim=kv_dim),
                FeedForward(dim * patches, mlp_dim)
            ]))

    def forward(self, img, x, z=None):
        '''
        img:img intensity
        x: positional encoding
        z:latent code
        '''
        pixels = x.shape[1]
        img = rearrange(img, 'b c h w -> b (h w) c')
        for attn, ff in self.layers:
            x = attn(x, z=z) + x
            #print(x.shape)
            x = rearrange(x,'b (n p) c -> b p (n c)', p=int(pixels / self.patches), n = self.patches)
            x = ff(x) + x
            x = rearrange(x, 'b p (n c) -> b (n p) c', p=int(pixels / self.patches), n = self.patches)
        return x

class Transformer_multiMLP_p(nn.Module):
    def __init__(self, patches, cell_size, dim, depth, heads, dim_head, mlp_dim, self_att=True, kv_dim=None):
        super().__init__()
        self.patches = patches
        self.cell_size = cell_size
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head,self_att=self_att,kv_dim=kv_dim),
                FeedForward_mul(self.patches, dim, mlp_dim)
            ]))

    def forward(self, img, x, z=None):
        '''
        img:img intensity
        x: positional encoding
        z: latent code
        '''
        pixels= x.shape[1]
        b, c, h, w = img.shape
        k_h = int(h / self.cell_size)
        k_w = int(w / self.cell_size)
        img = rearrange(img, 'b c h w -> b (h w) c')
        for attn, ff in self.layers:
            x = attn(x, z=z) + x
            x = rearrange(x,'b (h w) c -> b c h w',h=h,w=w)
            x = x.unfold(2, self.cell_size, self.cell_size).unfold(3, self.cell_size, self.cell_size)
            x = rearrange(x,'b c h w p1 p2 -> b (h w) (p1 p2) c')
            x = ff(x)+x
            x = rearrange(x, 'b (h w) (p1 p2) c -> b (h p1 w p2) c', h=k_h,w=k_w,p1=self.cell_size,p2=self.cell_size)
        return x

class Transformer_multiMLP_unreshaped(nn.Module):
    def __init__(self, patches, cell_size, dim, depth, heads, dim_head, mlp_dim, self_att=True, kv_dim=None):
        super().__init__()
        self.patches = patches
        self.cell_size = cell_size
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head,self_att=self_att,kv_dim=kv_dim),
                FeedForward_mul(self.patches, dim, mlp_dim)
            ]))

    def forward(self, img, x, z=None):
        '''
        img:img intensity
        x: positional encoding
        z: latent code
        '''
        pixels= x.shape[1]
        b, c, h, w = img.shape
        k_h = int(h / self.cell_size)
        k_w = int(w / self.cell_size)
        img = rearrange(img, 'b c h w -> b (h w) c')
        for attn, ff in self.layers:
            x = attn(x, z=z) + x
            x = rearrange(x,'b (h w) c -> b c h w',h=h,w=w)
            x = x.unfold(2, self.cell_size, self.cell_size).unfold(3, self.cell_size, self.cell_size)
            x = rearrange(x,'b c h w p1 p2 -> b (h w) (p1 p2) c')
            out = ff(x)+x
            x = rearrange(out, 'b (h w) (p1 p2) c -> b (h p1 w p2) c', h=k_h,w=k_w,p1=self.cell_size,p2=self.cell_size)
        return out

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

class CatEncoder(th.nn.Module):
    """ Same architecture as the image discriminator """

    def __init__(self, num_in, num_out, crop_size, norm_layer, ndf=64, kernel_size=3):
        super(CatEncoder, self).__init__()

        self.layer1 = norm_layer(nn.Conv2d(num_in, ndf, kernel_size, stride=2, padding=1))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kernel_size, stride=2, padding=1))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kernel_size, stride=2, padding=1))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kernel_size, stride=2, padding=1))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kernel_size, stride=2, padding=1))
        #if opt.crop_size >= 256:
        #    self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kernel_size, stride=2, padding=pw))
        s0 = int(crop_size[0] / pow(2,5))
        s1 = int(crop_size[1] / pow(2,5))
        self.fc = nn.Linear(ndf*8*s0*s1, num_out)
        self.actvn = nn.LeakyReLU(0.2, False)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        x = self.actvn(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output

class TransGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='instanceaffine')
        parser.set_defaults(no_instance_dist=True)
        return parser

    def __init__(self, opt, img_size=(128,160), hr_stream=None, lr_stream=None):
        super(TransGenerator, self).__init__()
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
        self.downsampling = opt.ds_factor
        norm_layer = get_nonspade_norm_layer(opt,'batch')
        #norm_layer = nn.BatchNorm2d
        self.encoder = CatEncoderV2(self.downsampling, self.num_inputs, self.num_params, self.img_size, norm_layer)
        
        self.decoder = CatDecoderV3(self.downsampling, img_size, num_inputs=self.num_inputs, latent_dim=self.num_params,
                                               num_outputs=opt.output_nc, width=opt.hr_width,
                                               depth=opt.hr_depth, coordinates=opt.hr_coor)
        self.transformer = Transformer(dim=512,depth=5,heads=16,dim_head=64,mlp_dim=2048)
        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)
        self.transformer.apply(init_weights)
    

    def use_gpu(self):
        return len(self.gpu_ids) > 0

    def forward(self, input):
        features = self.encoder(input)
        #print('before', features.shape)
        _, c, h, w = features.shape
        features = features.permute(0,2,3,1)
        pe = posemb_sincos_2d(features)
        #print(pe.shape)
        features = rearrange(features, 'b ... d -> b (...) d') + pe
        #print(features.shape)
        features = self.transformer(features)
        features = rearrange(features,'b (p1 p2) c -> b c p1 p2', p1=h, p2=w)
        #print('after', features.shape)
        output = self.decoder(input, features)
        return output, features#, lowres

class TransGeneratorV2(BaseNetwork):
    "using attention mechanism as decoder with multiple MLP"
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='instanceaffine')
        parser.set_defaults(no_instance_dist=True)
        return parser

    def __init__(self, opt, img_size=(128,160), hr_stream=None, lr_stream=None):
        super(TransGeneratorV2, self).__init__()
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
        self.downsampling = opt.ds_factor
        norm_layer = get_nonspade_norm_layer(opt,'batch')
        self.encoder = CatEncoderV2(self.downsampling, self.num_inputs, self.num_params, self.img_size, norm_layer)
        self.transdecoder = TransDecoderV2(self.img_size, self.downsampling, pe_factor=6, dim_head=64)
        self.encoder.apply(init_weights)
        self.transdecoder.apply(init_weights)
    

    def use_gpu(self):
        return len(self.gpu_ids) > 0

    def forward(self, input):
        features = self.encoder(input)
        features = rearrange(features,'b c h w -> b (h w) c')
        output = self.transdecoder(input, features)
        output = rearrange(output, 'b (h w) c -> b c h w',h=self.img_size[0],w=self.img_size[1])
        return output, features#, lowres

class TransGeneratorV3(BaseNetwork):
    "using attention mechanism as decoder with multiple MLP and last MLP"
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='instanceaffine')
        parser.set_defaults(no_instance_dist=True)
        return parser

    def __init__(self, opt, img_size=(128,160), hr_stream=None, lr_stream=None):
        super(TransGeneratorV3, self).__init__()
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
        self.downsampling = opt.ds_factor
        norm_layer = get_nonspade_norm_layer(opt,'batch')
        self.encoder = CatEncoderV2(self.downsampling, self.num_inputs, self.num_params, self.img_size, norm_layer)
        #self.transdecoder = TransDecoderV3(self.img_size, self.downsampling, pe_factor=6, dim_head=64, heads=12)
        self.transdecoder = TransDecoderV3(self.img_size, self.downsampling, pe_factor=6, dim_head=64, heads=12)

        # initialization
        self.encoder.apply(init_weights)
        self.transdecoder.apply(init_weights)
    

    def use_gpu(self):
        return len(self.gpu_ids) > 0

    def forward(self, input):
        features = self.encoder(input)
        features = rearrange(features,'b c h w -> b (h w) c')
        output = self.transdecoder(input, features)
        output = rearrange(output, 'b (h w) c -> b c h w',h=self.img_size[0],w=self.img_size[1])
        #print('output',output.shape)
        return output, features#, lowres

class TransGeneratorV4(BaseNetwork):
    "using attention mechanism as decoder with multiple MLP and last MLP"
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='instanceaffine')
        parser.set_defaults(no_instance_dist=True)
        return parser

    def __init__(self, opt, img_size=(128,160), hr_stream=None, lr_stream=None):
        super(TransGeneratorV4, self).__init__()
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
        self.downsampling = opt.ds_factor
        norm_layer = get_nonspade_norm_layer(opt,'batch')
        self.encoder = CatEncoderV2(self.downsampling, self.num_inputs, self.num_params, self.img_size, norm_layer)
        #self.transdecoder = TransDecoderV3(self.img_size, self.downsampling, pe_factor=6, dim_head=64, heads=12)
        self.transdecoder = TransDecoderV4(self.img_size, self.downsampling, pe_factor=6, dim_head=64, heads=12)

        # initialization
        self.encoder.apply(init_weights)
        self.transdecoder.apply(init_weights)
    

    def use_gpu(self):
        return len(self.gpu_ids) > 0

    def forward(self, input):
        features = self.encoder(input)
        features = rearrange(features,'b c h w -> b (h w) c')
        output = self.transdecoder(input, features)
        output = rearrange(output, 'b (h w) c -> b c h w',h=self.img_size[0],w=self.img_size[1])
        #print('output',output.shape)
        return output, features#, lowres

class TransGenerator_onemlp(BaseNetwork):
    "using attention mechanism as decoder"
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='instanceaffine')
        parser.set_defaults(no_instance_dist=True)
        return parser

    def __init__(self, opt, img_size=(128,160), hr_stream=None, lr_stream=None):
        super(TransGenerator_onemlp, self).__init__()
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
        self.downsampling = opt.ds_factor
        norm_layer = get_nonspade_norm_layer(opt,'batch')
        #norm_layer = nn.BatchNorm2d
        self.encoder = CatEncoderV2(self.downsampling, self.num_inputs, self.num_params, self.img_size, norm_layer)
        self.transdecoder = TransDecoder(self.img_size, self.downsampling, dim_head=64)
        self.encoder.apply(init_weights)
        self.transdecoder.apply(init_weights)
    

    def use_gpu(self):
        return len(self.gpu_ids) > 0

    def forward(self, input):
        features = self.encoder(input)
        features = rearrange(features,'b c h w -> b (h w) c')
        output = self.transdecoder(input, features)
        output = rearrange(output, 'b (h w) c -> b c h w',h=self.img_size[0],w=self.img_size[1])
        return output, features#, lowres

class TransDecoder(nn.Module):
    def __init__(self,img_size, pe_factor, dim_head):
        super(TransDecoder, self).__init__()
        self.img_size = img_size
        #self.pe_factor = pe_factor
        self.pe_factor = 6
        self.xy_coords = None

        ##mlp for positional encoding
        self.input_mlp = nn.Sequential(nn.Linear(4*(self.pe_factor+1) + 1,180),
                                        nn.ReLU())
        self.transformer = Transformer_img(dim=180, depth=2, heads=12, dim_head=dim_head, mlp_dim=12 * dim_head * 2, \
                                    self_att=False,kv_dim=768)
        out_layer = [nn.Linear(180,128),
                        nn.ReLU(),
                        nn.Linear(128,1),
                        nn.Tanh()]
        self.out_layer = nn.Sequential(*out_layer)

    def forward(self, img, x):
        bs, _, _ = x.shape
        h, w = self.img_size
        # Spatial encoding
        if self.xy_coords == None:
            self.xy_coords = _get_coords(bs, h, w, x.device, self.pe_factor)
        queries = torch.concat([img, self.xy_coords], dim=1)
        queries = rearrange(queries, 'b c h w -> b (h w) c')
        queries = self.input_mlp(queries)
        output = self.transformer(queries, x)
        return self.out_layer(output)

class TransDecoderV2(nn.Module):
    ''' REALLY patch-specific'''
    def __init__(self,img_size, ds_factor, pe_factor, dim_head=64, heads=12):
        super(TransDecoderV2, self).__init__()
        self.img_size = img_size
        self.pe_factor = pe_factor
        self.xy_coords = None

        ##mlp for positional encoding
        self.input_mlp = nn.Sequential(nn.Linear(4*(self.pe_factor+1) + 1,180),
                                        nn.ReLU())

        self.cell_size = pow(2, ds_factor)
        self.lr_h = int(img_size[0] / self.cell_size)
        self.lr_w = int(img_size[1] / self.cell_size)
        patches = self.lr_h * self.lr_w
        self.transformer = Transformer_multiMLP(patches, dim=180, depth=2, heads=heads, dim_head=dim_head, mlp_dim=heads * dim_head * 2, \
                                    self_att=False,kv_dim=dim_head * heads)
        out_layer = [nn.Linear(180,128),
                        nn.ReLU(),
                        nn.Linear(128,1),
                        nn.Tanh()]
        self.out_layer = nn.Sequential(*out_layer)

    def forward(self, img, features):
        bs, _, h, w = img.shape
        # Spatial encoding
        if self.xy_coords == None:
            self.xy_coords = _get_coords(bs, h, w, features.device, self.pe_factor)
        queries = torch.concat([img,self.xy_coords], dim=1)
        queries = rearrange(queries, 'b c h w -> b (h w) c')
        queries = self.input_mlp(queries)
        output = self.transformer(img, queries, features)
        
        return self.out_layer(output)

class TransDecoderV3(nn.Module):
    ''' REALLY patch-specific, and outlayer as well'''
    def __init__(self,img_size, ds_factor, pe_factor, dim_head=64, heads=12):
        super(TransDecoderV3, self).__init__()
        self.img_size = img_size
        self.pe_factor = pe_factor
        self.xy_coords = None

        ##mlp for positional encoding
        self.input_mlp = nn.Sequential(nn.Linear(4*(self.pe_factor+1) + 1,180),
                                        nn.ReLU())
        self.cell_size = pow(2, ds_factor)
        self.lr_h = int(img_size[0] / self.cell_size)
        self.lr_w = int(img_size[1] / self.cell_size)
        patches = self.lr_h * self.lr_w
        self.transformer = Transformer_multiMLP_p(patches, self.cell_size, dim=180, depth=2, heads=heads, dim_head=dim_head, mlp_dim=heads * dim_head * 2, \
                                    self_att=False,kv_dim=heads * dim_head)
        out_layer = [nn.Linear(180,128),
                        nn.ReLU(),
                        nn.Linear(128,1),
                        nn.Tanh()]
        self.out_layer = nn.Sequential(*out_layer)

    def forward(self, img, features):
        bs, _, h, w = img.shape
        # Spatial encoding
        if self.xy_coords == None:
            self.xy_coords = _get_coords(bs, h, w, features.device, self.pe_factor)
        queries = torch.concat([img,self.xy_coords], dim=1)
        queries = rearrange(queries, 'b c h w -> b (h w) c')
        queries = self.input_mlp(queries)
        output = self.transformer(img, queries, features)
        output = self.out_layer(output)
        return output

class TransDecoderV4(nn.Module):
    ''' REALLY patch-specific, and outlayer and last layer as well'''
    def __init__(self,img_size, ds_factor, pe_factor, dim_head=64, heads=12):
        super(TransDecoderV4, self).__init__()
        self.img_size = img_size
        self.pe_factor = pe_factor
        self.xy_coords = None

        ##mlp for positional encoding
        self.input_mlp = nn.Sequential(nn.Linear(4*(self.pe_factor+1) + 1,180),
                                        nn.ReLU())
        self.cell_size = pow(2, ds_factor)
        self.lr_h = int(img_size[0] / self.cell_size)
        self.lr_w = int(img_size[1] / self.cell_size)
        patches = self.lr_h * self.lr_w
        self.transformer = Transformer_multiMLP_unreshaped(patches, self.cell_size, dim=180, depth=2, heads=heads, dim_head=dim_head, mlp_dim=heads * dim_head * 2, \
                                    self_att=False,kv_dim=heads * dim_head)
        #out_layer = [nn.Linear(180,128),
        #                nn.ReLU(),
        #                nn.Linear(128,1),
        #                nn.Tanh()]
        self.fc1_w = nn.Parameter(torch.randn(patches, 180, 128))
        self.fc1_b = nn.Parameter(torch.randn(patches, 1, 128))
        self.fc2_w = nn.Parameter(torch.randn(patches, 128, 1))
        self.fc2_b = nn.Parameter(torch.randn(patches, 1, 1))
        self.act = nn.ReLU()
        self.outlayer = nn.Tanh()
        
        #self.out_layer = nn.Sequential(*out_layer)
        nn.init.normal_(self.fc1_w, mean=0.0, std=0.02)
        nn.init.constant_(self.fc1_b, 0)
        nn.init.normal_(self.fc2_w, mean=0.0, std=0.02)
        nn.init.constant_(self.fc2_b, 0)
    
    def forward(self, img, features):
        bs, _, h, w = img.shape
        # Spatial encoding
        if self.xy_coords == None:
            self.xy_coords = _get_coords(bs, h, w, features.device, self.pe_factor)
        queries = torch.concat([img,self.xy_coords], dim=1)
        queries = rearrange(queries, 'b c h w -> b (h w) c')
        queries = self.input_mlp(queries)
        output = self.transformer(img, queries, features)
        output = torch.matmul(output, self.fc1_w) + self.fc1_b
        output = self.act(output)
        output = torch.matmul(output, self.fc2_w) + self.fc2_b
        output = self.outlayer(output)
        output = rearrange(output, 'b (h w) (p1 p2) c -> b (h p1 w p2) c', h=self.lr_h,w=self.lr_w,p1=self.cell_size,p2=self.cell_size)
        #output = self.out_layer(output)
        return output

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
        #self.layer1 = norm_layer(nn.Conv2d(num_in, ndf, kernel_size, stride=2, padding=1))
        #self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kernel_size, stride=2, padding=1))
        #self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kernel_size, stride=2, padding=1))
        #self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kernel_size, stride=2, padding=1))
        #self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kernel_size, stride=2, padding=1))
        #self.fc = nn.Linear(ndf*8*s0*s1, num_out)
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
        #x = self.layer1(x)
        #x = self.layer2(self.actvn(x))
        #x = self.layer3(self.actvn(x))
        #x = self.layer4(self.actvn(x))
        #x = self.layer5(self.actvn(x))
        #x = self.actvn(x)
        x = self.dslayers(x)
        x = self.convlayers(self.actvn(x))
        #x = x.view(x.size(0), -1)
        output = self.final(x)
        return output

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
