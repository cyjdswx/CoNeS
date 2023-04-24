import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from collections import OrderedDict
import torch as th

def _get_coords(bs, h, w, device):
    x  = th.arange(0, w).float()
    y = th.arange(0, h).float()
    xn = x.float() * 2 / w -1
    yn = y.float() * 2 / h -1
    xn = xn.view(1, 1, 1, w).repeat(bs, 1, h, 1)
    yn = yn.view(1, 1, h, 1).repeat(bs, 1, 1, w)
    coords_cur = th.cat([xn, yn], 1).to(device)
    coords = coords_cur    # pyright: ignore
    return coords.to(device)

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def get_mgrid(bs, height, width):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    h = torch.linspace(-1, 1, steps=height)
    w = torch.linspace(-1, 1, steps=width)
    tensors = tuple([h, w])
    #tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    #mgrid = mgrid.reshape(-1, dim)
    #mgrid = mgrid.reshape(-1, 2)
    mgrid = mgrid.view(1,height,width,2).repeat(bs,1,1,1)
    mgrid = mgrid.permute(0,3,1,2)
    return mgrid

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
    
class Siren(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, hidden_layers, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.xy_coords = None
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
            #self.net.append(torch.sin(hidden_omega_0 * final_linear))
            self.net.append(nn.Tanh())
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, input, latent_code):
        #coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        bs, _, h, w = input.shape
        #self.xy_coords = _get_coords(bs, h, w, input.device)
        if self.xy_coords is None:
            self.xy_coords = get_mgrid(bs, h, w)
            self.xy_coords = self.xy_coords.cuda()
            #self.xy_coords = _get_coords(bs, h, w, input.device)
            self.xy_coords = self.xy_coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        #print(xy_coords)
        highres = th.cat([input, self.xy_coords, latent_code], 1)
        pixels = highres.permute(0,2,3,1)
        pixels = pixels.reshape(bs*h*w,-1)
        output = self.net(pixels)
        output= output.reshape((bs,1,h,w))
        #coords = self.xy_coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        return output, self.xy_coords
    '''
    def forward_with_activations(self, coords, retain_grad=False):
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                    
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)
                
                if retain_grad:
                    x.retain_grad()
                    
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations
    '''
