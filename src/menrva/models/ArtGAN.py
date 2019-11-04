import os
import numpy as np

import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.utils.data.sampler import Sampler

from tqdm.auto import tqdm


class ModelLog():
    def __init__(self):
        self.logs = {}

    def __getitem__(self,logname):
        if not logname in self.logs:        
            self.logs[logname] = []
        return self.logs[logname]


def pixel_norm(x,epsilon=1e-8):
    rsqx = 1.0/torch.sqrt(torch.mean(torch.pow(x,2),dim=1,keepdims=True) + epsilon)
    return (x*rsqx)

class Dense(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        use_wscale = False,
        gain = np.sqrt(2),
        lrmul = 1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_wscale = use_wscale
        self.lrmul = lrmul

        fan_in = in_channels*out_channels
        he_std = gain/np.sqrt(fan_in)

        if self.use_wscale:
            init_std = 1.0/lrmul
            self.runtime_coeff = he_std*lrmul
        else:
            init_std = he_std/lrmul
            self.runtime_coeff = lrmul

        self.weight = nn.init.normal_(nn.Parameter(torch.zeros(self.in_channels,self.out_channels)),0,init_std)
        self.bias = nn.Parameter(torch.zeros(self.out_channels))

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0],-1)
        x = torch.matmul(x,self.weight*self.runtime_coeff)
        return x + (self.bias*self.lrmul)


class LatentDisentanglingNetwork(nn.Module):
    def __init__(self,
        latent_size         = 512,       # Dimension of the Latent Manifold
        label_size          = 0,         # Dimensionality of the labels, 0 if None
        dlatent_size        = 512,       # Dimensionality of the target disentangled Manifold
        dlatent_broadcast   = None,      # Output the W latent vector in shape [B, dlatent_size] or [B, dlatent_broadcast, dlatent_size]
        mapping_layers      = 8,         # Number of Mapping Layers
        mapping_hdim        = 512,       # Dimensionality of the Hidden Layers in the Mapping Network
        mapping_lrmul       = 0.01,      # Learning Rate Multiplier for the mapping layers
        mapping_actfunc     = 'lrelu',   # Activation Function
        normalize_latents   = True,      # Normalize the Latent vectors before running through the network?
        dtype               = 'float32', # Data type to use for all activations and inputs  
        **_kwargs                        # Ignore all unknown keywords   
    ):
        super().__init__()
        self.act = { 'relu' : nn.ReLU(), 'lrelu' : nn.LeakyReLU(0.01) }[mapping_actfunc]

        self.latent_size = latent_size
        self.label_size = label_size
        self.dlatent_size = dlatent_size
        self.dlatent_broadcast = dlatent_broadcast
        self.mapping_layers = mapping_layers
        self.mapping_hdim = mapping_hdim
        self.mapping_lrmul = mapping_lrmul
        self.normalize_latents = normalize_latents
        self.dtype = dtype

        #If we have labels then we need to embed them an concat them onto the latent vectors
        if self.label_size > 0:
            self.label_embedding = nn.init.kaiming_normal_(nn.Parameter(torch.zeros(self.label_size,self.latent_size)))
        
        map_layers = []

        #Build out mapping layers
        curr_dim = self.latent_size
        for layer_idx in range(self.mapping_layers):
            hidden_dim = self.dlatent_size if layer_idx == self.mapping_layers -1 else self.mapping_hdim
            map_layers.append(Dense(curr_dim,hidden_dim))
            map_layers.append(self.act)
            curr_dim = hidden_dim

        self.main = nn.Sequential(*map_layers)

    def embed_and_concat_label(self, z, label):
        """ Embeds the one-hot label into the latent-space and then concats it as an additional channel to the latent vectors """
        emb = torch.matmul(label,self.label_embedding)
        return (torch.cat([z,emb],dim=1))

    def forward(self, z, labels = None):

        if labels is not None:
            z = self.embed_and_concat_label(z,labels)

        if self.normalize_latents:
            z = pixel_norm(z)

        #Transform to W space
        z = self.main(z)

        #Broadcast if requested
        if self.dlatent_broadcast is not None:
            z=z.unsqueeze(-1).repeat(1,self.dlatent_broadcast,1).reshape(z.shape[0],self.dlatent_broadcast,self.latent_size)

        return z



class StyleGANGenerator(nn.Module):
    def __init__(self,
        sample_shape = (128,128,3),     # Input Shape of the Signal
        noise_latent_dim = 512,         # Dimensionality of the latent noise space
        num_channels = 3,               # Number of signal channels
    ):
        pass

class Generator(nn.Module):
    def __init__(self, conv_dim=64, sample_shape = (128,128,3), noise_shape=(1,100)):
        super().__init__()
        self.sample_shape = sample_shape
        self.noise_shape = (1,100)
        self.nb_noise = 2
        self.conv_dim = conv_dim
        
        self.model = self.buildGAN()

        self.channel_first_swap = Rearrange('b w h c -> b c w h')
        self.channel_last_swap = Rearrange('b c w h -> b w h c')

    def cuda(self):
        self.model = self.model.cuda()
        self.fc8 = self.fc8.cuda()
        self.fc9 = self.fc9.cuda()
        return self
    
    def cpu(self):
        self.model = self.model.cpu()
        self.fc8 = self.fc8.cpu()
        self.fc9 = self.fc9.cpu()
    
        
    ##################################################################################################################
    ##################################################################################################################
    def resBlock(self, layer_list, dim_in, dim_out):
        layer_list.append(nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False))
        layer_list.append(nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))
        layer_list.append(nn.ReLU(inplace=True))
        layer_list.append(nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False))
        layer_list.append(nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))
        
        return layer_list
    
    def residual(self, t1, t2):
        return t1 + t2
    
    def buildGAN(self):
        block_list = nn.ModuleList()
        temp_layer_list = nn.ModuleList()
        
        #Initial Layers
        temp_layer_list.append(nn.Conv2d(self.sample_shape[2], self.conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        temp_layer_list.append(nn.InstanceNorm2d(self.conv_dim, affine=True, track_running_stats=True))
        temp_layer_list.append(nn.ReLU(inplace=True))
        block_list.append(temp_layer_list)
        temp_layer_list = nn.ModuleList()
        
        #Downscale Layers
        curr_dim = self.conv_dim
        temp_layer_list.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
        temp_layer_list.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True))
        temp_layer_list.append(nn.ReLU(inplace=True))
        block_list.append(temp_layer_list)
        temp_layer_list = nn.ModuleList()
        
        temp_layer_list.append(nn.Conv2d(curr_dim * 2, curr_dim * 4, kernel_size=4, stride=2, padding=1, bias=False))
        temp_layer_list.append(nn.InstanceNorm2d(curr_dim * 4, affine=True, track_running_stats=True))
        temp_layer_list.append(nn.ReLU(inplace=True))
        block_list.append(temp_layer_list)
        temp_layer_list = nn.ModuleList()
        
        #Residual Blocks
        temp_layer_list = self.resBlock(temp_layer_list, dim_in = curr_dim * 4, dim_out = curr_dim * 4)
        block_list.append(temp_layer_list)
        temp_layer_list = nn.ModuleList()
        
        temp_layer_list = self.resBlock(temp_layer_list, dim_in = curr_dim * 4, dim_out = curr_dim * 4)
        block_list.append(temp_layer_list)
        temp_layer_list = nn.ModuleList()
        
        temp_layer_list = self.resBlock(temp_layer_list, dim_in = curr_dim * 4, dim_out = curr_dim * 4)
        block_list.append(temp_layer_list)
        temp_layer_list = nn.ModuleList()
        
        temp_layer_list = self.resBlock(temp_layer_list, dim_in = curr_dim * 4, dim_out = curr_dim * 4)
        block_list.append(temp_layer_list)      
        temp_layer_list = nn.ModuleList()
        
        temp_layer_list = self.resBlock(temp_layer_list, dim_in = curr_dim * 4, dim_out = curr_dim * 4)
        block_list.append(temp_layer_list)
        temp_layer_list = nn.ModuleList()
        
        #Upscale Layers
        self.fc8 = nn.Linear(self.noise_shape[1],256*32*32)
        temp_layer_list.append(nn.ConvTranspose2d(curr_dim * 4, (curr_dim*4) // 2, kernel_size=4, stride=2, padding=1, bias=False))
        temp_layer_list.append(nn.InstanceNorm2d((self.conv_dim*4)//2, affine=True, track_running_stats=True))
        temp_layer_list.append(nn.ReLU(inplace=True))
        block_list.append(temp_layer_list)
        temp_layer_list = nn.ModuleList()
        
        self.fc9 = nn.Linear(self.noise_shape[1],128*64*64)
        temp_layer_list.append(nn.ConvTranspose2d(curr_dim * 2, (curr_dim*2) // 2, kernel_size=4, stride=2, padding=1, bias=False))
        temp_layer_list.append(nn.InstanceNorm2d((self.conv_dim*2)//2, affine=True, track_running_stats=True))
        temp_layer_list.append(nn.ReLU(inplace=True))
        block_list.append(temp_layer_list)
        temp_layer_list = nn.ModuleList()
        
        #Final Layers
        temp_layer_list.append(nn.Conv2d(curr_dim, self.sample_shape[2], kernel_size=7, stride=1, padding=3, bias=False))
        temp_layer_list.append(nn.Tanh())
        block_list.append(temp_layer_list)    
        temp_layer_list = nn.ModuleList()
        
        return block_list

    ##################################################################################################################
    ##################################################################################################################
        
    def forward(self, x, z=None):

        #If the image is passed in channel last then we need to swap it
        swap = False
        if x.shape[1:] == torch.Size(self.sample_shape):
            x = self.channel_first_swap(x)
            swap = True

        nb_filters = None
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        
        if z is None:
            z = []
            #Generate nb_noise noise vectors
            for i in range(self.nb_noise):
                z.append(torch.randn((x.shape[0],100)).to(x.device))
        else:
            assert type(z) == list, "z is not of type: list()"
                
        #dbg.set_trace()   
        #c = c.view(c.size(0), c.size(1), 1)
        #c = c.repeat(1, 1, x.size(2))
        #x = torch.cat([x, c], dim=1)

        noise_count = 0
        
        for b, block in enumerate(self.model):
            
            #concat noise into input to decoder section
            if (b == 8) or (b == 9):
                if b == 8:
                    temp_fc = self.fc8
                elif b == 9:
                    temp_fc = self.fc9
                #concat noise with x
                #temp_fc = nn.Linear(z[noise_count].shape[-1], x.shape[1] * nb_filters)
                z_rnd = temp_fc(z[noise_count].to(x.device))
                z_reshape_noise = z_rnd.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
                z_reshape_noise = F.normalize(z_reshape_noise, p=2, dim=1)
                noise_count += 1
                
                x = x + z_reshape_noise#skip x with noise
                
            for l, layer in enumerate(block):
                #do skip connections for res blocks
                if (b >= 3) and (b <= 7):
                    temp_x = x.clone()
                    x = layer(x)
                    x = self.residual(temp_x, x)
                else:
                    x = layer(x)

        if swap:
            x = self.channel_last_swap(x)
        
        return x

class Generator2(nn.Module):
    def __init__(self, conv_dim=64, sample_shape = (128,128,3), noise_shape=(1,100)):
        super().__init__()
        self.sample_shape = sample_shape
        self.noise_shape = (1,100)
        self.nb_noise = 2
        self.conv_dim = conv_dim
        
        self.model = self.buildGAN()

        self.channel_first_swap = Rearrange('b w h c -> b c w h')
        self.channel_last_swap = Rearrange('b c w h -> b w h c')

    def cuda(self):
        self.model = self.model.cuda()
        self.fc8 = self.fc8.cuda()
        self.fc9 = self.fc9.cuda()
        return self
    
    def cpu(self):
        self.model = self.model.cpu()
        self.fc8 = self.fc8.cpu()
        self.fc9 = self.fc9.cpu()
    
        
    ##################################################################################################################
    ##################################################################################################################
    def resBlock(self, layer_list, dim_in, dim_out):
        layer_list.append(nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False))
        layer_list.append(nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))
        layer_list.append(nn.ReLU(inplace=True))
        layer_list.append(nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False))
        layer_list.append(nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))
        
        return layer_list
    
    def residual(self, t1, t2):
        return t1 + t2
    
    def buildGAN(self):
        block_list = nn.ModuleList()
        temp_layer_list = nn.ModuleList()
        
        #Initial Layers
        temp_layer_list.append(nn.Conv2d(self.sample_shape[2], self.conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        temp_layer_list.append(nn.InstanceNorm2d(self.conv_dim, affine=True, track_running_stats=True))
        temp_layer_list.append(nn.ReLU(inplace=True))
        block_list.append(temp_layer_list)
        temp_layer_list = nn.ModuleList()
        
        #Downscale Layers
        curr_dim = self.conv_dim
        temp_layer_list.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
        temp_layer_list.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True))
        temp_layer_list.append(nn.ReLU(inplace=True))
        block_list.append(temp_layer_list)
        temp_layer_list = nn.ModuleList()
        
        temp_layer_list.append(nn.Conv2d(curr_dim * 2, curr_dim * 4, kernel_size=4, stride=2, padding=1, bias=False))
        temp_layer_list.append(nn.InstanceNorm2d(curr_dim * 4, affine=True, track_running_stats=True))
        temp_layer_list.append(nn.ReLU(inplace=True))
        block_list.append(temp_layer_list)
        temp_layer_list = nn.ModuleList()
        
        #Residual Blocks
        temp_layer_list = self.resBlock(temp_layer_list, dim_in = curr_dim * 4, dim_out = curr_dim * 4)
        block_list.append(temp_layer_list)
        temp_layer_list = nn.ModuleList()
        
        temp_layer_list = self.resBlock(temp_layer_list, dim_in = curr_dim * 4, dim_out = curr_dim * 4)
        block_list.append(temp_layer_list)
        temp_layer_list = nn.ModuleList()
        
        temp_layer_list = self.resBlock(temp_layer_list, dim_in = curr_dim * 4, dim_out = curr_dim * 4)
        block_list.append(temp_layer_list)
        temp_layer_list = nn.ModuleList()
        
        temp_layer_list = self.resBlock(temp_layer_list, dim_in = curr_dim * 4, dim_out = curr_dim * 4)
        block_list.append(temp_layer_list)      
        temp_layer_list = nn.ModuleList()
        
        temp_layer_list = self.resBlock(temp_layer_list, dim_in = curr_dim * 4, dim_out = curr_dim * 4)
        block_list.append(temp_layer_list)
        temp_layer_list = nn.ModuleList()
        
        #Upscale Layers
        self.fc8 = nn.Linear(self.noise_shape[1],self.sample_shape[0]*2*32*32)
        temp_layer_list.append(nn.ConvTranspose2d(curr_dim * 4, (curr_dim*4) // 2, kernel_size=4, stride=2, padding=1, bias=False))
        temp_layer_list.append(nn.InstanceNorm2d((self.conv_dim*4)//2, affine=True, track_running_stats=True))
        temp_layer_list.append(nn.ReLU(inplace=True))
        block_list.append(temp_layer_list)
        temp_layer_list = nn.ModuleList()
        
        self.fc9 = nn.Linear(self.noise_shape[1],self.sample_shape[0]*64*64)
        temp_layer_list.append(nn.ConvTranspose2d(curr_dim * 2, (curr_dim*2) // 2, kernel_size=4, stride=2, padding=1, bias=False))
        temp_layer_list.append(nn.InstanceNorm2d((self.conv_dim*2)//2, affine=True, track_running_stats=True))
        temp_layer_list.append(nn.ReLU(inplace=True))
        block_list.append(temp_layer_list)
        temp_layer_list = nn.ModuleList()
        
        #Final Layers
        temp_layer_list.append(nn.Conv2d(curr_dim, self.sample_shape[2], kernel_size=7, stride=1, padding=3, bias=False))
        temp_layer_list.append(nn.Tanh())
        block_list.append(temp_layer_list)    
        temp_layer_list = nn.ModuleList()
        
        return block_list

    ##################################################################################################################
    ##################################################################################################################
        
    def forward(self, x, z=None):

        #If the image is passed in channel last then we need to swap it
        swap = False
        if x.shape[1:] == torch.Size(self.sample_shape):
            x = self.channel_first_swap(x)
            swap = True

        nb_filters = None
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        
        if z is None:
            z = []
            #Generate nb_noise noise vectors
            for i in range(self.nb_noise):
                z.append(torch.randn((x.shape[0],100)).to(x.device))
        else:
            assert type(z) == list, "z is not of type: list()"
                
        #dbg.set_trace()   
        #c = c.view(c.size(0), c.size(1), 1)
        #c = c.repeat(1, 1, x.size(2))
        #x = torch.cat([x, c], dim=1)

        noise_count = 0
        
        for b, block in enumerate(self.model):
            
            #concat noise into input to decoder section
            if (b == 8) or (b == 9):
                if b == 8:
                    temp_fc = self.fc8
                elif b == 9:
                    temp_fc = self.fc9
                #concat noise with x
                #temp_fc = nn.Linear(z[noise_count].shape[-1], x.shape[1] * nb_filters)
                z_rnd = temp_fc(z[noise_count].to(x.device))
                z_reshape_noise = z_rnd.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
                z_reshape_noise = F.normalize(z_reshape_noise, p=2, dim=1)
                
                if noise_count > 0:
                    x = x + z_reshape_noise
                else:
                    x = z_reshape_noise #Replace with the noise

                noise_count += 1
                
            for l, layer in enumerate(block):
                #do skip connections for res blocks
                if (b >= 3) and (b <= 7):
                    temp_x = x.clone()
                    x = layer(x)
                    x = self.residual(temp_x, x)
                else:
                    x = layer(x)

        if swap:
            x = self.channel_last_swap(x)
        
        return x

class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, input_channels=3, image_size=128, conv_dim=64, repeat_num=6):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(input_channels, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.input_channels = input_channels
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.channel_first_swap = Rearrange('b w h c -> b c w h')
        #self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        #If the image is passed in channel last then we need to swap it
        if x.shape[-1] == self.input_channels:
            x = self.channel_first_swap(x)

        h = self.main(x)
        out_src = self.conv1(h)
        #out_cls = self.conv2(h)
        return out_src #, out_cls.view(out_cls.size(0), out_cls.size(1))

class PatchGANSolver2():
    def __init__(self,
        data_loader:torch.utils.data.DataLoader,
        config #Dictionary of Configuration parameters for the Model
    ):
        self.data_loader = data_loader
        self.config = config
        self.log = ModelLog()
        self.fixed_noise = None
        self.fixed_src = None

        self.lambda_gp = self.config['lambda_gp']


        #Setup the Star Generator and Discriminators
        self.G = Generator2(
            conv_dim = self.config['conv_dim'],
            sample_shape = self.config['sample_shape'],
            noise_shape = self.config['noise_shape']
        )

        self.D = Discriminator(
            input_channels= config['sample_shape'][2],
            image_size = config['sample_shape'][1],
            conv_dim = config['conv_dim'],           
        )

        self.optG = self.optD = None
        if self.config['opt_type'].upper() == "ADAM":
            self.optG = torch.optim.Adam(self.G.parameters(), self.config['lrG'], [self.config['beta1G'],self.config['beta2G']])
            self.optD = torch.optim.Adam(self.D.parameters(), self.config['lrD'], [self.config['beta1D'],self.config['beta2D']])
        elif self.config['opt_type'].upper() == "RMSPROP":
            self.optG = torch.optim.RMSprop(self.G.parameters(), self.config['lrG'],weight_decay=self.config['weight_decay_G'])
            self.optD = torch.optim.RMSprop(self.D.parameters(), self.config['lrD'],weight_decay=self.config['weight_decay_D'])
        else:
            assert False, "The specified optimizer type isn't supported yet!"

    def save_state(self, out_path):
        torch.save({
            'G_weights':self.G.state_dict(),
            'D_weights':self.D.state_dict(),
            'optG_weights':self.optG.state_dict(),
            'optD_weights':self.optD.state_dict(),
            'config':self.config,
            'log':self.log,
            'fixed_noise':[self.fixed_noise[0].cpu(),self.fixed_noise[1].cpu()],
            'fixed_src':self.fixed_src.cpu(),
        }, out_path)

    @staticmethod
    def load(state_data_path, data_loader):
        state_data = torch.load(state_data_path)
        sgs = PatchGANSolver2(data_loader,state_data['config'].copy())
        sgs.load_state(state_data_path, data_loader)
        del state_data
        return sgs

    def load_state(self, state_data_path, data_loader, keep_config=False):
        state_data = torch.load(state_data_path)
        self.G.load_state_dict(state_data['G_weights'])
        self.D.load_state_dict(state_data['D_weights'])
        self.optG.load_state_dict(state_data['optG_weights'])
        self.optD.load_state_dict(state_data['optD_weights'])
        self.data_loader = data_loader
        if keep_config:
            self.config = state_data['config']
        self.log = state_data['log']
        self.fixed_noise = state_data['fixed_noise']
        self.fixed_src = state_data['fixed_src']

    def cuda(self):
        self.G = self.G.cuda()
        self.D = self.D.cuda()
        return self

    def cpu(self):
        self.G = self.G.cpu()
        self.D = self.D.cpu()
        return self
    
    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.optG.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.optD.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.optG.zero_grad()
        self.optD.zero_grad()

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def remove_magnitude_channels(self,x):
        return torch.cat([x[:,0:3],x[:,4:7]],dim=1)

    def join_signals(self,xi,xj):
        xr = None
        if self.config['random_pair_order']:
            if random.uniform(0,1) > 0.5:
                xr = torch.cat([xi,xj],dim=1) #[B , ch*2, seq_length]
            else:
                xr = torch.cat([xj,xi],dim=1) #[B , ch*2, seq_length]
        else:
            xr = torch.cat([xi,xj],dim=1) #[B , ch*2, seq_length]

        return xr

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(y.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def train(self):

        if os.path.isdir(self.config['outdir']):
            #Determine where we have already trained up to
            import glob
            mfiles = glob.glob(self.config['outdir']+"/model*")
            if len(mfiles) > 0:
                mfile, trained_epoch, trained_step = sorted(list(map(lambda x: (x[0],int(x[1][-2]),int(x[1][-1].split('.')[0])), map(lambda x: (x,x.split('_')),mfiles))),key=lambda x: (x[1],x[2]))[::-1][0]
                
                self.load_state(mfile,self.data_loader)
                
                if trained_step == 0:
                    trained_step = -1
        else:
            os.makedirs(self.config['outdir'])
            mfile = ""
            trained_epoch = 0
            trained_step = -1

        for epoch in range(self.config['n_epochs']):
            if epoch < int(trained_epoch):
                print(f"Trained Epoch {epoch}...Skipping")
                continue
            print(f"Starting Epoch {epoch}")
            if trained_step > 0:
                print(f"Fast forwarding to step {trained_step}")
            for i, data in tqdm(enumerate(self.data_loader),total=len(self.data_loader)):
                
                if (trained_step > 0) and (i < trained_step):
                    #Skip forward in time
                    continue
                trained_step = 0 #We set this to zero so that on the next epoch it doesn't attempt to fast forward again

                xi = data
                xi = xi.cuda()
                
                # =================================================================================== #
                #                                Train the discriminator                              #
                # =================================================================================== # 
                self.reset_grad()
                #Compute the Loss with Real Signals
                out_src = self.D(xi)

                d_loss_real = -torch.mean(out_src)

                #Compute the Loss with Fake Signals
                x_fake = self.G(xi)
                out_src = self.D(x_fake)
                d_loss_fake = torch.mean(out_src)

                #Compute loss for gradient penalty
                alpha = torch.rand(xi.size(0),1,1,1).to(xi.device)
                x_hat = (alpha*xi.data + (1 - alpha)*x_fake.data).requires_grad_(True)
                out_src = self.D(x_hat)
                d_loss_gp = self.gradient_penalty(out_src, x_hat)

                #Backward and Optimize
                d_loss = d_loss_real + d_loss_fake + self.lambda_gp * d_loss_gp
                d_loss.backward()
                self.optD.step()

                #Log the losses
                self.log['D/loss_real'].append(d_loss_real.item())
                self.log['D/loss_fake'].append(d_loss_fake.item())
                self.log['D/loss_gp'].append(d_loss_gp.item())


                # =================================================================================== #
                #                                Train the generator                                  #
                # =================================================================================== # 
                self.reset_grad()
                if i % self.config['n_critic'] == 0:
                    x_fake = self.G(xi)
                    out_src = self.D(x_fake)
                    g_loss_fake = -torch.mean(out_src)

                    #Backward and Optimize
                    g_loss = g_loss_fake
                    g_loss.backward()
                    self.optG.step()

                    #Log the losses
                    self.log['G/loss_fake'].append(g_loss_fake.item())
                else:
                    self.log['G/loss_fake'].append(self.log['G/loss_fake'][-1])          

                # Output training stats
                if i % self.config['state_update_every'] == 0:
                    print('[%d/%d][%d/%d]--\n\tLoss_D: R:%.4f F:%.4f GP:%.4f\n\tLoss_G: F:%.4f' % (
                        epoch, 
                        self.config['n_epochs'], 
                        i, 
                        len(self.data_loader), 
                        self.log['D/loss_real'][-1],
                        self.log['D/loss_fake'][-1],
                        self.log['D/loss_gp'][-1], 
                        self.log['G/loss_fake'][-1]))

            if epoch % self.config['save_every'] == 0:
                with torch.no_grad():
                    s_fake = self.G(self.fixed_src.cuda(), self.fixed_noise).cpu().detach()
                    torch.save((self.fixed_src.cpu(),s_fake.cpu()),f"{self.config['outdir']}/fake_{epoch}_{i}.pkl")
                    self.save_state(f"{self.config['outdir']}/model_weights_{epoch}_{i}.pkl")
            
            if 'archive_every' in self.config:
                if epoch % self.config['archive_every'] == 0:
                    pass #Launch the archive script here...



class PatchGANSolver():
    def __init__(self,
        data_loader:torch.utils.data.DataLoader,
        config #Dictionary of Configuration parameters for the Model
    ):
        self.data_loader = data_loader
        self.config = config
        self.log = ModelLog()
        self.fixed_noise = None
        self.fixed_src = None

        self.lambda_gp = self.config['lambda_gp']


        #Setup the Star Generator and Discriminators
        self.G = Generator(
            conv_dim = self.config['conv_dim'],
            sample_shape = self.config['sample_shape'],
            noise_shape = self.config['noise_shape']
        )

        self.D = Discriminator(
            input_channels= config['sample_shape'][2],
            image_size = config['sample_shape'][1],
            conv_dim = config['conv_dim'],           
        )

        self.optG = self.optD = None
        if self.config['opt_type'].upper() == "ADAM":
            self.optG = torch.optim.Adam(self.G.parameters(), self.config['lrG'], [self.config['beta1G'],self.config['beta2G']])
            self.optD = torch.optim.Adam(self.D.parameters(), self.config['lrD'], [self.config['beta1D'],self.config['beta2D']])
        elif self.config['opt_type'].upper() == "RMSPROP":
            self.optG = torch.optim.RMSprop(self.G.parameters(), self.config['lrG'],weight_decay=self.config['weight_decay_G'])
            self.optD = torch.optim.RMSprop(self.D.parameters(), self.config['lrD'],weight_decay=self.config['weight_decay_D'])
        else:
            assert False, "The specified optimizer type isn't supported yet!"

    def save_state(self, out_path):
        torch.save({
            'G_weights':self.G.state_dict(),
            'D_weights':self.D.state_dict(),
            'optG_weights':self.optG.state_dict(),
            'optD_weights':self.optD.state_dict(),
            'config':self.config,
            'log':self.log,
            'fixed_noise':[self.fixed_noise[0].cpu(),self.fixed_noise[1].cpu()],
            'fixed_src':self.fixed_src.cpu(),
        }, out_path)

    @staticmethod
    def load(state_data_path, data_loader):
        state_data = torch.load(state_data_path)
        sgs = PatchGANSolver(data_loader,state_data['config'].copy())
        sgs.load_state(state_data_path, data_loader)
        del state_data
        return sgs

    def load_state(self, state_data_path, data_loader):
        state_data = torch.load(state_data_path)
        self.G.load_state_dict(state_data['G_weights'])
        self.D.load_state_dict(state_data['D_weights'])
        self.optG.load_state_dict(state_data['optG_weights'])
        self.optD.load_state_dict(state_data['optD_weights'])
        self.data_loader = data_loader
        self.config = state_data['config']
        self.log = state_data['log']
        self.fixed_noise = state_data['fixed_noise']
        self.fixed_src = state_data['fixed_src']

    def cuda(self):
        self.G = self.G.cuda()
        self.D = self.D.cuda()
        return self

    def cpu(self):
        self.G = self.G.cpu()
        self.D = self.D.cpu()
        return self
    
    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.optG.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.optD.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.optG.zero_grad()
        self.optD.zero_grad()

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def remove_magnitude_channels(self,x):
        return torch.cat([x[:,0:3],x[:,4:7]],dim=1)

    def join_signals(self,xi,xj):
        xr = None
        if self.config['random_pair_order']:
            if random.uniform(0,1) > 0.5:
                xr = torch.cat([xi,xj],dim=1) #[B , ch*2, seq_length]
            else:
                xr = torch.cat([xj,xi],dim=1) #[B , ch*2, seq_length]
        else:
            xr = torch.cat([xi,xj],dim=1) #[B , ch*2, seq_length]

        return xr

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(y.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def train(self):

        if os.path.isdir(self.config['outdir']):
            #Determine where we have already trained up to
            import glob
            mfiles = glob.glob(self.config['outdir']+"/model*")
            if len(mfiles) > 0:
                mfile, trained_epoch, trained_step = sorted(list(map(lambda x: (x[0],int(x[1][-2]),int(x[1][-1].split('.')[0])), map(lambda x: (x,x.split('_')),mfiles))),key=lambda x: (x[1],x[2]))[::-1][0]
                
                self.load_state(mfile,self.data_loader)
                
                if trained_step == 0:
                    trained_step = -1
        else:
            os.makedirs(self.config['outdir'])
            mfile = ""
            trained_epoch = 0
            trained_step = -1

        for epoch in range(self.config['n_epochs']):
            if epoch < int(trained_epoch):
                print(f"Trained Epoch {epoch}...Skipping")
                continue
            print(f"Starting Epoch {epoch}")
            if trained_step > 0:
                print(f"Fast forwarding to step {trained_step}")
            for i, data in tqdm(enumerate(self.data_loader),total=len(self.data_loader)):
                
                if (trained_step > 0) and (i <= trained_step):
                    #Skip forward in time
                    continue
                trained_step = 0 #We set this to zero so that on the next epoch it doesn't attempt to fast forward again

                xi = data
                xi = xi.cuda()
                
                # =================================================================================== #
                #                                Train the discriminator                              #
                # =================================================================================== # 
                self.reset_grad()
                #Compute the Loss with Real Signals
                out_src = self.D(xi)

                d_loss_real = -torch.mean(out_src)

                #Compute the Loss with Fake Signals
                x_fake = self.G(xi)
                out_src = self.D(x_fake)
                d_loss_fake = torch.mean(out_src)

                #Compute loss for gradient penalty
                alpha = torch.rand(xi.size(0),1,1,1).to(xi.device)
                x_hat = (alpha*xi.data + (1 - alpha)*x_fake.data).requires_grad_(True)
                out_src = self.D(x_hat)
                d_loss_gp = self.gradient_penalty(out_src, x_hat)

                #Backward and Optimize
                d_loss = d_loss_real + d_loss_fake + self.lambda_gp * d_loss_gp
                d_loss.backward()
                self.optD.step()

                #Log the losses
                self.log['D/loss_real'].append(d_loss_real.item())
                self.log['D/loss_fake'].append(d_loss_fake.item())
                self.log['D/loss_gp'].append(d_loss_gp.item())


                # =================================================================================== #
                #                                Train the generator                                  #
                # =================================================================================== # 
                self.reset_grad()
                if i % self.config['n_critic'] == 0:
                    x_fake = self.G(xi)
                    out_src = self.D(x_fake)
                    g_loss_fake = -torch.mean(out_src)

                    #Backward and Optimize
                    g_loss = g_loss_fake
                    g_loss.backward()
                    self.optG.step()

                    #Log the losses
                    self.log['G/loss_fake'].append(g_loss_fake.item())
                else:
                    self.log['G/loss_fake'].append(self.log['G/loss_fake'][-1])          

                # Output training stats
                if i % self.config['state_update_every'] == 0:
                    print('[%d/%d][%d/%d]--\n\tLoss_D: R:%.4f F:%.4f GP:%.4f\n\tLoss_G: F:%.4f' % (
                        epoch, 
                        self.config['n_epochs'], 
                        i, 
                        len(self.data_loader), 
                        self.log['D/loss_real'][-1],
                        self.log['D/loss_fake'][-1],
                        self.log['D/loss_gp'][-1], 
                        self.log['G/loss_fake'][-1]))

                if i % self.config['save_every'] == 0:
                    with torch.no_grad():
                        s_fake = self.G(self.fixed_src.cuda(), self.fixed_noise).cpu().detach()
                        torch.save((self.fixed_src.cpu(),s_fake.cpu()),f"{self.config['outdir']}/fake_{epoch}_{i}.pkl")
                        self.save_state(f"{self.config['outdir']}/model_weights_{epoch}_{i}.pkl")


class StarGANSolver():
    def __init__(self,
        data_loader:torch.utils.data.DataLoader,
        config #Dictionary of Configuration parameters for the Model
    ):
        self.data_loader = data_loader
        self.config = config
        self.log = ModelLog()
        self.fixed_noise = None
        self.fixed_src = None
        self.fixed_labels = None
        self.fixed_targets = None
        self.wtype_label_map = None

        self.lambda_cls = self.config['lambda_cls']
        self.lambda_gp = self.config['lambda_gp']
        self.lambda_rec = self.config['lambda_rec']


        #Setup the Star Generator and Discriminators
        self.G = StarGenerator(
            conv_dim = self.config['conv_dim'],
            c_dim = self.config['n_types'],
            sample_shape = self.config['sample_shape'],
            noise_shape = self.config['noise_shape']
        )

        self.D = StarDiscriminator(
            input_channels= 2*config['sample_shape'][0],
            seq_length = config['sample_shape'][1],
            conv_dim = config['conv_dim'],
            c_dim = config['n_types']            
        )

        self.optG = self.optD = None
        if self.config['opt_type'].upper() == "ADAM":
            self.optG = torch.optim.Adam(self.G.parameters(), self.config['lrG'], [self.config['beta1G'],self.config['beta2G']])
            self.optD = torch.optim.Adam(self.D.parameters(), self.config['lrD'], [self.config['beta1D'],self.config['beta2D']])
        elif self.config['opt_type'].upper() == "RMSPROP":
            self.optG = torch.optim.RMSprop(self.G.parameters(), self.config['lrG'],weight_decay=self.config['weight_decay_G'])
            self.optD = torch.optim.RMSprop(self.D.parameters(), self.config['lrD'],weight_decay=self.config['weight_decay_D'])
        else:
            assert False, "The specified optimizer type isn't supported yet!"

    def save_state(self, out_path):
        torch.save({
            'G_weights':self.G.state_dict(),
            'D_weights':self.D.state_dict(),
            'optG_weights':self.optG.state_dict(),
            'optD_weights':self.optD.state_dict(),
            'config':self.config,
            'log':self.log,
            'fixed_noise':self.fixed_noise,
            'fixed_src':self.fixed_src,
            'fixed_labels':self.fixed_labels,
            'fixed_targets':self.fixed_targets,
            'wtype_label_map':self.wtype_label_map
        }, out_path)

    @staticmethod
    def load(state_data_path, data_loader):
        state_data = torch.load(state_data_path)
        sgs = StarGANSolver(data_loader,state_data['config'].copy())
        sgs.load_state(state_data_path, data_loader)
        del state_data
        return sgs

    def load_state(self, state_data_path, data_loader):
        state_data = torch.load(state_data_path)
        self.G.load_state_dict(state_data['G_weights'])
        self.D.load_state_dict(state_data['D_weights'])
        self.optG.load_state_dict(state_data['optG_weights'])
        self.optD.load_state_dict(state_data['optD_weights'])
        self.data_loader = data_loader
        self.config = state_data['config']
        self.log = state_data['log']
        self.fixed_noise = state_data['fixed_noise']
        self.fixed_src = state_data['fixed_src']
        self.fixed_labels = state_data['fixed_labels']
        self.fixed_targets = state_data['fixed_targets']
        self.wtype_label_map = state_data['wtype_label_map']

    def cuda(self):
        self.G = self.G.cuda()
        self.D = self.D.cuda()
        return self

    def cpu(self):
        self.G = self.G.cpu()
        self.D = self.D.cpu()
        return self
    
    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.optG.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.optD.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.optG.zero_grad()
        self.optD.zero_grad()

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def normalize_tensor(self,x):
        amag = x[:,3]
        aa = torch.tensor(1/np.trapz(amag,dx=1/50)).unsqueeze(1)
        x[:,0] = aa*x[:,0]
        x[:,1] = aa*x[:,1]
        x[:,2] = aa*x[:,2]
        x[:,3] = torch.sqrt(torch.pow(x[:,0:3],2).sum(dim=1))

        gmag = x[:,7]
        aa = torch.tensor(1/np.trapz(gmag,dx=1/50)).unsqueeze(1)
        x[:,4] = aa*x[:,4]
        x[:,5] = aa*x[:,5]
        x[:,6] = aa*x[:,6]
        x[:,7] = torch.sqrt(torch.pow(x[:,4:7],2).sum(dim=1))
        return x

    def remove_magnitude_channels(self,x):
        return torch.cat([x[:,0:3],x[:,4:7]],dim=1)

    def join_signals(self,xi,xj):
        xr = None
        if self.config['random_pair_order']:
            if random.uniform(0,1) > 0.5:
                xr = torch.cat([xi,xj],dim=1) #[B , ch*2, seq_length]
            else:
                xr = torch.cat([xj,xi],dim=1) #[B , ch*2, seq_length]
        else:
            xr = torch.cat([xi,xj],dim=1) #[B , ch*2, seq_length]

        return xr

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(y.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def train(self):

        if os.path.isdir(self.config['outdir']):
            #Determine where we have already trained up to
            import glob
            mfiles = glob.glob(self.config['outdir']+"/model*")
            if len(mfiles) > 0:
                mfile, trained_epoch, trained_step = sorted(list(map(lambda x: (x[0],int(x[1][-2]),int(x[1][-1].split('.')[0])), map(lambda x: (x,x.split('_')),mfiles))),key=lambda x: (x[1],x[2]))[::-1][0]
                
                self.load_state(mfile,self.data_loader)
                
                if trained_step == 0:
                    trained_step = -1
        else:
            os.makedirs(self.config['outdir'])
            mfile = ""
            trained_epoch = 0
            trained_step = -1

        for epoch in range(self.config['n_epochs']):
            if epoch < int(trained_epoch):
                print(f"Trained Epoch {epoch}...Skipping")
                continue
            print(f"Starting Epoch {epoch}")
            if trained_step > 0:
                print(f"Fast forwarding to step {trained_step}")
            for i, data in tqdm(enumerate(self.data_loader),total=len(self.data_loader)):
                
                if (trained_step > 0) and (i <= trained_step):
                    #Skip forward in time
                    continue
                trained_step = 0 #We set this to zero so that on the next epoch it doesn't attempt to fast forward again

                xi = data[0]['sample']
                xj = data[1]['sample']
                li = data[0]['wtype_label']
                lj = data[1]['wtype_label']
                
                #Normalize the inputs to be in [-1,1] for each channel
                xi = self.normalize_tensor(xi)
                xj = self.normalize_tensor(xj)

                xi = self.remove_magnitude_channels(xi)
                xj = self.remove_magnitude_channels(xj)

                xi = xi.cuda()
                xj = xj.cuda()
                li = li.cuda()
                lj = lj.cuda()
                c_org = tu.make_one_hot(li,self.config['n_types'])
                c_target = tu.make_one_hot(lj,self.config['n_types'])

                # =================================================================================== #
                #                                Train the discriminator                              #
                # =================================================================================== # 
                self.reset_grad()
                xr = self.join_signals(xi,xj)
                #Compute the Loss with Real Signals
                out_src, out_cls = self.D(xr)

                d_loss_real = -torch.mean(out_src)
                d_loss_cls = F.cross_entropy(out_cls, li)

                #Compute the Loss with Fake Signals
                x_fake = self.G(xi, c_target.float())
                xf = self.join_signals(xi,x_fake)
                out_src, out_cls = self.D(xf.detach())
                d_loss_fake = torch.mean(out_src)

                #Compute loss for gradient penalty
                alpha = torch.rand(xi.size(0),1,1).to(xi.device)
                x_hat = (alpha*xi.data + (1 - alpha)*x_fake.data).requires_grad_(True)
                out_src, _ = self.D(self.join_signals(xi,x_hat))
                d_loss_gp = self.gradient_penalty(out_src, x_hat)

                #Backward and Optimize
                d_loss = d_loss_real + d_loss_fake + self.lambda_cls*d_loss_cls + self.lambda_gp * d_loss_gp
                d_loss.backward()
                self.optD.step()

                #Log the losses
                self.log['D/loss_real'].append(d_loss_real.item())
                self.log['D/loss_fake'].append(d_loss_fake.item())
                self.log['D/loss_cls'].append(d_loss_cls.item())
                self.log['D/loss_gp'].append(d_loss_gp.item())


                # =================================================================================== #
                #                                Train the generator                                  #
                # =================================================================================== # 
                self.reset_grad()
                if i % self.config['n_critic'] == 0:
                    #Original-to-target domain
                    x_fake = self.G(xi, c_target.float())
                    out_src, out_cls = self.D(self.join_signals(xi,x_fake))
                    g_loss_fake = -torch.mean(out_src)
                    g_loss_cls = F.cross_entropy(out_cls, lj)

                    #Now test the Target-to-original domain mapping
                    x_reconst = self.G(x_fake,c_org.float())
                    g_loss_rec = torch.mean(torch.abs(xi - x_reconst))

                    #Backward and Optimize
                    g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                    g_loss.backward()
                    self.optG.step()

                    #Log the losses
                    self.log['G/loss_fake'].append(g_loss_fake.item())
                    self.log['G/loss_rec'].append(g_loss_rec.item())
                    self.log['G/loss_cls'].append(g_loss_cls.item())
                else:
                    self.log['G/loss_fake'].append(self.log['G/loss_fake'][-1])
                    self.log['G/loss_rec'].append(self.log['G/loss_rec'][-1])
                    self.log['G/loss_cls'].append(self.log['G/loss_cls'][-1])          

                # Output training stats
                if i % self.config['state_update_every'] == 0:
                    print('[%d/%d][%d/%d]--\n\tLoss_D: R:%.4f F:%.4f C:%.4f GP:%.4f\n\tLoss_G: F:%.4f Rc:%.4f C:%.4f' % (
                        epoch, 
                        self.config['n_epochs'], 
                        i, 
                        len(self.data_loader), 
                        self.log['D/loss_real'][-1],
                        self.log['D/loss_fake'][-1],
                        self.log['D/loss_cls'][-1],
                        self.log['D/loss_gp'][-1], 
                        self.log['G/loss_fake'][-1],
                        self.log['G/loss_rec'][-1],
                        self.log['G/loss_cls'][-1]))

                if i % self.config['save_every'] == 0:
                    with torch.no_grad():
                        s_fake = self.G(self.fixed_src,tu.make_one_hot(self.fixed_targets,self.config['n_types']).float(),self.fixed_noise)
                        torch.save((self.fixed_src,s_fake),f"{self.config['outdir']}/fake_{epoch}_{i}.pkl")
                        self.save_state(f"{self.config['outdir']}/model_weights_{epoch}_{i}.pkl")