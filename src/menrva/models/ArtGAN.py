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
    org_type = x.dtype
    x = x.to(torch.float32)
    rsqx = 1.0/torch.sqrt(torch.mean(x*x,dim=1,keepdims=True) + torch.tensor(epsilon).to(x.dtype))
    x = (x*rsqx).to(org_type)
    return x

def instance_norm(x, epsilon=1e-8):
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x -= x.mean(dim=(2,3), keepdim=True)
    epsilon = torch.tensor(epsilon).to(x.dtype)
    rsqx = 1.0/torch.sqrt(torch.mean(x*x,dim=(2,3),keepdim=True)+epsilon)
    #rsqx = 1
    x = x*rsqx
    x = x.to(orig_dtype)
    return x



def blur_2d_layer(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*np.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, padding=1, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter


class Dense(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        use_wscale = False,
        gain = np.sqrt(2),
        lrmul = 1,
        dtype = torch.float32
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_wscale = use_wscale
        self.lrmul = torch.tensor(lrmul).to(dtype)

        fan_in = in_channels*out_channels
        he_std = gain/np.sqrt(fan_in)

        if self.use_wscale:
            init_std = 1.0/lrmul
            self.runtime_coeff = he_std*lrmul
        else:
            init_std = he_std/lrmul
            self.runtime_coeff = lrmul

        self.runtime_coeff = torch.tensor(self.runtime_coeff).to(dtype)

        if dtype == torch.float32:
            self.weight = nn.init.normal_(nn.Parameter(torch.zeros(self.in_channels,self.out_channels)),0,init_std).to(dtype)
        else:
            self.weight = nn.init.kaiming_normal_(nn.Parameter(torch.zeros(self.in_channels,self.out_channels))).to(dtype)
        self.bias = nn.Parameter(torch.zeros(self.out_channels)).to(dtype)

    def cuda(self):
#        print("Dense cuda()")
        def to_cuda(x):
            for child in x.children():
                to_cuda(child)
                child.cuda()
        to_cuda(self)
        self.weight = nn.Parameter(self.weight.cuda())
        self.bias = nn.Parameter(self.bias.cuda())
        return self

    def cpu(self):
        def to_cpu(x):
            for child in x.children():
                to_cpu(child)
                child.cpu()
        to_cpu(self)
        self.weight = nn.Parameter(self.weight.cpu())
        self.bias = nn.Parameter(self.weight.cpu())
        return self

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
        dtype               = torch.float32, # Data type to use for all activations and inputs  
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
            self.label_embedding = nn.init.kaiming_normal_(nn.Parameter(torch.zeros(self.label_size,self.latent_size).to(dtype)))
        else:
            self.label_embedding = None
        
        map_layers = []

        #Build out mapping layers
        curr_dim = self.latent_size
        for layer_idx in range(self.mapping_layers):
            hidden_dim = self.dlatent_size if layer_idx == self.mapping_layers -1 else self.mapping_hdim
            map_layers.append(Dense(curr_dim, hidden_dim, dtype=dtype))
            map_layers.append(self.act)
            curr_dim = hidden_dim

        self.main = nn.Sequential(*map_layers)

    def cuda(self):
       # print("Latent Disentanglment Network cuda()")
        def to_cuda(x):
            for child in x.children():
                to_cuda(child)
                child.cuda()
        to_cuda(self)
        if self.label_embedding is not None:
            self.label_embedding = nn.Parameter(self.label_embedding.cuda())
        return self

    def cpu(self):
        def to_cpu(x):
            for child in x.children():
                to_cpu(child)
                child.cpu()
        to_cpu(self)
        if self.label_embedding is not None:
            self.label_embedding = nn.Parameter(self.label_embedding.cuda())
        return self

    def embed_and_concat_label(self, z, label):
        """ Embeds the one-hot label into the latent-space and then concats it as an additional channel to the latent vectors """
        emb = torch.matmul(label,self.label_embedding)
        return (torch.cat([z,emb],dim=1))

    def forward(self, z, labels = None):

        if labels is not None:
            z = self.embed_and_concat_label(z,labels)

        if self.normalize_latents:
            pass
            #z = pixel_norm(z)

        #Transform to W space
        z = self.main(z)

        #Broadcast if requested
        if self.dlatent_broadcast is not None:
            z = z.unsqueeze(-1).repeat(1,self.dlatent_broadcast,1).reshape(z.shape[0],self.dlatent_broadcast,self.latent_size)

        return z


class LayerEpilog(nn.Module):
    def __init__(self,
        fmap_dim,
        dlatent_dim,
        lrmul, # Learning Rate Multiplier
        activation,
        use_pixel_norm,
        use_instance_norm,
        use_styles,
        dtype = torch.float32
    ):
        super().__init__()

        self.act = {'relu': nn.ReLU(), 'lrelu': nn.LeakyReLU(0.01)}[activation]
        self.bias = nn.Parameter(torch.zeros(fmap_dim)).to(dtype)
        self.noise_scaling_factors = nn.Parameter(torch.zeros(fmap_dim)).to(dtype)# These are the learned feature wise noise scaling parameters
        self.use_pixel_norm = use_pixel_norm
        self.use_instance_norm = use_instance_norm
        self.use_styles = use_styles
        self.lrmul = torch.tensor(lrmul).to(dtype)

        if self.use_styles:
            self.style_mod_layer = LayerStyleMod(dlatent_dim, fmap_dim, dtype=dtype)
        else:
            self.style_mod_layer = None

        self.instancenorm = nn.InstanceNorm2d(fmap_dim).to(dtype)

    def cuda(self):
       # print("LayerEpilog cuda()")
        def to_cuda(x):
            for child in x.children():
                to_cuda(child)
                child.cuda()
        to_cuda(self)
        self.lrmul = self.lrmul.cuda()
        self.bias = nn.Parameter(self.bias.cuda())
        self.noise_scaling_factors = nn.Parameter(self.noise_scaling_factors.cuda())
        self.instancenorm = self.instancenorm.cuda()
        return self

    def cpu(self):
        def to_cpu(x):
            for child in x.children():
                to_cpu(child)
                child.cpu()
        to_cpu(self)
        self.lrmul = self.lrmul.cpu()
        self.bias = nn.Parameter(self.bias.cpu())
        self.noise_scaling_factors = nn.Parameter(self.noise_scaling_factors.cpu())
        return self

    def forward(self, x, dlatent, use_noise, noise=None):
        if use_noise:
            if noise is None:
                noise = torch.randn((x.shape[0], 1, x.shape[2], x.shape[3])).to(x.dtype).to(x.device)#2d
            else:
                noise = noise.to(x.dtype).to(x.device)
            # Add the noise with the channel wise scaling
            x = x + noise * (self.noise_scaling_factors.reshape(1,-1,1,1).to(x.dtype).to(x.device))#2d
        
        #Apply the bias
        if len(x.shape) == 2:
            x = x + (self.bias * self.lrmul)
        else:
            x = x + (self.bias.reshape((1,-1,1,1)) * self.lrmul) #This will broadcast the bias value across the entire channel dimension #2d
        
        #Apply the nonlinearity 
        x = self.act(x)

        if self.use_pixel_norm:
            x = pixel_norm(x)
        if self.use_instance_norm:
            x = self.instancenorm(x)
        if self.use_styles:
            x = self.style_mod_layer(x, dlatent)
            
        return x

class LayerStyleMod(nn.Module):
    def __init__(self,
        dlatent_dim, #Dimensionality of the dlatent space
        layer_fmaps, #The number of feature maps for the layer this style is applied on
        dtype=torch.float32
    ):
        super().__init__()

        self.dense = Dense(dlatent_dim,2*layer_fmaps,gain=1,dtype=dtype)

    def cuda(self):
       # print("LayerStyleMod cuda()")
        super().cuda()
        self.dense = self.dense.cuda()
        return self

    def forward(self, x, dlatent):
        style = self.dense(dlatent)
        style = style.reshape([-1,2,x.shape[1]] + [1] * (len(x.shape) -2))
        # Apply the adaptive instance norm transform -- x should have been instance normed prior to this
        return x*(style[:,0] + 1) + style[:,1]


class Block(nn.Module):
    def __init__(self,
        block_resolution, #log2 resolution of this block
        fmaps_base, # Multiplier for the number of feature maps
        dlatent_dim, # The Dimensionality of the W space 
        activation,
        use_pixel_norm,
        use_instance_norm,
        use_styles,
        dtype = torch.float32
    ):
        super().__init__()
        def nf(stage): return min(int(fmaps_base/(2.0**(stage*1.0))),512)
        def calc_padding(hout,hin,dilation,kernel,stride): return int(np.floor((stride*(hout-1)-hin+dilation*(kernel-1)+1)/2))

        self.block_resolution = block_resolution
        

        self.epilogs = nn.ModuleList([
            LayerEpilog(
                nf(self.block_resolution-1),
                dlatent_dim,
                1.0,
                activation,
                use_pixel_norm,
                use_instance_norm,
                use_styles,
                dtype = dtype
            ),
            LayerEpilog(
                nf(self.block_resolution-1),
                dlatent_dim,
                1.0,
                activation,
                use_pixel_norm,
                use_instance_norm,
                use_styles,
                dtype = dtype
            )
        ])

        self.upscale_conv = nn.ConvTranspose2d(
            in_channels = nf(self.block_resolution-2),
            out_channels = nf(self.block_resolution-1),
            kernel_size=3,
            stride = 2,
            padding = 1,
            output_padding = 1,
            bias=False #The bias term is handled in the Layer Epilogs
        ).to(dtype)

        self.out_conv = nn.Conv2d(
            in_channels = nf(self.block_resolution-1),
            out_channels = nf(self.block_resolution-1),
            kernel_size=3,
            stride = 1,
            padding = calc_padding(2**self.block_resolution, 2**self.block_resolution,1,3,1),
            bias=False #The bias term is handled in the LayerEpilog
        ).to(dtype)

        #The gaussian blur that is applied after the upscale to help mitigate checkerboard effects
        self.blur_layer = blur_2d_layer(3,1,nf(self.block_resolution-1)).to(dtype)
    
    def cuda(self):
        #print("Block cuda()")
        def to_cuda(x):
            for child in x.children():
                to_cuda(child)
                child.cuda()
        to_cuda(self)
        return self

    def forward(self, x, dlatent, use_noise, noise0=None, noise1=None):
        # Upscale and blur the sample
        x = self.epilogs[0](self.blur_layer(self.upscale_conv(x)), dlatent, use_noise, noise0)
        # Apply second conv
        x = self.epilogs[1](self.out_conv(x), dlatent, use_noise, noise1)
        return x



class StyleGANSynthesizer(nn.Module):
    def __init__(self, 
        output_resolution = 256, # The final output resolution of the generated image -- best to use a power of two
        dlatent_dim = 512, # Dimensionality of the disentangled latent noise space W
        num_channels = 3,  # Number of output channels for the generated signal
        fmap_base = 8192, # Multiplier for the number of feature maps
        fmap_decay = 1.0, # log2 feature map reduction when doubling the resolution
        fmap_max = 512, # Maximum number of feature maps allowed in any layer.
        use_styles = True, # Enable the Style Inputs?
        const_input_layer = True, # Is the first layer of the generator a learned constant?
        use_noise = True, # Enable noise inputs?
        randomize_noise = True, # True = Randomize Noise inputs every time, False = read noise inputs from variables
        activation = 'lrelu', # Activation function, can be 'relu' or 'lrelu'
        use_wscale = True, # Enabled the Equalized Learning Rate?
        use_pixel_norm = False, # Enable pixelwise feature vector normalization?
        use_instance_norm = True, # Enable Instance Normalization?
        dtype = torch.float32,
        fused_scale = 'auto', # True = fused convolution + scaling, False = separate ops, 'auto' = decide automatically
        blur_filter = [1,2,1], # Low-pass filter to apply when resampling, None = no filtering
        structure = 'auto', # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, 'auto' = select automatically,
        **_kwargs
    ):
        super().__init__()
        resolution_log2 = int(np.log2(output_resolution))
        assert output_resolution == 2**resolution_log2 and output_resolution >= 4
        def nf(stage): return min(int(fmap_base/(2.0**(stage*fmap_decay))),fmap_max)
        def blur(x): return blur2d(x, blur_filter) if blur_filter else x
        
        #Calculates the output shape of the convolution operations
        def conv_out_size(s_in, padding, dilation, kernel, stride): return int(np.floor(((s_in+2*padding-dilation*(kernel-1)-1)/stride) + 1))
        def calc_padding(hout,hin,dilation,kernel,stride): return int(np.floor((stride*(hout-1)-hin+dilation*(kernel-1)+1)/2))
        if structure == 'auto': 
            self.structure = 'fixed' #Default to fixed for the moment
        else:
            self.structure = structure
        
        self.act = {'relu': nn.ReLU, 'lrelu': nn.LeakyReLU(0.01)}[activation]

        self.num_layers = resolution_log2 * 2 - 2
        num_styles = self.num_layers if use_styles else 1

        self.use_noise = use_noise
        self.use_wscale = use_wscale
        self.use_styles = use_styles
        self.use_pixel_norm = use_pixel_norm
        self.use_instance_norm = use_instance_norm
        self.fused_scale = fused_scale
        self.blur_filter = blur_filter
        self.tensor_dtype = dtype
        self.dlatent_dim = dlatent_dim

        self.contexts = resolution_log2 + 1 - 3

        self.const_tensor = nn.Parameter(torch.ones(1,nf(1),4,4)).to(dtype) #The learned starting const tensor

        # Early Layers
        self.const_layer_epilog = LayerEpilog(
                nf(1),
                dlatent_dim,
                1.0,
                activation,
                use_pixel_norm,
                use_instance_norm,
                use_styles,
                dtype=dtype
        )            
        self.conv_4x4 = nn.Conv2d(
            nf(1),
            nf(1),
            kernel_size = 3,
            stride = 1,
            padding = calc_padding(4,4,1,3,1),
            bias = False
        ).to(dtype)

        self.conv_4x4_epilog = LayerEpilog(
            nf(1),
            dlatent_dim,
            1.0,
            activation,
            use_pixel_norm,
            use_instance_norm,
            use_styles,
            dtype=dtype
        )

        blocks = []
        if self.structure == 'fixed':
            for res in range(3, resolution_log2 + 1):
                blocks.append(Block(
                    block_resolution = res, #log2 resolution of this block
                    fmaps_base = fmap_base, # Multiplier for the number of feature maps
                    dlatent_dim = dlatent_dim, # The Dimensionality of the W space 
                    activation = activation,
                    use_pixel_norm = self.use_pixel_norm,
                    use_instance_norm = self.use_instance_norm,
                    use_styles = self.use_styles,
                    dtype=dtype
                ))

            self.body = nn.ModuleList(blocks)
            #The final convolutional layer to output the expected number of channels
            self.to_final_shape = nn.Conv2d(
                nf(resolution_log2 - 1),
                num_channels,
                kernel_size=1,
                stride=1,
                padding=0
            ).to(dtype)
        else:
            raise NotImplementedError()

    def cuda(self):
        def to_cuda(x):
            for child in x.children():
                to_cuda(child)
                child.cuda()
        to_cuda(self)
        self.const_tensor = nn.Parameter(self.const_tensor.cuda())
        return self

    def cpu(self):
        def to_cpu(x):
            for child in x.children():
                to_cpu(child)
                child.cpu()
        to_cpu(self)
        self.const_tensor = nn.Parameter(self.const_tensor.cpu())
        return self

    def forward(self, w_latents, noise=None):
        #w_latents should have shape [B,num_layers,dlatent_dim]
        if noise is not None:
            assert len(noise) == self.num_layers #Make sure we have a noise vector for each expected layer
        else:
            noise = [None] * self.num_layers #Create a list of None noises to pass into the layers

        # Early Layers
        x = self.const_layer_epilog(self.const_tensor.to(self.tensor_dtype).repeat((w_latents.shape[0],1,1,1)),w_latents[:,0],self.use_noise,noise[0])
        x = self.conv_4x4_epilog(self.conv_4x4(x),w_latents[:,1],self.use_noise,noise[1])

        if self.structure == 'fixed':
            #Apply the main body of the network
            c = 2
            for block in self.body:
                #print(c)
                x = block(x,w_latents[:,c],self.use_noise,noise[c],noise[c+1])
                c += 2 #Advance the noise pointer
            x = self.to_final_shape(x)
        else:
            raise NotImplementedError()

        return x





class StyleGANGenerator(nn.Module):
    def __init__(self,
        output_resolution = 256, # The final output resolution of the generated image -- must use a power of two
        latent_space_dim = 512, # Dimension of the Latent Manifold
        label_size = 0, # Dimensionality of the labels, 0 if None
        dlatent_dim = 512, # Dimensionality of the disentangled latent noise space W
        num_channels = 3,  # Number of output channels for the generated signal
        fmap_base = 8192, # Multiplier for the number of feature maps
        fmap_decay = 1.0, # log2 feature map reduction when doubling the resolution
        fmap_max = 512, # Maximum number of feature maps allowed in any layer.
        use_styles = True, # Enable the Style Inputs?
        const_input_layer = True, # Is the first layer of the generator a learned constant?
        use_noise = True, # Enable noise inputs?
        randomize_noise = True, # True = Randomize Noise inputs every time, False = read noise inputs from variables
        activation = 'lrelu', # Activation function, can be 'relu' or 'lrelu'
        use_wscale = True, # Enabled the Equalized Learning Rate?
        use_pixel_norm = False, # Enable pixelwise feature vector normalization?
        use_instance_norm = True, # Enable Instance Normalization?
        dtype = torch.float32,
        fused_scale = 'auto', # True = fused convolution + scaling, False = separate ops, 'auto' = decide automatically
        blur_filter = [1,2,1], # Low-pass filter to apply when resampling, None = no filtering
        structure = 'auto', # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, 'auto' = select automatically,
        **_kwargs
    ):
        super().__init__()
        self.synthesizer = StyleGANSynthesizer(
            output_resolution = output_resolution,
            dlatent_dim = dlatent_dim,
            num_channels = num_channels,
            fmap_base = fmap_base,
            fmap_decay = fmap_decay,
            fmap_max = fmap_max,
            use_styles = use_styles,
            const_input_layer = const_input_layer,
            use_noise = use_noise,
            randomize_noise = randomize_noise,
            activation=activation,
            use_wscale = use_wscale,
            use_pixel_norm = use_pixel_norm,
            use_instance_norm = use_instance_norm,
            dtype = dtype,
            fused_scale = fused_scale,
            blur_filter = blur_filter,
            structure = structure
        )

        self.mapping_network = LatentDisentanglingNetwork(
            latent_size = latent_space_dim,
            label_size = label_size,
            dlatent_size = dlatent_dim,
            dlatent_broadcast = self.synthesizer.num_layers,
            mapping_layers = 8,
            mapping_hdim = 512,
            mapping_lrmul = 0.01,
            mapping_actfunc = 'lrelu',
            normalize_latents = True,
            dtype = dtype
        )

    def cuda(self):
        self.synthesizer = self.synthesizer.cuda()
        self.mapping_network = self.mapping_network.cuda()
        return self

    def cpu(self):
        self.synthesizer = self.synthesizer.cpu()
        self.mapping_network = self.mapping_network.cpu()
        return self

    def forward(self, z_latents, labels=None, noise=None):
        dlatents = self.mapping_network(z_latents,labels)
        return self.synthesizer(dlatents,noise)



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
    def __init__(self, input_channels=3, image_size=128, conv_dim=64, repeat_num=6, dtype=torch.float32):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(input_channels, conv_dim, kernel_size=4, stride=2, padding=1).to(dtype))
        layers.append(nn.LeakyReLU(0.01).to(dtype))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1).to(dtype))
            layers.append(nn.LeakyReLU(0.01).to(dtype))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.input_channels = input_channels
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False).to(dtype)
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


class StyleGANSolver():
    def __init__(self,
        data_loader:torch.utils.data.DataLoader,
        config, #Dictionary of Configuration parameters for the Model
        dtype=torch.float32
    ):
        self.data_loader = data_loader
        self.config = config
        self.log = ModelLog()
        self.fixed_noise = None
        self.fixed_src = None
        self.tensor_dtype=dtype

        self.lambda_gp = self.config['lambda_gp']


        #Setup the Generator and Discriminators
        self.G = StyleGANGenerator(
            output_resolution = 512,
            dtype=dtype
        )

        self.D = Discriminator(
            input_channels= config['sample_shape'][2],
            image_size = config['sample_shape'][1],
            conv_dim = config['conv_dim'],
            dtype=dtype         
        )

        self.channel_swap = Rearrange('b c h w -> b h w c')

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
        self.channel_swap = self.channel_swap.cuda()
        return self

    def cpu(self):
        self.G = self.G.cpu()
        self.D = self.D.cpu()
        self.channel_swap = self.channel_swap.cpu()
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
                xi = xi.to(self.tensor_dtype).cuda()
                
                # =================================================================================== #
                #                                Train the discriminator                              #
                # =================================================================================== # 
                self.reset_grad()
                #Compute the Loss with Real Signals
                out_src = self.D(xi)

                d_loss_real = -torch.mean(out_src)

                #Compute the Loss with Fake Signals
                #Sample the latent vectors for this round
                latents = torch.randn((xi.shape[0],self.config['latent_dim'])).to(self.tensor_dtype).cuda()
                x_fake = self.G(latents)
                out_src = self.D(x_fake)
                d_loss_fake = torch.mean(out_src)

                #Compute loss for gradient penalty
                alpha = torch.rand(xi.size(0),1,1,1).to(xi.device)
                x_hat = (alpha*xi.data + (1 - alpha)*self.channel_swap(x_fake).data).requires_grad_(True).to(xi.dtype).to(xi.device)
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
                    latents = torch.randn((xi.shape[0],self.config['latent_dim'])).to(self.tensor_dtype).cuda()
                    x_fake = self.G(latents)
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
                    fake_out = []
                    for fake in self.fixed_noise:
                        fake_out.append(self.G(fake.unsqueeze(0).cuda()).unsqueeze(0).detach().cpu())
                    #s_fake = self.G(self.fixed_noise.cuda()).cpu().detach()
                    s_fake = torch.cat(fake_out,dim=0).cpu()
                    torch.save((self.fixed_src.cpu(),s_fake.cpu()),f"{self.config['outdir']}/fake_{epoch}_{i}.pkl")
                    self.save_state(f"{self.config['outdir']}/model_weights_{epoch}_{i}.pkl")
            
            if 'archive_every' in self.config:
                if epoch % self.config['archive_every'] == 0:
                    pass #Launch the archive script here...

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