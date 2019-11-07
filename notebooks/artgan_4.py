#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import imageio
from einops import rearrange
import random


# In[2]:


import menrva.data.ImagePatchDataset as ipd
import menrva.models.ArtGAN as ag


# In[3]:


random.seed(42)
torch.random.manual_seed(42)
np.random.seed(42)


# In[4]:


res = ipd.ImagePatchDatabase(patch_width=512,patch_height=512)


# In[5]:


loader = torch.utils.data.DataLoader(res,batch_size=4,shuffle=True)


# In[6]:


config = {
    'lambda_gp':10.0,
    'conv_dim':64,
    'sample_shape': (512,512,3),
    'latent_dim': 512,
    'opt_type':"rmsprop",
    'lrG':0.0001,
    'lrD':0.0001,
    'weight_decay_G':0.0001,
    'weight_decay_D':0.0001,
    'outdir':'ArtGAN_4',
    'n_epochs':5000,
    'n_critic':5,
    'state_update_every':25,
    'save_every':5,
}


# In[7]:


solver = ag.StyleGANSolver(loader,config,dtype=torch.float32)


# In[8]:


fixed_src = []
fixed_noise = []
fixed_src_idx = np.random.choice(range(len(res)),40,replace=False)
for idx in fixed_src_idx:
    fixed_src.append(res[idx].unsqueeze(0))
fixed_src = torch.cat(fixed_src,dim=0).to(torch.float32)
fixed_noise = torch.randn((40,config['latent_dim'])).to(torch.float32)


# In[9]:


solver.fixed_noise = fixed_noise
solver.fixed_src = fixed_src

# In[10]:


solver = solver.cuda()


# In[12]:


solver.train()


# In[ ]:




