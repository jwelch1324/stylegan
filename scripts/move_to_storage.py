#!/usr/bin/env python
# coding: utf-8

# In[61]:


import os
import sys
import subprocess
import glob
import argparse
import shutil
from tqdm import tqdm


# In[62]:


#parser = argparse.ArgumentParser(description='Archive Old Model Checkpoints')
#parser.add_argument('source_dir',metavar='source',type=str,help='Path to the Model Checkpoint Directory')
#parser.add_argument('target_dir',metavar='target',type=str,help='Path to the directory where the files will be moved')


# In[63]:


config = {'outdir':'/home/jwelch/Code/menrva/notebooks/ArtGAN_1'}


# In[64]:


mfiles = glob.glob(config['outdir']+"/model*")


# In[65]:


#mfile, trained_epoch, trained_step = sorted(list(map(lambda x: (x[0],int(x[1][-2]),int(x[1][-1].split('.')[0])), map(lambda x: (x,x.split('_')),mfiles))),key=lambda x: (x[1],x[2]))[::-1][10]


# In[66]:


files_to_move = list(map(lambda x: (x[1],x[2]),sorted(list(map(lambda x: (x[0],int(x[1][-2]),int(x[1][-1].split('.')[0])), map(lambda x: (x,x.split('_')),mfiles))),key=lambda x: (x[1],x[2]))[::-1][2:]))


# In[67]:


files = []
for f in files_to_move:
    files.append(config['outdir']+f"/model_weights_{f[0]}_{f[1]}.pkl")
    #files.append(config['outdir']+f"/fake_{f[0]}_{f[1]}.pkl")
    


# In[68]:


print(files)


# In[59]:


if not os.path.isdir('/data/cold_storage/ArtGAN_1'):
    os.makedirs('/data/cold_storage/ArtGAN_1')


# In[60]:


for f in tqdm(files):
    shutil.move(f,os.path.join('/data/cold_storage/ArtGAN_1/',os.path.basename(f)))


# In[ ]:




