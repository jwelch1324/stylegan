import torch
import numpy as np
import imageio
from einops import rearrange
import glob
import os
from tqdm.auto import tqdm

class ImagePatchDatabase(torch.utils.data.Dataset):
    def __init__(self,
        source_directory = '/data/ag_imgs/org',
        patch_width = 200,
        patch_height = 200
    ):

        self.source_directory = source_directory
        self.patch_width = patch_width
        self.patch_height = patch_height

        self.imagepatchdatasets = []

        self.indices = []

        self.load_images()

    def load_images(self):
        imgs = glob.glob(os.path.join(self.source_directory,"*.jpg"))
        imgs.extend(glob.glob(os.path.join(self.source_directory,"*.png")))

        for i in tqdm(imgs):
            img = imageio.imread(i)
            if (img.shape[0] < self.patch_height) or (img.shape[1] < self.patch_width):
                print("Source image is too small to divide up into patches! Skipping")
                continue
            self.imagepatchdatasets.append(ImagePatchDataset(img,self.patch_width,self.patch_height))

        for i in range(len(self.imagepatchdatasets)):
            n_patches = len(self.imagepatchdatasets[i])
            idxes = list(zip([i]*n_patches,range(n_patches)))
            self.indices.extend(idxes)

    def __getitem__(self,i):
        dataset_idx, patch_idx = self.indices[i]
        return self.imagepatchdatasets[dataset_idx][patch_idx]

    def __len__(self):
        return len(self.indices)



class ImagePatchDataset(torch.utils.data.Dataset):
    def __init__(self,
        source_image_array : np.ndarray,
        width = 200,
        height = 200,
        step_size = 100
    ):
        self.image = source_image_array
        self.width = width
        self.height = height
        self.step_size = step_size

        self.patches = rearrange(
            torch.tensor(self.image).unfold(0,self.width,self.step_size).unfold(1,self.height,self.step_size).reshape(-1,3,self.width,self.height), 
            'b c w h -> b w h c') #Swap the channel and wh dimensions
    
    def __getitem__(self,idx):
        img = self.patches[idx].float()
        img = img/255

        #Minmax to [-1,1]
        img = 2*(img-torch.min(img))/(torch.max(img)-torch.min(img))-1
        return img

    def __len__(self):
        return self.patches.shape[0]

