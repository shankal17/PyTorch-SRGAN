import os
import torch
import torchvision
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset

class SRDataset(Dataset):
    
    def __init__(self, data_folder, split_type, downsample_factor=2, crop_size=96):
        self.data_folder = data_folder
        all_imgs = os.listdir(data_folder)
        self.total_imgs = natsorted(all_imgs)
        self.split_type = split_type.lower()
        self.downsample_factor = downsample_factor
        self.crop_size = crop_size
        assert self.split_type in {'train', 'test'} #TODO:add more tests

    def __getitem__(self, idx):
        img_loc = os.path.join(self.data_folder, self.total_imgs[idx])
        hr_img = Image.open(img_loc).convert('RGB')
        hr_img = torchvision.transforms.RandomCrop(size=self.crop_size).__call__(hr_img)
        lr_img = hr_img.resize((int(hr_img.width/self.downsample_factor), int(hr_img.height/self.downsample_factor)), Image.BICUBIC)
        hr_img = torchvision.transforms.ToTensor().__call__(hr_img)
        lr_img = torchvision.transforms.ToTensor().__call__(lr_img)
        return hr_img, lr_img

    def __len__(self):
        return len(self.total_imgs)