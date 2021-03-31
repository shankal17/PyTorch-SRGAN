import os
import torch
import shutil
from PIL import Image
from tqdm import tqdm
from dataset import PreProcessedSRDataset

filter_size = 96
img_dir = 'path/to/raw/data'
processed_dir = 'path/to/processed/images'
img_paths = [os.path.join(img_dir, filename) for filename in os.listdir(img_dir)]

for i, path in enumerate(tqdm(img_paths)):
    img = Image.open(path)
    width, height = img.size
    if width > filter_size and height > filter_size:
        shutil.copy(path, processed_dir)
