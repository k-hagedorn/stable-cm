import os 
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2, InterpolationMode 
from torchvision.io import read_image


class ADTDataset(Dataset):
    def __init__(self, path, size):
        super().__init__()
        #torch.set_default_dtype(torch.bfloat16)
        self.path = path
        self.hr_size = size
        self.upscale = v2.Compose([
            v2.Resize((size,size),interpolation=InterpolationMode.BICUBIC),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
        self.rescale =v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.hr_transform = v2.Compose([
            
            v2.Resize((size,size), interpolation=InterpolationMode.BICUBIC),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
    
    def __len__(self):
        return len(os.listdir(self.path))
    
    def __getitem__(self, index):
        index = index % len(os.listdir(self.path))
        image_pair_path = os.listdir(self.path)[index]
        hr_pair = []
        #lr_pair = []
        for i in os.listdir(os.path.join(self.path, image_pair_path)):
            img = read_image(os.path.join(self.path, image_pair_path,i))
            img = self.hr_transform(img)
            #img1 = self.degradation.feed_data(img)
            #img1 = self.upscale(img1)
            #img1 = (img1 * 2 - 1)
            #img = (img * 2 - 1)
            hr_pair.append(img.to(dtype=torch.float))
            #lr_pair.append(img1.to(dtype=torch.float))

        return hr_pair
           