import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


# Using CrossEntropy loss
class BodyDataset(Dataset): 
    def __init__(self, df, transform=None, standardization=False):
        self.df = df
        self.classes = ('normal', 'abnormal')
        self.transform = transform
        self.standardization = standardization

    @staticmethod
    def reshape_image(img):
        return np.expand_dims(img, axis=0)
    
    def load_image(self, img_path):
        img = np.array(Image.open(img_path))
        if img.ndim == 2:
            img = BodyDataset.reshape_image(img)
        else:
            img = np.transpose(img, (2,0,1))

        img = torch.from_numpy(np.array(img, dtype=np.float32) / 255.)
        
        if self.standardization:
            if self.standardization == '20':
                img = transforms.Normalize([0.2242], [0.3448])(img)
            elif self.standardization == '30':
                img = transforms.Normalize([0.2088], [0.2965])(img)
            elif self.standardization == '3channel':
                img = transforms.Normalize([0.2096, 0.2226, 0.2109], [0.2953, 0.3468, 0.3482])(img)
                
        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, index):
        img = self.load_image(self.df.iloc[index][0])
        label = torch.LongTensor([self.df.iloc[index][1]])
        return [img, label]
        
    def __len__(self):
        return len(self.df)