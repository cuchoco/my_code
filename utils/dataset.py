import os
import SimpleITK as sitk
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset


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


class BrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.brain_classes = ('normal', 'abnormal')
        self.transform = transform
        
    def reshape_image(self, img):
        img = np.squeeze(img)
        img = np.expand_dims(img, axis=2)
        return img

    def windowing(self, input_img, mode):
        if mode == 'hemorrhage':
            windowing_level = 40
            windowing_width = 160

        elif mode == 'fracture': 
            windowing_level = 600
            windowing_width = 2000

        elif mode == 'normal':
            windowing_level = 40
            windowing_width = 80

        density_low = windowing_level - windowing_width // 2    #intensity = density
        density_high = density_low + windowing_width

        output_img = (input_img-density_low) / (density_high-density_low)
        output_img[output_img < 0.] = 0.           # windowing range
        output_img[output_img > 1.] = 1.

        return np.array(output_img, dtype='float32')
    
    def load_image(self, img_path):
        img = self.reshape_image(sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype('float32'))    
        img = np.concatenate([self.windowing(img, 'hemorrhage'), self.windowing(img, 'fracture'), self.windowing(img, 'normal')], axis=2)
        return img
    
    def __getitem__(self, index):
        img = self.load_image(os.path.join(self.df.loc[index]['path'],
                                           self.df.loc[index]['file_name']))
        gt = self.df.loc[index]['gt']
        
        
        if self.transform:
            img = self.transform(image=img)['image']
        
        sample = {'img': img,
                  'label': gt}
        return sample
    
    
    def __len__(self):
        return len(self.df)


def brain_collater(data):
    
    imgs = [s['img'] for s in data]
    labels = [s['label'] for s in data]
    
    imgs = torch.from_numpy(np.stack(imgs, axis=0)).to(torch.float32)
    labels= torch.from_numpy(np.stack(labels,axis=0)).to(torch.int64)
    
    imgs = imgs.permute(0, 3, 1, 2)

    return imgs, labels



