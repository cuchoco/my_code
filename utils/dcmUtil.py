import SimpleITK as sitk
import numpy as np
import torch
import matplotlib.pyplot as plt

# input: dicom file path
def getArray(dicom_path : str) -> np.array : 
    return sitk.GetArrayFromImage(sitk.ReadImage(dicom_path)).squeeze()


def windowing(image: np.array, window_center: int, window_width: int) -> np.array : 
    lower = window_center - window_width/2
    upper = window_center + window_width/2
    
    image = ((np.clip(image, lower, upper) - lower) / window_width) * 255
    return np.array(image, dtype = np.uint8)


def tensorShow(arr:torch.tensor):
    assert arr.ndim == 3
    arr = arr.permute(1,2,0)
    
    if arr.shape[2] == 1:
        plt.imshow(arr, cmap='gray')
    else:
        plt.imshow(arr)

    plt.axis('off')
    plt.show()
