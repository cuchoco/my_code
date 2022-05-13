import SimpleITK as sitk
import numpy as np

# input: dicom file path
def get_array(dicom_path : str) -> np.array : 
    return sitk.GetArrayFromImage(sitk.ReadImage(dicom_path)).squeeze()


def windowing(image: np.array, window_center: int, window_width: int) -> np.array : 
    lower = window_center - window_width/2
    upper = window_center + window_width/2
    
    image = ((np.clip(image, lower, upper) - lower) / window_width) * 255
    return np.array(image, dtype = np.uint8)