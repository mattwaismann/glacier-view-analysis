from helpers.landsat_bands import landsat_bands
import numpy as np
import torchvision
import torch

def get_common_bands(rasters, common_bands: list):
    rasters_temp = rasters.copy()
    for file_name in rasters_temp:
        landsat_satellite = file_name.split("_")[2].lower()
        before_bands = list(landsat_bands[landsat_satellite].keys())
        bands_idx = np.isin(before_bands,common_bands)
        rasters_temp[file_name] = rasters_temp[file_name][:,:,bands_idx]
  
    return rasters_temp
     
def normalize_rasters(rasters):
    rasters_temp = rasters.copy()
    for file_name in rasters_temp:
        input_raster = rasters_temp[file_name]
        n_bands = input_raster.shape[2]
        for i in range(n_bands):
            first_min = input_raster[:,:,i].min()
            first_max = input_raster[:,:,i].max()
            if first_min == first_max:
                if first_min == 0:
                    continue
                else:
                    rasters_temp[file_name][:,:,i] = input_raster[:,:,i]/first_min
                    continue
            unique_intensities = np.unique(input_raster[:,:,i])
            second_min = unique_intensities[1]
            second_max = unique_intensities[-2]
            replaced_img = np.where(input_raster[:,:,i]==first_min, second_min, input_raster[:,:,i])
            replaced_img = np.where(input_raster[:,:,i]==first_max, second_max, replaced_img)
            rasters_temp[file_name] = rasters_temp[file_name].astype(np.double)
            rasters_temp[file_name][:,:,i] = (replaced_img - second_min)/(second_max - second_min)

    return rasters_temp

def resize_rasters(rasters,dim):
    rasters_temp = rasters.copy()  
    for file_name in rasters_temp:
        resizer = torchvision.transforms.Resize(dim)
        img = torch.from_numpy(rasters_temp[file_name]).permute((2, 0, 1))
        rasters_temp[file_name] = resizer(img).permute((1,2,0)).numpy()
    return rasters_temp
 