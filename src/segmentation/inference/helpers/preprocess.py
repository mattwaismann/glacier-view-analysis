import tensorflow as tf
from src.segmentation.inference.helpers.landsat_bands import landsat_bands
import numpy as np

def get_common_bands(rasters, common_bands: list):
    rasters_temp = rasters.copy()
    for file_name in rasters_temp:
        landsat_satellite = file_name.split("_")[2].lower()
        before_bands = list(landsat_bands[landsat_satellite].keys())
        bands_idx = np.isin(before_bands, common_bands)

        rasters_temp[file_name] = rasters_temp[file_name][:,:,bands_idx]
  
    return rasters_temp
     
def normalize_rasters(rasters):
    rasters_temp = rasters.copy()
    for file_name in rasters_temp:
        n_bands = rasters_temp[file_name].shape[2]
        for i in range(n_bands):
            temp_max = rasters_temp[file_name][:,:,i].max()
            temp_min = rasters_temp[file_name][:,:,i].min()
            if temp_max != temp_min: 
                rasters_temp[file_name][:,:,i] = (rasters_temp[file_name][:,:,i]-temp_min)/(temp_max-temp_min)
    return rasters_temp

def resize_rasters(rasters,dim):
    rasters_temp = rasters.copy()  
    for file_name in rasters_temp:
        rasters_temp[file_name] = tf.image.resize(rasters_temp[file_name], dim).numpy()
    return rasters_temp
 