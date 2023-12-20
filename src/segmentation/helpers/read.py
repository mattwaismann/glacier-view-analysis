import rasterio
from rasterio.mask import mask
import os
import geopandas as gpd
from rasterio.warp import calculate_default_transform, reproject, Resampling
import pandas as pd
import numpy as np
from numpy import inf
from datetime import date

# reproject raster
def reproject_raster(infp, outfp, dst_crs='EPSG:4326'):
    with rasterio.open(infp) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'nodata': 0
        })

        with rasterio.open(outfp, 'w(', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i), 
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                    dst_nodata = 0)

def get_rasters(dir_path_rasters, file_name = None):
    rasters = {}
    image_file_names = next(os.walk(dir_path_rasters))[2]
    if file_name:
        image_file_names = [file_name]
    else:
        image_file_names = [file_name for file_name in image_file_names if file_name.endswith('.tif')]
    for tif in image_file_names:
        with rasterio.open(os.path.join(dir_path_rasters,tif)) as src:
            raster = src.read()
            raster[raster == -inf] = 0
            raster[raster < 0 ] = 0
            raster = np.rollaxis(raster,0,3)
            rasters[tif] = raster
    return rasters

def get_dem(file_path_dem):
    file_name = file_path_dem.split("/")[-1]
    dem = {}
    with rasterio.open(file_path_dem) as src:
        dem_tmp = src.read()
        dem_tmp = dem_tmp[[0],:,:]
        dem_tmp[dem_tmp == -inf] = 0
        dem_tmp[dem_tmp < 0 ] = 0
        dem_tmp = np.rollaxis(dem_tmp,0,3)
        dem[file_name] = dem_tmp
    return dem

def get_shapevalue_df():
    return gpd.read_file('polygons/joined.shp')

#### time series ####
def ts_get_images(folder_path, folders):
      # list of all the rasters in original shape
    rasters = []
    dates = []
    for folder in folders:
        ts_tifs = next(os.walk(os.path.join(folder_path,folder)))[2]
        rasters_temp = []
        dates_temp = []
        ts_tifs.sort()
        for i in range(len(ts_tifs)):
           	if '(' in ts_tifs[i] or 'US' in ts_tifs[i]:
           		continue
           	with rasterio.open(os.path.join(folder_path,folder,ts_tifs[i])) as src:
           		rasters_temp.append(np.nan_to_num(src.read()))
           		dates_temp.append(ts_tifs[i])
        rasters.append(rasters_temp)
        dates.append(dates_temp)
    clean_dates = []
    for date_it in dates:
        temp_clean_dates = []
        for sub_date in date_it:
            temp_clean_dates.append(date.fromisoformat(sub_date.split(".")[0]))
        clean_dates.append(temp_clean_dates)
    return(clean_dates,rasters)