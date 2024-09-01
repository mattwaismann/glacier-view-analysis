import sys
from tqdm import tqdm
import os
from helpers import read, preprocess
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.mask import mask
# import pyvista as pv
# import rioxarray as riox
from skimage.filters import gaussian
from datetime import datetime
import cv2
import imageio.v2 as imageio
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from inference.cnn import UNet, conv_block
import torchvision
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)).replace("src",""))
PROB_THRESH = 0.5



AREAS0 = "areas_no_threshold.csv"
AREAS1 = "areas_05_thresh.csv"
AREAS2 = "areas_binary_05.csv"

first_df = pd.DataFrame()
second_df = pd.DataFrame()
third_df = pd.DataFrame()

first_df['Dates'] = pd.date_range(start='1/1/1979', end='1/1/2024', freq='MS')
second_df['Dates'] = pd.date_range(start='1/1/1979', end='1/1/2024', freq='MS')
third_df['Dates'] = pd.date_range(start='1/1/1979', end='1/1/2024', freq='MS')


with open(AREAS0, "w") as f:
    first_df.to_csv(f)
with open(AREAS1, "w") as f:
    second_df.to_csv(f)
with open(AREAS2, "w") as f:
    third_df.to_csv(f)
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

torch_model = torch.load("model")
torch_model.to(device)
torch_model.eval()

# def adj_elevation(pred):
#
#     # Read the data from a DEM file
#     data = riox.open_rasterio(dem_path)
#     data = data[0]
#
#     resize = torchvision.transforms.Resize(data.shape, antialias=True)
#     pred = resize(pred)
#     pred = pred.detach().cpu().numpy()[0]
#
#     idx=np.argwhere(pred.T>0.5)
#     ids = []
#
#     for xy in idx:
#         ids.append(xy[0]*pred.shape[0]+xy[1])
#
#     # Save the raster data as an array
#     values = np.asarray(data)
#
#     # Create a mesh grid
#     x, y = np.meshgrid(data['x'], data['y'])
#
#     # Set the z values and create a StructuredGrid
#     z = np.zeros_like(x)
#     mesh = pv.StructuredGrid(x, y, z)
#
#     # Assign Elevation Values
#     mesh["Elevation"] = values.ravel(order='F')
#
#     # Warp the mesh by scalar
#     topo = mesh.warp_by_scalar(scalars="Elevation", factor=0.000015).extract_points(ids)
#     return topo.area/mesh.extract_points(ids).area


ee_data_dir = os.path.join(BASE_DIR,  "src", "earth_engine","data","ee_landing_zone","full_time_series")
landsat_dir = os.path.join(ee_data_dir, "landsat")
dem_dir = os.path.join(ee_data_dir, "dems")

glacier_ids = os.listdir(os.path.join(BASE_DIR,  "src", "earth_engine","data","ee_landing_zone","full_time_series","landsat"))

for num, glims_id in enumerate(glacier_ids[1:]):
    # try:
        glacier_dir = os.path.join(landsat_dir, glims_id)
        dem_path = os.path.join(dem_dir, f"{glims_id}_NASADEM.tif")

        common_bands = ['blue','green','red','nir','swir','thermal']
        dim = (128,128)

        images = read.get_rasters(glacier_dir)
        #read and preprocess images
        sample_image_key = list(images.keys())[0]
        original_sizes = [images[sample_image_key].shape[:-1] for key in images.keys()]
        # print(f"Before preprocessing single image shape: {images[sample_image_key].shape}")
        nimages = preprocess.get_common_bands(images,common_bands)
        nimages = preprocess.normalize_rasters(nimages)
        nimages = preprocess.resize_rasters(nimages,dim)
        # print(f"After preprocessing single image shape: {nimages[sample_image_key].shape}")

        #read and preprocess dems

        dem = read.get_dem(dem_path)

        dem_key = list(dem.keys())[0]
        print(f"Before preprocessing single dem shape: {dem[dem_key].shape}")
        dem = preprocess.resize_rasters(dem, dim)
        dem = preprocess.normalize_rasters(dem)
        print(f"After preprocessing single dem shape: {dem[dem_key].shape}")

        combined_images_and_dems = [np.concatenate((nimages[file_name], dem[dem_key]),axis = 2) for file_name in sorted(nimages.keys())]

        #match images with labels
        X = np.stack(combined_images_and_dems)
        X = np.nan_to_num(X, copy=True, nan=X.mean())


        image_file_names_ordered = sorted(nimages.keys())
        image_dates = [datetime.strptime(f.split("_")[1],'%Y-%m-%d') for f in image_file_names_ordered]

        inputs = torch.tensor(X)

        inputs = inputs.permute(0,3,1,2)
        SMOOTH_FACTOR = 0

        green = inputs[:,1,:,:]
        swir = inputs[:,4,:,:]
        nir = inputs[:,3,:,:]

        ndsi = (green - swir)/(green + swir+SMOOTH_FACTOR)
        ndsi = ndsi.unsqueeze(dim=1)
        inputs = torch.cat((ndsi, inputs), dim=1)

        ndwi = (green - nir)/(green + nir+SMOOTH_FACTOR)
        ndwi = ndwi.unsqueeze(dim=1)
        inputs = torch.cat((ndwi, inputs), dim=1)

        # REMOVE THIS LINE - only for testing
        inputs = torch.cat((ndwi, inputs), dim=1)

        X_smoothed = gaussian(inputs, sigma=[20, 0, 0, 0], mode='reflect')
        # X_smoothed = X
        inputs = torch.tensor(X_smoothed)
        prediction_dataset = TensorDataset(inputs) # create your datset
        prediction_dataloader = DataLoader(prediction_dataset, batch_size=64,shuffle=False) # create your dataloader


        predictions = []
        predictions_05 = []
        predictions_bin = []
        
        for i in tqdm(prediction_dataloader):
            with torch.no_grad():
                outputs = torch_model.forward(i[0].to(device=device,dtype=torch.float))
                outputs = torch.softmax(outputs, dim=1)
                outputs = outputs[:,1,:,:].unsqueeze(1)
            m = torch.nn.Threshold(0, 0)
            n = torch.nn.Threshold(PROB_THRESH, 0)
            o = torch.where(outputs>PROB_THRESH,1.0,0.0)
            
            predictions.append(m(outputs))
            predictions_05.append(n(outputs))
            predictions_bin.append(o)

        predictions = torch.cat(predictions, dim=0)
        predictions_05 = torch.cat(predictions_05, dim=0)
        predictions_bin = torch.cat(predictions_bin, dim=0)

        resized_predictions = []
        resized_predictions_05 = []
        resized_predictions_bin = []
        area_per_pred = []
        area_05 = []
        area_bin = []
        resized_imgs = []
        for index, size in enumerate(original_sizes):
            # resize = torchvision.transforms.Resize(size, antialias=True)
            # pred = resize(predictions[index])
            # resized_imgs.append(pred.detach().cpu().numpy()[0])
            # area = pred.sum().detach().cpu().numpy()
            # area = area*0.0009
            resize = torchvision.transforms.Resize(size, antialias=True)
            pred = resize(predictions[index]).detach().cpu().numpy()[0]
            pred1 = resize(predictions_05[index]).detach().cpu().numpy()[0]
            pred2 = resize(predictions_bin[index]).detach().cpu().numpy()[0]
            
            resized_predictions.append(pred)
            resized_imgs.append(pred)
            
            
            area = pred.sum()
            area = area*0.0009
            # elevation_factor = adj_elevation(pred=predictions[index])
            # area*=elevation_factor
            area_per_pred.append(area)
            
            area = pred1.sum()
            area = area*0.0009
            # elevation_factor = adj_elevation(pred=predictions[index])
            # area*=elevation_factor
            area_05.append(area)
            
            area = pred2.sum()
            area = area*0.0009
            # elevation_factor = adj_elevation(pred=predictions[index])
            # area*=elevation_factor
            area_bin.append(area)
            
            
            resized_predictions.append(resize(predictions[index]))

        df = pd.DataFrame(area_per_pred,index=image_dates, columns=[glims_id])
        mo = df.groupby(pd.PeriodIndex(df.index, freq="Y"))[df.columns[0]].mean()
        mo.index = mo.index.to_timestamp().date
        mo.index = mo.index + pd.DateOffset(months=6)

        mo = mo.to_frame()
        mo["Dates"] = mo.index
        first_df = pd.merge(first_df, mo,how="left",on=["Dates"])

        with open(AREAS0, "w") as f:
            first_df.to_csv(f,index_label="Dates", index=True)
            
        df = pd.DataFrame(area_05,index=image_dates, columns=[glims_id])
        mo = df.groupby(pd.PeriodIndex(df.index, freq="Y"))[df.columns[0]].mean()
        mo.index = mo.index.to_timestamp().date
        mo.index = mo.index + pd.DateOffset(months=6)

        mo = mo.to_frame()
        mo["Dates"] = mo.index
        second_df = pd.merge(second_df, mo,how="left",on=["Dates"])

        with open(AREAS1, "w") as f:
            second_df.to_csv(f,index_label="Dates", index=True)
            
        df = pd.DataFrame(area_bin,index=image_dates, columns=[glims_id])
        mo = df.groupby(pd.PeriodIndex(df.index, freq="Y"))[df.columns[0]].mean()
        mo.index = mo.index.to_timestamp().date
        mo.index = mo.index + pd.DateOffset(months=6)

        mo = mo.to_frame()
        mo["Dates"] = mo.index
        third_df = pd.merge(third_df, mo,how="left",on=["Dates"])

        with open(AREAS2, "w") as f:
            third_df.to_csv(f,index_label="Dates", index=True)
            
        df = pd.DataFrame(area_per_pred, image_dates)

        df[glims_id] = df[0]
        df.drop([0],axis=1)
        df =  df[f"{glims_id}"]
        df = df.to_frame()
        df.index.names = ['Dates']
        df.to_csv(f"glacier_areas\\{glims_id}_areas.csv")
        

        # create GIF
        predictions = predictions.detach().cpu().numpy()

        gif_creation_dir = os.path.join(BASE_DIR,"src","segmentation","tmp",f"{glims_id}")
        # gif_creation_dir =  f"E:\\glacier-view-analysis\\src\\segmentation\\tmp\\{glims_id}"
        is_exist = os.path.exists(gif_creation_dir)
        if not is_exist:
           os.makedirs(gif_creation_dir)

        gif_output_dir = os.path.join(BASE_DIR,"src","segmentation","gifs")
        is_exist = os.path.exists(gif_output_dir)
        if not is_exist:
           os.makedirs(gif_output_dir)


        # gif_creation_dir = os.path.join(os.path.expanduser("~"), "PycharmProjects","glacier-view-analysis", "src", "segmentation","tmp","gif_creation")
        # gif_output_dir = os.path.join(os.path.expanduser("~"), "PycharmProjects","glacier-view-analysis", "src", "segmentation", "gifs")

        for f in os.listdir(gif_creation_dir):
            os.remove(os.path.join(gif_creation_dir, f))

        for i in range(X_smoothed.shape[0]):
            if i%10 == 0: #ignores 80% of images to run faster
                fig, axs = plt.subplots(2, figsize=(10,10)) ##update the number of suplots to equal the number of layers you want to display
                fig.suptitle(image_file_names_ordered[i])
                axs[0].imshow((X_smoothed[i,:,:,:][:,:,[2,1,0]]))
                # axs[0].imshow(predictions[i,0,:,:],alpha=0.1)
                axs[1].imshow((X_smoothed[i,:,:,:][:,:,[2,1,0]]))
                axs[1].imshow(predictions[i,0,:,:],alpha=0.2)

                plt.savefig(os.path.join(gif_creation_dir,f'{image_file_names_ordered[i]}_final.png'), dpi = 100)
                # plt.show()

        with imageio.get_writer(os.path.join(gif_output_dir,f"{glims_id}_final.gif"), mode='I') as writer:
            for filename in sorted(os.listdir(gif_creation_dir)):
                image = imageio.imread(os.path.join(gif_creation_dir,filename))
                writer.append_data(image)

        print(num)
    # except:
    #     print(f"Error {glims_id}")

