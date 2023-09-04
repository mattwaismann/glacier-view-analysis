from pathlib import Path

print("Haha")
print(Path(__file__).parent.parent)
"""
import sys
import os

sys.path.insert(0,
                os.path.join(os.path.expanduser("~"), "PycharmProjects", "glacier-view-analysis", "src", "segmentation",
                             "helpers"))

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import read
import preprocess
import landsat_bands
import matplotlib.pyplot as plt
import rasterio
from skimage.filters import gaussian
from datetime import datetime
import cv2
import imageio.v2 as imageio
import torch

# define inputs
# glims_id = 'G006628E45300N'
# glims_id = 'G006819E45785N' #lex blanche
glims_id = 'G007026E45991N'  # trient
# glims_id = 'G086519E27919N'
PROB_THRESH = 0.3

data_label = "full_time_series"
ee_data_dir = os.path.join(os.path.expanduser("~"),
                           "PycharmProjects", "glacier-view-analysis", "src", "earth_engine", "data", "ee_landing_zone",
                           data_label)
dem_data_dir = os.path.join(os.path.expanduser("~"),
                            "PycharmProjects", "glacier-view-analysis", "src", "earth_engine", "data",
                            "ee_landing_zone", "full_time_series")
landsat_dir = os.path.join(ee_data_dir, "landsat")
dem_dir = os.path.join(dem_data_dir, "dems")

glacier_dir = os.path.join(landsat_dir, glims_id)
dem_path = os.path.join(dem_dir, f"{glims_id}_NASADEM.tif")
common_bands = ['blue', 'green', 'red', 'nir', 'swir', 'thermal', 'swir_2']
dim = (128, 128)
# read and preprocess images

images = read.get_rasters(glacier_dir)

sample_image_key = list(images.keys())[0]
print(f"Before preprocessing single image shape: {images[sample_image_key].shape}")
images = preprocess.get_common_bands(images, common_bands)
images = preprocess.normalize_rasters(images)
images = preprocess.resize_rasters(images, dim)
print(f"After preprocessing single image shape: {images[sample_image_key].shape}")
# read and preprocess dems

dem = read.get_dem(dem_path)

dem_key = list(dem.keys())[0]
print(f"Before preprocessing single dem shape: {dem[dem_key].shape}")
dem = preprocess.resize_rasters(dem, dim)
dem = preprocess.normalize_rasters(dem)
print(f"After preprocessing single dem shape: {dem[dem_key].shape}")
dem.keys()
combined_images_and_dems = [np.concatenate((images[file_name], dem[dem_key]), axis=2) for file_name in
                            sorted(images.keys())]
# match images with labels
X = np.stack(combined_images_and_dems)
X_smoothed = gaussian(X, sigma=[20, 0, 0, 0], mode='reflect')
image_file_names_ordered = sorted(images.keys())
image_dates = [datetime.strptime(f.split("_")[1], '%Y-%m-%d') for f in image_file_names_ordered]

import torch

import torchvision.models as models
import torch.nn as nn
import torch
from torch.nn.functional import interpolate


# from torchvision.models import resnet50, ResNet50_Weights


class UNet(nn.Module):
    def __init__(self, n_class, freeze_encoder=True):
        self.n_class = n_class
        self.freeze_encoder = freeze_encoder
        super(UNet, self).__init__()

        resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')

        if self.freeze_encoder:
            for param in resnet.parameters():
                param.requires_grad = False
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.resnet[0] = nn.Conv2d(10, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.deconv2 = nn.ConvTranspose2d(2048, 512, kernel_size=3, stride=2, padding=1, dilation=1,
                                          output_padding=1)
        self.bn2 = nn.BatchNorm2d(512)

        self.deconv3 = nn.ConvTranspose2d(1024, 256, kernel_size=3, stride=2, padding=1, dilation=1,
                                          output_padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(512, 64, kernel_size=3, stride=2, padding=1, dilation=1,
                                          output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.deconv6 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.deconv7 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn7 = nn.BatchNorm2d(16)
        self.deconv8 = nn.ConvTranspose2d(16, 4, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn8 = nn.BatchNorm2d(4)
        self.c8 = nn.Conv2d(4, 4, 6, stride=8, padding=0)

        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Conv2d(4, self.n_class, kernel_size=1)
        # self.classifier = nn.Conv2d(4, 1, kernel_size=1)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, images):
        x0 = self.resnet[0](images)
        x1 = self.resnet[1](x0)
        x2 = self.resnet[2](x1)
        x3 = self.resnet[3](x2)
        x4 = self.resnet[4](x3)
        x5 = self.resnet[5](x4)
        x6 = self.resnet[6](x5)
        out = self.resnet[7](x6)

        y1 = self.bn1(self.relu(self.deconv1(out)))
        y1 = torch.cat([y1, x6], dim=1)

        y2 = self.bn2(self.relu(self.deconv2(y1)))
        y2 = torch.cat([y2, x5], dim=1)

        y3 = self.bn3(self.relu(self.deconv3(y2)))
        y3 = torch.cat([y3, x4], dim=1)

        y4 = self.bn4(self.relu(self.deconv4(y3)))
        y4 = torch.cat([y4, x2], dim=1)

        y5 = self.bn5(self.relu(self.deconv5(y4)))
        y6 = self.bn6(self.relu(self.deconv6(y5)))
        y7 = self.bn7(self.relu(self.deconv7(y6)))
        y8 = self.bn8(self.relu(self.deconv8(y7)))
        y9 = self.c8(y8)

        # score = self.softmax(self.classifier(y9))
        score = self.classifier(y9)
        # score = y8
        return score


torch_model = torch.load('model').eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
# apply the saved model
# saved_models_dir = os.path.join(os.path.expanduser("~"),"PycharmProjects","glacier-view-analysis",  "src","segmentation","saved_models")
# saved_model_path = os.path.join(saved_models_dir, "re_ni_sw_de_v1.h5")

# model = keras.models.load_model(saved_model_path, compile = False)
# predictions = model.predict(X_smoothed)
inputs = torch.tensor(X_smoothed)
from torch.utils.data import TensorDataset, DataLoader

inputs = torch.tensor(X_smoothed)

inputs = inputs.permute(0, 3, 1, 2)
SMOOTH_FACTOR = 0.0001

green = inputs[:, 1, :, :]
swir = inputs[:, 4, :, :]
nir = inputs[:, 3, :, :]

ndsi = (green - swir + SMOOTH_FACTOR) / (green + swir + SMOOTH_FACTOR)
ndsi = ndsi.unsqueeze(dim=1)
inputs = torch.cat((ndsi, inputs), dim=1)

ndwi = (green - nir + SMOOTH_FACTOR) / (green + nir + SMOOTH_FACTOR)
ndwi = ndwi.unsqueeze(dim=1)
inputs = torch.cat((ndwi, inputs), dim=1)

inputs.shape
prediction_dataset = TensorDataset(inputs)  # create your datset
prediction_dataloader = DataLoader(prediction_dataset, batch_size=64)  # create your dataloader
predictions = []
for i in prediction_dataloader:
    with torch.no_grad():
        outputs = torch_model.forward(i[0].to(device))
        outputs = torch.softmax(outputs, dim=1)
        outputs = outputs[:, 1, :, :].unsqueeze(1)
    m = torch.nn.Threshold(PROB_THRESH, 0)
    predictions.append(m(outputs))

predictions = torch.cat(predictions, dim=0)
predictions = predictions.detach().cpu().numpy()
# create GIF
gif_creation_dir = os.path.join(os.path.expanduser("~"), "PycharmProjects", "glacier-view-analysis", "src",
                                "segmentation", "tmp", "gif_creation")
gif_output_dir = os.path.join(os.path.expanduser("~"), "PycharmProjects", "glacier-view-analysis", "src",
                              "segmentation", "gifs")

for f in os.listdir(gif_creation_dir):
    os.remove(os.path.join(gif_creation_dir, f))

for i in range(X_smoothed.shape[0]):
    if i % 5 == 0:  # ignores 80% of images to run faster
        fig, axs = plt.subplots(2, figsize=(
        10, 10))  ##update the number of suplots to equal the number of layers you want to display
        fig.suptitle(image_file_names_ordered[i])
        axs[0].imshow((X_smoothed[i, :, :, :][:, :, [2, 1, 0]]))
        # axs[0].imshow((X_smoothed[i,:,:,:][:,:,[5]]))
        axs[1].imshow(predictions[i, 0, :, :])  #

        plt.savefig(os.path.join(gif_creation_dir, f'{image_file_names_ordered[i]}.png'), dpi=100)
        # plt.show()

with imageio.get_writer(os.path.join(gif_output_dir, f"{glims_id}.gif"), mode='I') as writer:
    for filename in sorted(os.listdir(gif_creation_dir)):
        image = imageio.imread(os.path.join(gif_creation_dir, filename))
        writer.append_data(image)

# generate and save surface area time_series

ts_output_dir = os.path.join(os.path.expanduser("~"), "PycharmProjects", "glacier-view-analysis", "src", "segmentation",
                             "surface_area_time_series")
total_areas = []
for prediction in predictions:
    total_areas.append(np.sqrt(np.sum(prediction)))
plt.plot(image_dates, total_areas)
plt.title("Estimated Surface Area (no units)")
# plt.savefig(os.path.join(ts_output_dir, f"{glims_id}.png"))

plt.show()
"""