import os
import glob
from osgeo import gdal
import numpy as np
import cv2


def convert_to_8bit(band_data, lower_percent=0.2, higher_percent=99.8):

    valid_mask = (band_data != -32768.0)
    band_lower = np.percentile(band_data[valid_mask], lower_percent)
    band_higher = np.percentile(band_data[valid_mask], higher_percent)

    band_data = (band_data - band_lower) * 255 / (band_higher - band_lower)
    
    band_data[band_data<0] = 0
    band_data[band_data>255] = 255
    band_data[~valid_mask] = 0

    return band_data.astype(np.uint8)


def convert(split = "train"):

    scene_list = glob.glob("../data/raw_data/{}/*/VH_dB.tif".format(split))
    save_path = "../data/{}_8bit".format(split)
    if not os.path.exists(save_path):
        os.makedirs(save_path)    
    print(len(scene_list))
    for scene_path in scene_list:
        scene_id = scene_path.split("/")[-2]
        print(scene_id)

        VH_dataset = gdal.Open(scene_path)
        VV_dataset = gdal.Open(scene_path.replace("VH_dB.tif","VV_dB.tif"))
        img_width = VH_dataset.RasterXSize
        img_height = VH_dataset.RasterYSize

        band_VH = VH_dataset.GetRasterBand(1).ReadAsArray(0,0, img_width, img_height)
        band_VV = VV_dataset.GetRasterBand(1).ReadAsArray(0,0, img_width, img_height)

        band_VH = convert_to_8bit(band_VH)
        band_VV = convert_to_8bit(band_VV)

        cv2.imwrite(os.path.join(save_path, "{}_VH.png".format(scene_id)), band_VH)
        cv2.imwrite(os.path.join(save_path, "{}_VV.png".format(scene_id)), band_VV)


if __name__=='__main__':

    convert("train")
    convert("valid")

