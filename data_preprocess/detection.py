import os
import glob
import pandas as pd
import tifffile as tiff
from osgeo import gdal
import numpy as np
import cv2
import multiprocessing
from tqdm  import tqdm
from numba import jit, float32, int32
from functools import partial

def mkdir_if_missing(d):
    if not os.path.exists(d):
        os.makedirs(d)


@jit(float32[:, :](float32[:, :], float32[:, :], int32[:, :], int32[:, :], float32), nopython=True, fastmath=True)
def apply_gaussian(accumulate_confid_map, centers, xx, yy, sigma):
    for i in range(len(centers)):
        center = centers[i]
        d2 = (xx - center[1]) ** 2 + (yy - center[0]) ** 2
        exponent = d2 / 2.0 / sigma / sigma
        mask = exponent <= 4.6052
        cofid_map = np.exp(-exponent)
        cofid_map = np.multiply(mask, cofid_map)
        max_mask = cofid_map > accumulate_confid_map
        accumulate_confid_map += cofid_map
        # accumulate_confid_map[max_mask] = cofid_map[max_mask]
    return accumulate_confid_map


def gen_gaussian_map(centers, shape, sigma):
    centers = np.float32(centers)
    sigma = np.float32(sigma)
    accumulate_confid_map = np.zeros(shape, dtype=np.float32)
    y_range = np.arange(accumulate_confid_map.shape[0], dtype=np.int32)
    x_range = np.arange(accumulate_confid_map.shape[1], dtype=np.int32)
    xx, yy = np.meshgrid(x_range, y_range)

    accumulate_confid_map = apply_gaussian(accumulate_confid_map, centers, xx, yy, sigma)
    accumulate_confid_map[accumulate_confid_map > 1.0] = 1.0
    
    return accumulate_confid_map


def detect_single(scene_path, detections, split):

    scene_id = scene_path.split("/")[-1].split("_")[0]
    scene_detects = detections[detections["scene_id"] == scene_id]
    
    band_VH = cv2.imread(scene_path, 0)
    band_VV = cv2.imread(scene_path.replace("VH.png","VV.png"), 0)
    img_height = band_VH.shape[0]  # 栅格矩阵的列数
    img_width = band_VH.shape[1]  # 栅格矩阵的行数
    
    patch_size = 1024
    stride = 1024 - 128
    detect_folder = os.path.join("../data/{}_detect_crop/".format(split), scene_id)
    mkdir_if_missing(detect_folder)

    mask_vessel = np.zeros((img_height, img_width))
    mask_no_vessel = np.zeros((img_height, img_width))

    for i in range(len(scene_detects)):
        row = scene_detects.iloc[i]
        detect_row = row["detect_scene_row"]
        detect_col = row["detect_scene_column"]
        confidence  = row["confidence"]
        is_vessel = row["is_vessel"]
        if confidence != "LOW":
            if is_vessel:
                mask_vessel[detect_row, detect_col] = 1
            else:
                mask_no_vessel[detect_row, detect_col] = 1

    for start_h in range(0, img_height-patch_size+stride, stride):
        for start_w in range(0, img_width-patch_size+stride, stride):
            cur_end_h = min(start_h+patch_size, img_height)
            cur_end_w = min(start_w+patch_size, img_width)
            cur_start_h = cur_end_h-patch_size
            cur_start_w = cur_end_w-patch_size

            band_VH_data = band_VH[cur_start_h:cur_end_h, cur_start_w:cur_end_w]
            band_VV_data = band_VV[cur_start_h:cur_end_h, cur_start_w:cur_end_w]
            
            if np.sum(band_VH_data) == 0:
                continue

            window_str = "{}_{}".format(cur_start_h, cur_start_w)
            img_name = window_str

            cv2.imwrite("{}/{}_VH.png".format(detect_folder, img_name), band_VH_data)
            cv2.imwrite("{}/{}_VV.png".format(detect_folder, img_name), band_VV_data)

            mask_data = mask_vessel[cur_start_h:cur_end_h, cur_start_w:cur_end_w]
            center_list = np.argwhere(mask_data == 1)
            mask_data = gen_gaussian_map(center_list, (patch_size, patch_size), 3) * 255           
            cv2.imwrite("{}/{}_Mask_Vessel.png".format(detect_folder, img_name), mask_data)

            mask_no_vessel_data = mask_no_vessel[cur_start_h:cur_end_h, cur_start_w:cur_end_w]
            center_list = np.argwhere(mask_no_vessel_data == 1)
            mask_data = gen_gaussian_map(center_list, (patch_size, patch_size), 3) * 255           
            cv2.imwrite("{}/{}_Mask_No_Vessel.png".format(detect_folder, img_name), mask_data)


def main_detect(split="valid"):
    detect_file = "../data/raw_data/{}.csv".format(split)
    detections = pd.read_csv(detect_file, low_memory=False)
    scene_list = glob.glob("../data/{}_8bit/*VH.png".format(split))

    pool = multiprocessing.Pool(processes=4)
    with tqdm(total=len(scene_list), ncols=120) as t:
        for _ in pool.imap_unordered(partial(detect_single, detections=detections, split=split), scene_list):
            t.update(1)


def get_img_path_list(scene_id, scene_detects, split):
    crop_list = []
    stride = 1024 - 128
    detect_folder = os.path.join("../data/{}_detect_crop/".format(split), scene_id)

    img_all_list = glob.glob("../data/{}_detect_crop/{}/*_Mask_Vessel.png".format(split, scene_id))

    for i in range(len(scene_detects)):
        row = scene_detects.iloc[i]
        detect_row = row["detect_scene_row"]
        detect_col = row["detect_scene_column"]
        distance_from_shore_km = row["distance_from_shore_km"]

        confidence  = row["confidence"]
        if confidence != "LOW":
            start_h = (detect_row // stride) * stride
            start_w = (detect_col // stride) * stride
            window_str = "{}_{}".format(start_h, start_w)
            crop_path = os.path.join(detect_folder, "{}_Mask_Vessel.png".format(window_str))
            if os.path.isfile(crop_path) and crop_path not in crop_list:
                crop_list.append(crop_path)
    return crop_list, img_all_list


def split_k_fold_detect_valid(split = "valid", k_fold=5):
    mkdir_if_missing("../data/split_txt/")
    detect_file = "../data/raw_data/{}.csv".format(split)
    detections = pd.read_csv(detect_file, low_memory=False)

    scene_list = glob.glob("../data/{}_8bit/*VH.png".format(split))
   
    for k in range(k_fold):
        
        fold_txt_train_only = open("../data/split_txt/detect_{}_{}_crop_train_only.txt".format(split, k), "w")
        fold_txt_train_all = open("../data/split_txt/detect_{}_{}_crop_train_all.txt".format(split, k), "w")

        fold_txt_valid_only = open("../data/split_txt/detect_{}_{}_crop_valid_only.txt".format(split, k), "w")
        fold_txt_valid_all = open("../data/split_txt/detect_{}_{}_crop_valid_all.txt".format(split, k), "w")

        pbar = tqdm(total=len(scene_list), ncols=120)
        for i in range(len(scene_list)):
            # print(scene_path)
            scene_path = scene_list[i]
            scene_id = scene_path.split("/")[-1].split("_")[0]
            scene_detects = detections[detections["scene_id"] == scene_id]
            pbar.update(1)
            if i % k_fold == k:
                valid_only_list, valid_all_list = get_img_path_list(scene_id, scene_detects, split)
                for path in valid_only_list:
                    fold_txt_valid_only.write(path+"\n")
                for path in valid_all_list:
                    fold_txt_valid_all.write(path+"\n")
            else:    
                train_only_list, train_all_list = get_img_path_list(scene_id, scene_detects, split)
                for path in train_only_list:
                    fold_txt_train_only.write(path+"\n")
                for path in train_all_list:
                    fold_txt_train_all.write(path+"\n")
        pbar.close()


if __name__=='__main__':

    # main_detect(split="valid")
    split_k_fold_detect_valid(split ="valid", k_fold=5)