import os
import glob
import pandas as pd
from osgeo import gdal
import numpy as np
import cv2
import multiprocessing
from tqdm  import tqdm
from functools import partial


def mkdir_if_missing(d):
    if not os.path.exists(d):
        os.makedirs(d)


def class_single(scene_path, detections, split):

    scene_id = scene_path.split("/")[-1].split("_")[0]
    scene_detects = detections[detections["scene_id"] == scene_id]

    VH_dataset = gdal.Open(scene_path)
    VV_dataset = gdal.Open(scene_path.replace("VH.png","VV.png"))

    img_w = VV_dataset.RasterXSize
    img_h = VV_dataset.RasterYSize

    class_folder = os.path.join("../data/{}_class_crop/".format(split), scene_id)
    mkdir_if_missing(class_folder)

    for i in range(len(scene_detects)):
        row = scene_detects.iloc[i]
        detect_row = int(row["detect_scene_row"])
        detect_col = int(row["detect_scene_column"])

        vessel_length_m = row["vessel_length_m"]
        is_vessel = row["is_vessel"]
        is_fishing = row["is_fishing"]
        distance_from_shore_km = row["distance_from_shore_km"]
        confidence  = row["confidence"]
        if confidence != "LOW":
            img_name = scene_id + "_{}".format(i)
            if not np.isnan(is_vessel):
                xoff = max(0, detect_col-112)
                yoff = max(0, detect_row-112)
                crop_h, crop_w = 224, 224
                if yoff + 224 > img_h:
                    crop_h = img_h - yoff
                if xoff + 224 > img_w:
                    crop_w = img_w - xoff 
                crop_VV = VV_dataset.ReadAsArray(xoff, yoff, crop_w, crop_h)
                crop_VH = VH_dataset.ReadAsArray(xoff, yoff, crop_w, crop_h)
                cv2.imwrite("{}/{}_VV.png".format(class_folder, img_name), crop_VV)
                cv2.imwrite("{}/{}_VH.png".format(class_folder, img_name), crop_VH)


def main_class(split = "train"):
    detect_file = "../data/raw_data/{}.csv".format(split)
    detections = pd.read_csv(detect_file, low_memory=False)
    scene_list = glob.glob("../data/{}_8bit/*VH.png".format(split))

    pool = multiprocessing.Pool(processes=8)
    with tqdm(total=len(scene_list), ncols=120) as t:
        for _ in pool.imap_unordered(partial(class_single, detections=detections, split=split), scene_list):
            t.update(1)


def get_img_path_list_class(scene_id, scene_detects, split):
    img_name_list = []
    class_folder = os.path.join("../data/{}_class_crop/".format(split), scene_id)
    for i in range(len(scene_detects)):
        row = scene_detects.iloc[i]
        # detect_row = row["detect_scene_row"]
        # detect_col = row["detect_scene_column"]

        vessel_length_m = row["vessel_length_m"]
        is_vessel = row["is_vessel"]
        is_fishing = row["is_fishing"]
        distance_from_shore_km = row["distance_from_shore_km"]
        confidence  = row["confidence"]
        if confidence != "LOW":
            img_name = scene_id + "_{}".format(i)
            if not np.isnan(is_vessel):
                line = "{},{},{},{},{},{},{}".format(
                    img_name, is_vessel, is_fishing, vessel_length_m, distance_from_shore_km, confidence, split)
                img_name_list.append(line)
    return img_name_list


def split_k_fold_class(split = "valid", k_fold=5):
    detect_file = "../data/raw_data/{}.csv".format(split)
    detections = pd.read_csv(detect_file, low_memory=False)

    scene_list = glob.glob("../data/{}_8bit/*VH.png".format(split))
    mkdir_if_missing("../data/split_txt/")
    for k in range(k_fold):
        
        fold_txt_train = open("../data/split_txt/class_{}_{}_crop_train.txt".format(split, k), "w")
        fold_txt_valid = open("../data/split_txt/class_{}_{}_crop_valid.txt".format(split, k), "w")

        pbar = tqdm(total=len(scene_list), ncols=120)
        for i in range(len(scene_list)):
            # print(scene_path)
            scene_path = scene_list[i]
            scene_id = scene_path.split("/")[-1].split("_")[0]
            scene_detects = detections[detections["scene_id"] == scene_id]
            pbar.update(1)
            if i % k_fold == k:
                valid_list = get_img_path_list_class(scene_id, scene_detects, split)
                for path in valid_list:
                    fold_txt_valid.write(path+"\n")
            else:    
                train_list = get_img_path_list_class(scene_id, scene_detects, split)
                for path in train_list:
                    fold_txt_train.write(path+"\n")
        pbar.close()


def merge_k_fold_class_txt(split = "merge"):
    for k in range(10):
        for s in ["train", "valid"]:
            txt_write = open("../data/split_txt/class_{}_{}_crop_{}.txt".format(split, k, s), "w")
            for s_n in ["train", "valid"]:
                if s_n == "valid":
                    k_ = k % 5
                else:
                    k_ = k
                txt_read = open("../data/split_txt/class_{}_{}_crop_{}.txt".format(s_n, k_, s), "r").readlines()
                for line in txt_read:
                    txt_write.write(line)
            txt_write.close()


if __name__=='__main__':
    main_class(split = "train")
    main_class(split = "valid")
    split_k_fold_class(split = "valid", k_fold=5)
    split_k_fold_class(split = "train", k_fold=10)
    merge_k_fold_class_txt()
