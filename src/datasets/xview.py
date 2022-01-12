from numpy.core.shape_base import _vhstack_dispatcher
import torch
import json
import os
import numpy as np
import cv2
import glob
import tifffile as tiff
# import pandas as pd
# from .aug import randomHueSaturationValue, randomShiftScaleRotate, randomHorizontalFlip, randomVerticleFlip, randomRotate90
from torch.utils import data
from PIL import Image
import albumentations as A
import h5py


class Xview3_Loader(data.Dataset):

    def __init__(self, root, split="train", img_size=(512, 512)):

        self.root = root
        self.split = split

        self.crop_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([102.53276416,  81.83246092])
        self.var = np.array([47.25883539, 45.92983817])

        self.img_names = []

        txt = open(os.path.join(root, "split_txt", "{}.txt".format(split))).readlines()

        for img_path in txt:
            self.img_names.append(os.path.join(root, img_path.strip()))

        print("> Found %d %s images..." % (len(self.img_names), split))

    def __len__(self):
        """__len__"""
        return len(self.img_names)

    def __getitem__(self, index):
        
        img_name = self.img_names[index]

        VV_path = img_name.replace("Mask_Vessel", "VV")
        VH_path = img_name.replace("Mask_Vessel", "VH")

        vessel_path = img_name
        no_vessel_path = img_name.replace("Mask_Vessel", "Mask_No_Vessel")

        conf_path = img_name.replace("Mask", "Mask_No_Vessel_Conf")

        VV = cv2.imread(VV_path, 0)
        VH = cv2.imread(VH_path, 0)
        
        vessel = cv2.imread(vessel_path, 0)
        others = cv2.imread(no_vessel_path, 0)
        # print(VV.shape, VH.shape, vessel.shape, others.shape)
        img = np.zeros((1024,1024,2))
        img[:,:,0] = VV
        img[:,:,1] = VH

        lbl = np.zeros((1024,1024,2))
        lbl[:,:,0] = vessel
        lbl[:,:,1] = others

        img, lbl = self.randn_crop(img, lbl)
        
        img = self.transform(img)
        
        lbl = lbl / 255.0
        lbl = lbl.transpose(2, 0, 1)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).float()     

        return img, lbl
    
    def transform(self, img):
        img = img.astype(float)
        img = (img / 255.0 - 0.5) * 2
        img = img.transpose(2, 0, 1)

        return img

    def randn_crop(self, img, lbl):
        h, w, _ = img.shape
        crop_h, crop_w = self.crop_size
        if crop_h < h:

            start_x = np.random.randint(0, w - crop_w)
            start_y = np.random.randint(0, h - crop_h)

            img_crop = img[start_y : start_y + crop_h, start_x : start_x + crop_w, :]
            lbl_crop = lbl[start_y : start_y + crop_h, start_x : start_x + crop_w]
            return img_crop, lbl_crop
        else:
            return img, lbl


class Xview3_Loader_h5(data.Dataset):

    def __init__(self, root, split="train", img_size=(512, 512)):

        self.root = root
        self.split = split

        self.crop_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([102.53276416,  81.83246092])
        self.var = np.array([47.25883539, 45.92983817])

        self.img_names = []
        # self.images_base = os.path.join(self.root, split+"_images")
        print(os.path.join(root, "split_txt", "{}.txt".format(split)))
        txt = open(os.path.join(root, "split_txt", "{}.txt".format(split))).readlines()
        # img_list = glob.glob("{}/{}_detect_crop/*/*_Mask.png".format(root, split))
        # img_list = []
        self.f = h5py.File("/amax/home/dlh/data/xview/xview3.h5", "r")
        for img_path in txt:
            self.img_names.append(img_path.strip())
        # self.img_names = self.img_names[:300]
        print("> Found %d %s images..." % (len(self.img_names), split))

    def __len__(self):
        """__len__"""
        return len(self.img_names)

    def __getitem__(self, index):
        
        img_name = self.img_names[index].split("/")
        key_name = img_name[1][:5] + "_" + img_name[2] + "_" + img_name[3].split("_Mask")[0]

        read_img = self.f[key_name]

        VH = read_img[3]
        VV = read_img[4]

        if img_name[1][:5] == "valid":
            # mask = read_img[0]
            vessel = read_img[2]
            others = read_img[1]
            vessel_pred = vessel
            others_pred = others
            conf = np.zeros_like(vessel, dtype=np.uint8) + 255
        else:
            vessel = read_img[9]
            others = read_img[7]
            vessel_pred = read_img[8]
            others_pred = read_img[6]
            conf = read_img[5]
            # mask = vessel_pred + others_pred 

        img = np.zeros((1024,1024,2))
        img[:,:,0] = VV
        img[:,:,1] = VH

        lbl = np.zeros((1024,1024,6))
        lbl[:,:,0] = vessel
        lbl[:,:,1] = others
        lbl[:,:,2] = vessel_pred
        lbl[:,:,3] = others_pred
        lbl[:,:,4] = conf
        lbl[:,:,5] = conf

        img, lbl = self.randn_crop(img, lbl)
        
        img = self.transform(img)
        
        lbl = lbl / 255.0
        lbl = lbl.transpose(2, 0, 1)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).float()     

        return img, lbl
    
    def transform(self, img):
        img = img.astype(float)
        # img -= self.mean
        # img /= self.var
        img = (img / 255.0 - 0.5) * 2
        img = img.transpose(2, 0, 1)

        return img

    def randn_crop(self, img, lbl):
        h, w, _ = img.shape
        crop_h, crop_w = self.crop_size
        # print(crop_h, h)
        if crop_h < h:

            start_x = np.random.randint(0, w - crop_w)
            start_y = np.random.randint(0, h - crop_h)

            img_crop = img[start_y : start_y + crop_h, start_x : start_x + crop_w, :]
            lbl_crop = lbl[start_y : start_y + crop_h, start_x : start_x + crop_w]
            return img_crop, lbl_crop
        else:
            return img, lbl


class Xview3_Class_Loader(data.Dataset):

    def __init__(self, root, txt_split="valid", img_size=(512, 512)):

        self.root = root

        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([102.53276416,  81.83246092])
        self.var = np.array([47.25883539, 45.92983817])

        self.img_names = []
        # self.images_base = os.path.join(self.root, split+"_class_crop")

        txt_file = open(os.path.join(root, "split_txt", "{}.txt".format(txt_split))).readlines()

        for line in txt_file:
            self.img_names.append(line.strip())
        # self.img_names = self.img_names[:640*2]
        print("> Found %d %s images..." % (len(self.img_names), txt_split))

    def __len__(self):
        """__len__"""
        return len(self.img_names)

    def __getitem__(self, index):
        #128443d1e98e2839v_13,False,nan,nan,9999.99,HIGH       
        line = self.img_names[index].split(",")
        split = line[-1]
        img_name = line[0]
        fold_name = img_name.split("_")[0]

        VV_path = os.path.join(os.path.join(self.root, split+"_class_crop"), fold_name, img_name + "_VV.png")
        VH_path = os.path.join(os.path.join(self.root, split+"_class_crop"), fold_name, img_name + "_VH.png")
        # print(VV_path)
        VV = cv2.resize(cv2.imread(VV_path, 0), (224,224))
        VH = cv2.resize(cv2.imread(VH_path, 0), (224,224))
        # print(VV_path, VV.shape)
        img = np.zeros((224,224,2))
        img[:,:,0] = VV
        img[:,:,1] = VH

        img = self.transform(img)
        
        is_vessel = self.transform_lbl(line[1])
        is_fishing = self.transform_lbl(line[2])

        if line[3] == "nan":
            len_vessel = -1
        else:
            len_vessel = float(line[3])
        
        # if line[4] == "9999.99":
        #     dis_vessel = -1
        # else:
        #     dis_vessel = float(line[4])

        img = torch.from_numpy(img).float()
        is_vessel = torch.from_numpy(np.array(is_vessel)).float()
        is_fishing = torch.from_numpy(np.array(is_fishing)).float()
        len_vessel = torch.from_numpy(np.array(len_vessel)).float()
        return img, is_vessel, is_fishing, len_vessel
    
    def transform(self, img):
        img = img.astype(float)
        # img -= self.mean
        # img /= self.var
        img = (img / 255.0 - 0.5) * 2
        img = img.transpose(2, 0, 1)

        return img


    def transform_lbl(self, str_class):
        out = 255
        if str_class == "False":
            out = 0
        if str_class == "True":
            out = 1        
        return out


if __name__ == '__main__':

    local_path = "/home/dlh/data/xview/"
    dst = Xview3_Loader(local_path, split="detect_merge_0_crop_valid_only", img_size=(1024, 1024))
    print(len(dst))
    bs = 1
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=4, shuffle=False)
    
    for i, data in enumerate(trainloader):
        # if i % 1000 == 0:
        print("batch :", i)
        img, lbl = data
        print(img.size(), lbl.size())
        img = img.squeeze().numpy()
        lbl = lbl.squeeze().numpy()
        img = img.transpose(1,2,0)
        img = ((img / 2 + 0.5) * 255).astype(np.uint8)
        lbl = (lbl * 255).astype(np.uint8)
        cv2.imwrite("/home/dlh/data/xview/data_loader_vis/{}_VV.png".format(i), img[:,:,0])
        cv2.imwrite("/home/dlh/data/xview/data_loader_vis/{}_VH.png".format(i), img[:,:,1])
        cv2.imwrite("/home/dlh/data/xview/data_loader_vis/{}_V.png".format(i), lbl[0])
        cv2.imwrite("/home/dlh/data/xview/data_loader_vis/{}_O.png".format(i), lbl[1])
        # print(img.size(), lbl.size(), aux_data_array.size())
