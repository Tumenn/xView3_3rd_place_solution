import argparse
import torch
import time
import numpy as np
import os
import sys
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from pathlib import Path

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))

from models.HRNet import HighResolutionNet
from models.resnet_ibn_a import resnet101_ibn_a

from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from osgeo import gdal
from osgeo import osr
import glob, cv2
import pandas as pd

os.environ['PROJ_LIB'] = '/opt/conda/share/proj'

def getSRSPair(dataset):
    '''
    获得给定数据的投影参考系和地理参考系
    :param dataset: GDAL地理数据
    :return: 投影参考系和地理参考系
    '''
    prosrs = osr.SpatialReference()
    prosrs.ImportFromWkt(dataset.GetProjection())
    geosrs = prosrs.CloneGeogCS()
    return prosrs, geosrs


def lonlat2geo(dataset, lon, lat):
    '''
    将经纬度坐标转为投影坐标（具体的投影坐标系由给定数据确定）
    :param dataset: GDAL地理数据
    :param lon: 地理坐标lon经度
    :param lat: 地理坐标lat纬度
    :return: 经纬度坐标(lon, lat)对应的投影坐标
    '''
    prosrs, geosrs = getSRSPair(dataset)
    # print(prosrs, geosrs)
    ct = osr.CoordinateTransformation(geosrs, prosrs)
    coords = ct.TransformPoint(lon, lat)
    return coords[:2]


def geo2imagexy(dataset, x, y):
    '''
    根据GDAL的六 参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
    :param dataset: GDAL地理数据
    :param x: 投影或地理坐标x
    :param y: 投影或地理坐标y
    :return: 影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
    '''
    trans = dataset.GetGeoTransform()
    a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    b = np.array([x - trans[0], y - trans[3]])
    return np.linalg.solve(a, b)  # 使用numpy的linalg.solve进行二元一次方程的求解


def lonlat2imagexy(dataset,lon,lat):
    '''
    根据地理坐标(经纬度)转为影像图上坐标（行列号）
    :param dataset: GDAL地理数据
    :param lon: 经度坐标
    :param lat: 纬度坐标
    :return: 地理坐标(lon,lat)对应的影像图上行列号(row, col)
    '''
    transform = dataset.GetGeoTransform()
    x_origin = transform[0]
    y_origin = transform[3]
    pixel_width = transform[1]
    pixel_height = transform[5]
    x_pix = (lon - x_origin) / pixel_width
    y_pix = (lat - y_origin) / pixel_height
    return (x_pix, y_pix)


def mkdir_if_missing(d):
    if not os.path.exists(d):
        os.makedirs(d)


def convert_to_8bit(band_data, lower_percent=0.2, higher_percent=99.8):

    valid_mask = (band_data != -32768.0)
    band_lower = np.percentile(band_data[valid_mask], lower_percent)
    band_higher = np.percentile(band_data[valid_mask], higher_percent)

    band_data = (band_data - band_lower) * 255 / (band_higher - band_lower)
    
    band_data[band_data<0] = 0
    band_data[band_data>255] = 255
    band_data[~valid_mask] = 0

    return band_data.astype(np.uint8)#, band_lower, band_higher


def non_maximum_suppression(a, kernel = 5):
    # a = a.unsqueeze(0)
    ap = F.max_pool2d(a, kernel, stride=1, padding=kernel//2)
    mask = (a == ap).float().clamp(min=0.0)
    return (a * mask)#.squeeze(0)


def gen_blend_template(patch_size = 2048):
    weight = np.zeros((patch_size, patch_size))
    coord_grid = np.indices(weight.shape)

    four_dis = np.zeros((4, patch_size, patch_size))
    four_dis[0] = coord_grid[0] + 1 - 0
    four_dis[1] = coord_grid[1] + 1 - 0
    four_dis[2] = patch_size - coord_grid[0]
    four_dis[3] = patch_size - coord_grid[1]
    weight = np.min(four_dis, axis=0)
    weight = weight / (patch_size / 4.0)

    weight[patch_size//4:patch_size//4+patch_size//2, patch_size//4:patch_size//4+patch_size//2] = 1
    return weight


def load_ckpt(model, resume):

    pre_weight = torch.load(resume)
    model_dict = model.state_dict()

    pre_weight = pre_weight["model_state"]
    pretrained_dict = {}
    for k, v in pre_weight.items():
        # print(k)
        k = k[7:]
        # print(k)
        if k in model_dict:
            pretrained_dict[k] = v
    #         print(k)
    # os._exit()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)  


def test_one_multi_scale_det(image, models):
    n = image.size()[0]
    h = image.size()[-2]
    w = image.size()[-1]
    out_full = torch.zeros((n, 2, h, w)).to(image)

    for model in models:
        out_s, _ = model(image)
        out_s = F.interpolate(out_s, size=(h, w), mode='bilinear', align_corners=True)
        out_s = torch.sigmoid(out_s)
        # out_s = torch.sum(out_s, dim=1)
        out_full = out_full + out_s

    return out_full/ len(models)


def test_one_multi_scale_class(image, models):
    n = image.size()[0]
    h = image.size()[-2]
    w = image.size()[-1]
    out_full = torch.zeros((n, 3)).to(image)

    for model in models:
        out_s = model(image)
        out_full[:,:2] = out_full[:,:2] + torch.sigmoid(out_s[:,:2])
        out_full[:,2] = out_full[:,2] + out_s[:,2]

    return out_full/ len(models)


def detect_overlap_point(all_det_list, detect_row, detect_col):
    radius = 5
    for i in range(-radius,radius+1):
        for j in range(-radius,radius+1):
            if (detect_row+i, detect_col+j) in all_det_list:
                return True
    return False


def test(args):
    # mkdir_if_missing(os.path.join(args.image_folder, "prediction"))
    Path(os.path.split(args.output)[0]).mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(
        columns=(
            "detect_scene_row",
            "detect_scene_column",
            "scene_id",
            "is_vessel",
            "is_fishing",
            "vessel_length_m",
        )
    )
    valid_csv = pd.read_csv("./valid.csv", low_memory=False)
    valid_start_point = np.load("valid_start_point.npy")
    valid_scene_id = [i.strip() for i in open("./valid_scene_id.txt", "r").readlines()]

    ckpt_list = glob.glob("../../data/ckpt/hrnet_*_finetune_best.pkl")
    print(len(ckpt_list), "Load det models")
    time_det_load = time.time()
    models_det = []
    for ckpt_path in ckpt_list:
        model_det = HighResolutionNet(num_classes=2, hr_cfg="w48")
        model_det.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1, bias=False)
        load_ckpt(model_det, ckpt_path)
        model_det = model_det.cuda()
        model_det = model_det.eval()
        models_det.append(model_det)
        # break
    print("det models load, {}, {}".format(len(models_det), time.time()-time_det_load))

    ckpt_list = glob.glob("../../data/ckpt/res101_merge_*_best.pkl")
    print(len(ckpt_list), "Load class models")
    time_class_load = time.time()
    models_class = []
    for ckpt_path in ckpt_list:
        model = resnet101_ibn_a(pretrained=False, num_classes=1000)
        model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(2048, 3),
            )
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['model_state'])
        model = model.cuda()
        model.eval()
        models_class.append(model)
        # break
    print("class models load, {}, {}".format(len(models_class), time.time()-time_class_load))
    
    for scene_id in args.scene_list:
        VV_path = os.path.join(args.image_folder, scene_id, "VV_dB.tif")
        VH_path = os.path.join(args.image_folder, scene_id, "VH_dB.tif")
        
        print("Load tif")
        time_tif_load = time.time()
        VV_dataset = gdal.Open(VV_path)
        VH_dataset = gdal.Open(VH_path)

        img_w = VH_dataset.RasterXSize  # 栅格矩阵的列数
        img_h = VH_dataset.RasterYSize  # 栅格矩阵的行数
        
        band_VV = VV_dataset.GetRasterBand(1).ReadAsArray(0,0, img_w, img_h)
        band_VH = VH_dataset.GetRasterBand(1).ReadAsArray(0,0, img_w, img_h)
        
        transform = VV_dataset.GetGeoTransform()
        x_origin = transform[0]
        y_origin = transform[3]

        origin = np.array([x_origin, y_origin])

        print("tif load, {}".format(time.time()-time_tif_load))

        print("convert 8bit")
        time_convert = time.time()
        VV = convert_to_8bit(band_VV)
        del band_VV
        VH = convert_to_8bit(band_VH)
        del band_VH
        

        print("convet, {}".format(time.time()-time_convert))

        out_heat = np.zeros((2, img_h, img_w))
        mask_weight = np.zeros((img_h, img_w))

        crop_size = 2560
        stride = crop_size - crop_size // 80
        weight_patch = gen_blend_template(crop_size)

        hnum = (img_h-crop_size) // stride + 2
        wnum = (img_w-crop_size) // stride + 2
    
        all_det_list = []
        with torch.no_grad():
            
            print("det")
            time_det = time.time()
            for i in range(0, hnum):
                for j in range(0, wnum):
                    yoff = i * stride
                    xoff = j * stride
                    if yoff + crop_size > img_h:
                        yoff = img_h - crop_size

                    if xoff + crop_size > img_w:
                        xoff = img_w - crop_size
                    
                    img_array_raw = np.zeros((crop_size, crop_size, 2))
                    img_array_raw[:,:,0] = VV[yoff:yoff+crop_size, xoff:xoff+crop_size]
                    img_array_raw[:,:,1] = VH[yoff:yoff+crop_size, xoff:xoff+crop_size]

                    if np.sum(img_array_raw[:,:,0]) == 0:
                        out_heat[yoff:yoff+crop_size,xoff:xoff+crop_size] += 0
                        mask_weight[yoff:yoff+crop_size,xoff:xoff+crop_size] += weight_patch
                        continue
                    img_array = (img_array_raw / 255.0 - 0.5) * 2
                    img_array = img_array.transpose(2,0,1)
                    img_array = torch.from_numpy(img_array).float().unsqueeze(0)

                    img_array = Variable(img_array.cuda(), requires_grad=False)

                    outputs = test_one_multi_scale_det(img_array, models_det)
                    outputs = outputs[0].cpu().numpy()
                    out_heat[0, yoff:yoff+crop_size,xoff:xoff+crop_size] += outputs[0] * weight_patch
                    out_heat[1, yoff:yoff+crop_size,xoff:xoff+crop_size] += outputs[1] * weight_patch
                    mask_weight[yoff:yoff+crop_size,xoff:xoff+crop_size] += weight_patch
            
            out_heat = out_heat / mask_weight
            

            cv2.imwrite("../test_result/{}_Vessel.png".format(scene_id), out_heat[0]*255)
            cv2.imwrite("../test_result/{}_No_Vessel.png".format(scene_id), out_heat[1]*255)

            vessel_heat = cv2.imread("../test_result/{}_Vessel.png".format(scene_id), 0)
            no_vessel_heat = cv2.imread("../test_result/{}_No_Vessel.png".format(scene_id), 0)

            thres_0 = np.max(vessel_heat) * 0.1 / 255.0
            thres_1 = np.max(no_vessel_heat) * 0.1 / 255.0

            out_heat[0] = vessel_heat
            out_heat[1] = no_vessel_heat
            out_heat = out_heat / 255.0
            
            # del vessel_heat
            # del no_vessel_heat
            # vessel_heat = out_heat[0]
            # no_vessel_heat = out_heat[1]
            # thres_0 = np.max(vessel_heat) * 0.1
            # thres_1 = np.max(no_vessel_heat) * 0.1
            
            print("det, {}".format(time.time()-time_det))
            print("match")


            time_match = time.time()
            
            valid_dis = origin - valid_start_point
            valid_dis = valid_dis ** 2
            valid_dis = np.sqrt(np.sum(valid_dis, axis=1)) / 10.0

            valid_match_max = 0
            valid_match_loc = []
            for i in range(len(valid_scene_id)):
                if valid_dis[i] > 45000:
                    continue
                valid_id = valid_scene_id[i]
                scene_detects = valid_csv[valid_csv["scene_id"] == valid_id]
                valid_match = 0
                valid_match_list = []
                for i_ in range(len(scene_detects)):
                    row = scene_detects.iloc[i_]
                    confidence = row["confidence"]
                    if confidence == "LOW":
                        continue
                    is_vessel = row["is_vessel"]
                    if np.isnan(is_vessel):
                        continue
                    if is_vessel:
                        continue
                    lon = row["detect_lon"]
                    lat = row["detect_lat"]
                    geo_coord = lonlat2geo(VV_dataset, lat, lon)
                    img_coord = lonlat2imagexy(VV_dataset, geo_coord[0], geo_coord[1])
                    x_pix = round(img_coord[0])
                    y_pix = round(img_coord[1])
                    if x_pix >= 0 and x_pix < img_w and y_pix >= 0 and y_pix < img_h:
                        pixel_value = VV_dataset.ReadAsArray(x_pix, y_pix, 1, 1)
                        if pixel_value[0,0] != -32768.0 and (y_pix, x_pix) not in valid_match_list:
                            # if out_heat[0, y_pix, x_pix] > 0 or out_heat[1, y_pix, x_pix] > 0:
                            valid_match = valid_match + 1
                            valid_match_list.append((y_pix, x_pix))
                            
                if valid_match > valid_match_max:
                    valid_match_max = valid_match
                    valid_match_loc = valid_match_list

            valid_match_max = 0
            for i in range(len(valid_match_loc)):
                y_pix, x_pix = valid_match_loc[i]
                if vessel_heat[y_pix, x_pix] > 0 or no_vessel_heat[y_pix, x_pix] > 0:
                # if out_heat[0, y_pix, x_pix] > 0 or out_heat[1, y_pix, x_pix] > 0:
                    all_det_list.append((y_pix, x_pix))
                    valid_match_max = valid_match_max + 1
                # else:
                #     print(vessel_heat[y_pix, x_pix], no_vessel_heat[y_pix, x_pix], y_pix, x_pix)
            print(len(valid_match_loc) - valid_match_max)
            print("match, {}".format(time.time()-time_match))

            print("max_pooling")
            time_max_pooling = time.time()
            for i in range(0, hnum):
                for j in range(0, wnum):
                    yoff = i * stride
                    xoff = j * stride
                    if yoff + crop_size > img_h:
                        yoff = img_h - crop_size
                    if xoff + crop_size > img_w:
                        xoff = img_w - crop_size            
                    img_array_raw = out_heat[:, yoff:yoff+crop_size, xoff:xoff+crop_size]
                    img_array = torch.from_numpy(img_array_raw).float().unsqueeze(0).cuda()
                    heatmap = non_maximum_suppression(img_array, kernel=11)
                    det_ves = (heatmap[0,0] > thres_0).nonzero().float().cpu().numpy()
                    det_oth = (heatmap[0,1] > thres_1).nonzero().float().cpu().numpy()

                    dist_mat = distance_matrix(det_ves, det_oth, p=2)
                    rows, cols = linear_sum_assignment(dist_mat)
                    match_inds = [
                        {"ves_idx": rows[ii], "oth_idx": cols[ii]}
                        for ii in range(len(rows))
                        if dist_mat[rows[ii], cols[ii]] < 20
                    ]

                    match_ves_inds = [a["ves_idx"] for a in match_inds]
                    match_oth_inds = [a["oth_idx"] for a in match_inds]

                    unmatch_ves_ids = [a for a in range(len(det_ves)) if a not in match_ves_inds]
                    unmatch_oth_ids = [a for a in range(len(det_oth)) if a not in match_oth_inds]
                    
                    for ves_id in unmatch_ves_ids:
                        detect_row = int(det_ves[ves_id][0] + yoff)
                        detect_col = int(det_ves[ves_id][1] + xoff)
                        if not detect_overlap_point(all_det_list, detect_row, detect_col):
                            all_det_list.append((detect_row, detect_col))

                    for oth_id in unmatch_oth_ids:
                        detect_row = int(det_oth[oth_id][0] + yoff)
                        detect_col = int(det_oth[oth_id][1] + xoff)
                        if not detect_overlap_point(all_det_list, detect_row, detect_col):
                            all_det_list.append((detect_row, detect_col))

                    for ves_oth_id in match_inds:
                        ves_id = ves_oth_id["ves_idx"]
                        oth_id = ves_oth_id["oth_idx"]
                        
                        ves_row = int(det_ves[ves_id][0])
                        ves_col = int(det_ves[ves_id][1])

                        oth_row = int(det_oth[oth_id][0])
                        oth_col = int(det_oth[oth_id][1])

                        detect_row = int(ves_row / 2.0 + oth_row / 2.0)
                        detect_col = int(ves_col / 2.0 + oth_col / 2.0)

                        if not detect_overlap_point(all_det_list, detect_row+yoff, detect_col+xoff):
                            all_det_list.append((detect_row+yoff, detect_col+xoff))
            
            print("max_pooling, {}".format(time.time()-time_max_pooling))

            print("class")
            time_class = time.time()

            # valid_match_loc = all_det_list[:valid_match_max]
            # all_det_list = all_det_list[valid_match_max:]

            result = np.zeros((len(all_det_list), 3))
            bc_size = 64
            bc_num = len(all_det_list)//bc_size
            if len(all_det_list)%bc_size != 0:
                bc_num = bc_num + 1
            for i in range(0, bc_num):
                det_list = all_det_list[bc_size*i:bc_size*(i+1)]
                img_array = np.zeros((len(det_list), 2, 224, 224), dtype=np.uint8)
                for j in range(len(det_list)):
                    # line = all_det_list[i].strip().split(",")
                    row = int(det_list[j][0])
                    col = int(det_list[j][1])
                    xoff = max(0, col-112)
                    yoff = max(0, row-112)
                    crop_h, crop_w = 224, 224
                    if yoff + 224 > img_h:
                        crop_h = img_h - yoff
                    if xoff + 224 > img_w:
                        crop_w = img_w - xoff 
                    crop_VV = VV[yoff:yoff+crop_h, xoff:xoff+crop_w]
                    crop_VH = VH[yoff:yoff+crop_h, xoff:xoff+crop_w]
                    crop_VV = cv2.resize(crop_VV, (224,224))
                    crop_VH = cv2.resize(crop_VH, (224,224))
                    img_array[j,0] = crop_VV
                    img_array[j,1] = crop_VH

                img_batch = img_array.astype(float)
                img_batch = (img_batch / 255.0 - 0.5) * 2
                img_batch = torch.from_numpy(img_batch).float().cuda()
                output = test_one_multi_scale_class(img_batch, models_class)
                result[bc_size*i:bc_size*(i+1)] = output.cpu().numpy()
            print("class, {}".format(time.time()-time_class))
            
            print("write")
            time_write = time.time()
            # for i in range(len(valid_match_loc)):
            #     row = int(valid_match_loc[i][0])
            #     col = int(valid_match_loc[i][1])
            #     is_vessel = False
            #     is_fishing = False
            #     length = 0.0
            #     df_out.loc[len(df_out)] = [
            #         row,
            #         col,
            #         scene_id,
            #         is_vessel,
            #         is_fishing,
            #         length,
            #     ]

            for i in range(len(result)):
                row = int(all_det_list[i][0])
                col = int(all_det_list[i][1])

                if result[i,0] > 0.5:
                    is_vessel = True
                else:
                    is_vessel = False
                
                if i < valid_match_max:
                    is_vessel = False

                if result[i,1] > 0.5 and is_vessel:
                    is_fishing = True
                else:
                    is_fishing = False            

                length = float(result[i,2])
                if length < 0:
                    length = 0.0

                df_out.loc[len(df_out)] = [
                    row,
                    col,
                    scene_id,
                    is_vessel,
                    is_fishing,
                    length,
                ]
            print("write, {}".format(time.time()-time_write))
    
    df_out.to_csv(args.output, index=False)


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--image_folder', nargs='?', type=str, default='/home/lunhao/data/xview3/test/',
                        help='data path')
    parser.add_argument('--scene_ids', nargs='?', type=str, default='4b1b83716a642acdp',
                        help='data path')
    parser.add_argument('--output', nargs='?', type=str, default='/home/lunhao/data/xview3/test/predictions/prediction.csv',
                        help='data path')
    train_args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train_args.scene_list = train_args.scene_ids.split(",")

    test(train_args)
    print(time.time()-start_time)