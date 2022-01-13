import argparse
import torch
import time
import math
import numpy as np
import os
import sys
from torch.autograd import Variable
from torch.utils import data
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.backends.cudnn as cudnn

from tqdm import tqdm
import random

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))

from datasets.xview import Xview3_Loader
from models.HRNet import HighResolutionNet
from utils import AverageTracker
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree, distance_matrix

def non_maximum_suppression(a, kernel = 5):
    # a = a.unsqueeze(0)
    ap = F.max_pool2d(a, kernel, stride=1, padding=kernel//2)
    mask = (a == ap).float().clamp(min=0.0)
    return (a * mask)#.squeeze(0)


def train(args):

    if not os.path.exists("{}/ckpt".format(args.data_path)):
        os.makedirs("{}/ckpt".format(args.data_path))    

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    img_size = (args.img_rows, args.img_cols)

    if args.finetune:
        train_loader = Xview3_Loader(args.data_path, split="detect_valid_{}_crop_train_all".format(args.fold), img_size=img_size)
        valid_loader = Xview3_Loader(args.data_path, split="detect_valid_{}_crop_valid_all".format(args.fold), img_size=(1024, 1024))
    else:
        train_loader = Xview3_Loader(args.data_path, split="detect_valid_{}_crop_train_only".format(args.fold), img_size=img_size)
        valid_loader = Xview3_Loader(args.data_path, split="detect_valid_{}_crop_valid_only".format(args.fold), img_size=(1024, 1024))

    hr_cfg = args.width
    model = HighResolutionNet(num_classes=2, hr_cfg=hr_cfg)
    model.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1, bias=False)
    if args.finetune:
        print("initial weight from imagenet")
        args.resume = "{}/ckpt/{}_scrach_best.pkl".format(args.data_path, args.model)
        model.load_state_dict(torch.load(args.resume)['model_state'])
    else:
        # model.init_weights("../../data/ckpt/hrnetv2_{}_imagenet_pretrained.pth".format(hr_cfg))
        print("initial weight from imagenet")
        model.init_weights("/amax/home/dlh/data/pretrained/hrnetv2_w48_imagenet_pretrained.pth")
    # return
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.l_rate)

    args.start_epoch = 0

    best_f1 = 0
    best_recall = 0
    no_optim = 0

    train_loader = data.DataLoader(train_loader, batch_size=args.batch_size, num_workers=32, shuffle=True)
    valid_loader = data.DataLoader(valid_loader, batch_size=args.batch_size, num_workers=32)

    scaler = torch.cuda.amp.GradScaler()

    num_batches = len(train_loader)
    curr_iter = 0

    for epoch in range(args.start_epoch, args.n_epoch):
        model.train()

        train_loss = AverageTracker()
        pbar = tqdm(np.arange(num_batches), ncols=180)
        for train_i, (img, lbl) in enumerate(train_loader):
            curr_iter += 1
            optimizer.zero_grad()

            img = Variable(img.cuda())
            lbl = Variable(lbl.cuda())

            with torch.cuda.amp.autocast():
                output, _ = model(img)
                output = torch.sigmoid(output)
                output = F.interpolate(output, size=img_size, mode='bilinear', align_corners=True)
                loss = F.mse_loss(output, lbl, reduction='mean')
                        
            train_loss.update(loss.item())

            pbar.update(1)
            pbar.set_description("> Epoch [%d/%d]" % (epoch + 1, args.n_epoch))
            pbar.set_postfix(AVe_Loss = train_loss.avg, Loss=loss.item())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()

        C_gt_sum = 0
        C_true = 0
        C_false = 0
        for i_val, (img, lbl) in enumerate(valid_loader):
            with torch.no_grad():
                img = Variable(img.cuda())
                lbl = Variable(lbl.cuda())

                output, _ = model(img)
                output = torch.sigmoid(output)
                output = F.interpolate(output, size=(1024, 1024), mode='bilinear', align_corners=True)
                output = torch.sum(output, dim=1)
                lbl = torch.sum(lbl, dim=1)

                output = non_maximum_suppression(output.unsqueeze(1), kernel=5).squeeze()
                lbl = non_maximum_suppression(lbl.unsqueeze(1), kernel=5).squeeze()
                
                for bs_id in range(len(output)):
                    out_i = output[bs_id]
                    lbl_i = lbl[bs_id]

                    pred_point = (out_i > 0.05).nonzero().float().cpu().numpy()
                    gt_point = (lbl_i > 0.1).nonzero().float().cpu().numpy()
                    # print(len(lbl), len(output))
                    dist_mat = distance_matrix(pred_point, gt_point, p=2)
                    rows, cols = linear_sum_assignment(dist_mat)

                    tp_inds = [
                        {"pred_idx": rows[ii], "gt_idx": cols[ii]}
                        for ii in range(len(rows))
                        if dist_mat[rows[ii], cols[ii]] < 20
                    ]
                    C_true = C_true + len(tp_inds)
                    C_false = C_false + len(pred_point) - len(tp_inds)
                    C_gt_sum = C_gt_sum + len(gt_point)

        confusion_matrix = np.array([C_false, C_true, C_gt_sum])
        C_acc = confusion_matrix[1] * 1.0 / (confusion_matrix[1] + confusion_matrix[0] + 1e-6)
        C_recall = confusion_matrix[1] * 1.0 / (confusion_matrix[2] + 1e-6)
        C_f1 = 2 * C_acc * C_recall/(C_acc+C_recall+1e-6)         

        pbar.set_postfix(Train_Loss=train_loss.avg, F1=C_f1, Acc=C_acc, Rec=C_recall, Pred=confusion_matrix[1] + confusion_matrix[0], GT=confusion_matrix[2]) 
        pbar.close()
        if C_f1 > best_f1:
            best_f1 = C_f1
            state = {'epoch': epoch + 1,
                    "best_f1": best_f1,
                    "best_recall": best_recall,
                    'model_state': model.state_dict()}
                    # 'optimizer_state': optimizer.state_dict()}
            f1_4 = '%.4f'%C_f1
            acc_4 = '%.4f'%C_acc
            recall_4 = '%.4f'%C_recall
            torch.save(state, "{}ckpt/{}_f1_{}_rec_{}_acc_{}.pkl".format(args.data_path, args.model, f1_4, recall_4, acc_4))
            if args.finetune:
                torch.save(state, "{}ckpt/{}_finetune_best.pkl".format(args.data_path,args.model))
            else:
                torch.save(state, "{}ckpt/{}_scrach_best.pkl".format(args.data_path,args.model))
            no_optim = 0
        else:
            no_optim = no_optim + 1
        
        if no_optim == 3:
            optimizer.lr = args.l_rate / 2
            args.l_rate = args.l_rate / 2

        # if no_optim == 6:
        #     print("Early stop")
        #     os._exit(0)
            
        # return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument("--seed", default=2, type=int)
    parser.add_argument("--finetune", type=bool, default=False)
    parser.add_argument('--model', nargs='?', type=str, default='hrnet',
                        help='Dataset to use [\'cityscapes, mvd etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=512,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=512,
                        help='Height of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=20,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=8,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-4,
                        help='Learning Rate')
    parser.add_argument('--fold', nargs='?', type=int, default=0,
                        help='Learning Rate')
    parser.add_argument('--gpu', nargs='?', type=str, default='5',
                        help='Learning Rate')
    parser.add_argument('--width', nargs='?', type=str, default='w48',
                        help='imagenet pretrain or not')
    parser.add_argument('--data_path', nargs='?', type=str, default='../data/',
                        help='data path')
    # parser.add_argument('--resume', nargs='?', type=str, default= "/home/dlh/data/xview/ckpt/hrnet_all_w48_0_f1_0.5674_rec_0.6927_acc_0.4804.pkl",
    #                     help='Path to previous saved model to restart from')

    train_args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = train_args.gpu
    train_args.model = "hrnet_valid_only_{}_{}".format(train_args.width, train_args.fold)
    train(train_args)
