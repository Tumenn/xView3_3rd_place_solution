import torch.nn.functional as F
import argparse
import torch
import math
import numpy as np
import os
import sys
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm
import random

from datasets.xview import Xview3_Class_Loader
from models.resnet_ibn_a import resnet101_ibn_a
from utils import AverageTracker

import warnings
warnings.filterwarnings('ignore')

def train(args):
    if not os.path.exists("{}/ckpt".format(args.data_path)):
        os.makedirs("{}/ckpt".format(args.data_path))    

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader = Xview3_Class_Loader(args.data_path, txt_split="class_merge_{}_crop_train".format(args.fold))
    valid_loader = Xview3_Class_Loader(args.data_path, txt_split="class_merge_{}_crop_train".format(args.fold))

    model = resnet101_ibn_a(pretrained=True, num_classes=1000)
    model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048, 3),
        )
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.l_rate)

    args.start_epoch = 0
    best_acc = 0
    no_optim = 0

    train_loader = data.DataLoader(train_loader, batch_size=args.batch_size, num_workers=32, shuffle=True)
    valid_loader = data.DataLoader(valid_loader, batch_size=args.batch_size, num_workers=32)
    num_batches = len(train_loader)
    curr_iter = 0
    for epoch in np.arange(args.start_epoch, args.n_epoch):
        pbar = tqdm(np.arange(num_batches),ncols=180)

        model.train()
        train_loss, train_vessel, train_fishing = AverageTracker(), AverageTracker(), AverageTracker()
        train_length = AverageTracker()
        for train_i, (images, is_vessel, is_fishing, len_vessel) in enumerate(train_loader):
            # optimizer.lr = args.l_rate * (1 - float(curr_iter) / (num_batches * args.n_epoch)) ** 0.9
            curr_iter += 1            
            pbar.update(1)
            pbar.set_description("> Epoch [%d/%d]" % (epoch + 1, args.n_epoch))

            images = Variable(images.cuda())
            is_vessel = Variable(is_vessel.cuda())
            is_fishing = Variable(is_fishing.cuda())
            len_vessel = Variable(len_vessel.cuda())

            optimizer.zero_grad()

            outputs = model(images)
            
            loss_vessel = F.binary_cross_entropy_with_logits(outputs[:,0], is_vessel, reduction="mean")
            if (is_fishing != 255).sum() == 0:
                loss_fishing = 0
            else:
                loss_fishing = F.binary_cross_entropy_with_logits(outputs[:,1][is_fishing != 255], is_fishing[is_fishing != 255], reduction="mean")
            if (len_vessel != -1).sum() == 0:
                loss_length = 0
            else:
                loss_length = F.l1_loss(outputs[:,2][len_vessel != -1], len_vessel[len_vessel != -1], reduction="none") / len_vessel[len_vessel != -1]
                loss_length = loss_length.mean()
            
            loss = loss_vessel + loss_fishing + loss_length
            # loss = loss_fn(outputs, is_vessel)
            loss.backward()
            optimizer.step()

            cur_acc_vessel = ((torch.sigmoid(outputs[:,0]) > 0.5) * 1 == is_vessel)[is_vessel != 255].sum() / ((is_vessel != 255).sum() + 1e-6)
            cur_acc_fishing = ((torch.sigmoid(outputs[:,1]) > 0.5) * 1 == is_fishing)[is_fishing != 255].sum() / ((is_fishing != 255).sum() + 1e-6)
            train_loss.update(loss.item())
            train_vessel.update(cur_acc_vessel.item())
            if loss_fishing != 0:
                train_fishing.update(cur_acc_fishing.item())
            if loss_length != 0:
                train_length.update(loss_length.item())
            pbar.set_postfix(T_Loss=train_loss.avg, V=train_vessel.avg, F=train_fishing.avg, L=train_length.avg)

        model.eval()
        valid_loss, valid_vessel, valid_fishing = AverageTracker(), AverageTracker(), AverageTracker()
        valid_length = AverageTracker()
        for i_val, (images_val, is_vessel, is_fishing, len_vessel) in enumerate(valid_loader):
            with torch.no_grad():
                images_val = Variable(images_val.cuda(), requires_grad=False)
                is_vessel = Variable(is_vessel.cuda())
                is_fishing = Variable(is_fishing.cuda())
                len_vessel = Variable(len_vessel.cuda())

                outputs = model(images_val)

                loss_vessel = F.binary_cross_entropy_with_logits(outputs[:,0], is_vessel, reduction="mean")
                if (is_fishing != 255).sum() == 0:
                    loss_fishing = 0
                else:
                    loss_fishing = F.binary_cross_entropy_with_logits(outputs[:,1][is_fishing != 255], is_fishing[is_fishing != 255], reduction="mean")
                if (len_vessel != -1).sum() == 0:
                    loss_length = 0
                else:
                    loss_length = F.l1_loss(outputs[:,2][len_vessel != -1], len_vessel[len_vessel != -1], reduction="none") / len_vessel[len_vessel != -1]
                    loss_length = loss_length.mean()
                
                loss = loss_vessel + loss_fishing + loss_length

                cur_acc_vessel = ((torch.sigmoid(outputs[:,0]) > 0.5) * 1 == is_vessel)[is_vessel != 255].sum() / ((is_vessel != 255).sum() + 1e-6)
                cur_acc_fishing = ((torch.sigmoid(outputs[:,1]) > 0.5) * 1 == is_fishing)[is_fishing != 255].sum() / ((is_fishing != 255).sum() + 1e-6)
                valid_loss.update(loss.item())
                valid_vessel.update(cur_acc_vessel.item())
                if loss_fishing != 0:
                    valid_fishing.update(cur_acc_fishing.item())
                if loss_length != 0:
                    valid_length.update(loss_length.item())
                pbar.set_postfix(T_Loss=valid_loss.avg, V=valid_vessel.avg, F=valid_fishing.avg, L=valid_length.avg)

        pbar.set_postfix(T_Loss=train_loss.avg, V_Loss=valid_loss.avg, 
                         T_V=train_vessel.avg, V_V=valid_vessel.avg,
                         T_F=train_fishing.avg, V_F=valid_fishing.avg, 
                         T_L=train_length.avg, V_L=valid_length.avg)

        if valid_vessel.avg + valid_fishing.avg + (1-valid_length.avg) >= best_acc:
            best_acc = valid_vessel.avg + valid_fishing.avg + (1-valid_length.avg)
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),}
                    #  'optimizer_state' : optimizer.state_dict(),}
            ves_4 = '%.4f'%valid_vessel.avg
            fis_4 = '%.4f'%valid_fishing.avg
            len_4 = '%.4f'%(1-valid_length.avg)
            torch.save(state, "{}ckpt/{}_ves_{}_fis_{}_len_{}.pkl".format(args.data_path,args.model,ves_4,fis_4,len_4))
            torch.save(state, "{}ckpt/{}_best.pkl".format(args.data_path,args.model))
            no_optim = 0
        else:
            no_optim = no_optim + 1
        
        if no_optim == 5:
            optimizer.lr = args.l_rate / 2
            args.l_rate = args.l_rate / 2

        if no_optim == 10:
            print("Early stop")
            os._exit(0)

        pbar.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument("--seed", default=2, type=int)
    parser.add_argument('--model', nargs='?', type=str, default='resnet101_ibna',
                        help='Dataset to use [\'cityscapes, mvd etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=128,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=128,
                        help='Height of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=100,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=128,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-2,
                        help='Learning Rate')
    parser.add_argument('--fold', nargs='?', type=int, default=0,
                        help='Learning Rate')
    parser.add_argument('--gpu', nargs='?', type=str, default='5',
                        help='Learning Rate')

    parser.add_argument('--data_path', nargs='?', type=str, default="../../data/",
                        help='data path')

    train_args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = train_args.gpu
    train_args.model = "res101_merge_new_{}".format(train_args.fold)
    train(train_args)

