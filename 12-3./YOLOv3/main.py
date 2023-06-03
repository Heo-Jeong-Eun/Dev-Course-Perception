import torch
import argparse
import os, sys

from ast import parse
from torch.utils.data.dataloader import DataLoader

from utils.tools import *
from dataloader.yolodata import *
from dataloader.data_transform import *
from model.yolov3 import *

def parse_args():
    parser = argparse.ArgumentParser(description = 'YOLOv3 PyTorch Arguments')
    parser.add_argument('--gpys', type = int, nargs = '+', default = [], help = 'List of GPU Device ID')
    parser.add_argument('--mode', type = str, help = 'Mode : train / eval / demo', default = None)
    parser.add_argument('--cfg', type = str, help = 'Model Config Path', default = None)
    parser.add_argument('--checkpoint', type = str, help = 'Check Point Path', default = None)

    # 예외 처리 
    if len(sys.argv) == 1:
        parser.print_help()
        # 코드 종료 
        sys.exit(1) 
    args = parser.parse_args()
    
    return args

def train(cfg_param = None, using_gpus = None):
    print('train')

    my_transform = get_transformations(cfg_param = cfg_param, is_train = True)

    # data loader 6081 images / batch = 4
    train_data = Yolodata(is_train = True, 
                        transform = my_transform, 
                        cfg_param = cfg_param)
    
    train_loader = DataLoader(train_data,
                              batch_size = cfg_param['batch'],
                              num_workers = 0, 
                              pin_memory = True,
                              drop_last = True,
                              shuffle = True)

    model = DarkNet53(args.cfg, cfg_param)
    
    # batch shape 출력 
    for i, batch in enumerate(train_loader):
        img, targets, anno_path = batch
        print('iter {} img {}, targets {}, anno_path {}'.format(i, img.shape, targets.shape, anno_path))
        
        # print(i, len(batch))
        # print(batch)

        # image 출력
        drawBox(img[0].detach().cpu())
                              
def eval(cfg_param = None, using_gpus = None):
    print('eval')

def demo(cfg_param = None, using_gpus = None):
    print('demo')

if __name__ == '__main__':
    args = parse_args()

    # cfgs path
    net_data = parse_hyperparam_config(args.cfg)
    cfg_param = get_hyperparam(net_data)
    print(cfg_param)

    if args.mode == 'train':
        # training 
        train(cfg_param = cfg_param)
    elif args.mode == 'eval':
        # testing
        eval(cfg_param = cfg_param)
    elif args.mode == 'demo':
        # demo
        demo(cfg_param = cfg_param)
    else:
        print('unknown')  

    print('finish')