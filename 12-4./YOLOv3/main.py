import torch

from torch.utils.data.dataloader import DataLoader

import os, sys
import argparse

from ast import parse

from utils.tools import *
from dataloader.yolodata import *
from dataloader.data_transform import *
from model.yolov3 import *
from train.trainer import * 

from tensorboardX import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser(description = 'YOLOv3 PyTorch Arguments')
    parser.add_argument('--gpys', type = int, nargs = '+', default = [], help = 'List of GPU Device ID')
    parser.add_argument('--mode', type = str, help = 'Mode : train / eval / demo', default = None)
    parser.add_argument('--cfg', type = str, help = 'Model Config Path', default = None)
    parser.add_argument('--checkpoint', type = str, help = 'Check Point Path', default = None)

    # 예외 처리 
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1) # 코드 종료 
    args = parser.parse_args()
    
    return args

def collate_fn(batch):
    batch = [data for data in batch if data is not None]

    # skip invalid data
    if len(batch) == 0:
        return 
    
    images, targets, anno_path = list(zip(*batch))
    images = torch.stack([image for image in images])

    for i, boxes in enumerate(targets):
        # insert index of batch 
        boxes[:, 0] = i
    targets = torch.cat(targets, 0)
    
    return images, targets, anno_path

    # print([images[0]].shape, images.shape)

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
                              shuffle = True,
                              collate_fn = collate_fn)
    
    model = DarkNet53(args.cfg, cfg_param, training = True)

    model.train()
    model.initialize_weights()

    # check 
    print('gpu : ', torch.cuda.is_available())
    
    # set device
    if torch.cuda.is_available():
        device = torch.device('cuda : 0')
    else:
        device = torch.device('cpu')
    model = model.to(device)

    torch_writer = SummaryWriter('./output')

    trainer = Trainer(model = model, train_loader = train_loader, eval_loader = None, hparam = cfg_param, device = device, torch_writer = torch_writer)
    trainer.run()

    # for name, param in model.parameters():
    #     print(f'name : {name}, shape : {param}')

    # # batch shape 출력 
    # for i, batch in enumerate(train_loader):
    #     img, targets, anno_path = batch
    #     print('iter {} img {}, targets {}, anno_path {}'.format(i, img.shape, targets.shape, anno_path))
        
    #     # print(i, len(batch))
    #     # print(batch)

    #     # image 출력
    #     drawBox(img[0].detach().cpu())

    # for i, batch in enumerate(train_loader):
    #     image, targets, anno_path = batch

    #     output = model(image)

    #     print('output len : {}, 0th shape : {}'.format(len(output), output[0].shape))
    #     sys.exit(1)
                              
# def eval(cfg_param = None, using_gpus = None):
#     print('eval')

# def demo(cfg_param = None, using_gpus = None):
#     print('demo')

# if __name__ == '__main__':
#     args = parse_args()

#     # cfgs path
#     net_data = parse_hyperparam_config(args.cfg)
#     cfg_param = get_hyperparam(net_data)
#     print(cfg_param)

#     if args.mode == 'train':
#         train(cfg_param = cfg_param) # training 
#     elif args.mode == 'eval':
#         eval(cfg_param = cfg_param) # testing
#     elif args.mode == 'demo':
#         demo(cfg_param = cfg_param) # demo
#     else:
#         print('unknown')  

#     print('finish')