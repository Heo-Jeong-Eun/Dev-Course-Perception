import os, sys
import numpy as np 

import torch
import torch.nn as nn

from utils.tools import *

def make_conv_layer(layer_idx : int, modules : nn.Module, layer_info : dict, in_channel : int):
    filters = int(layer_info['filters']) # output channel size
    size = int(layer_info['size']) # kernel size
    stride = int(layer_info['stride']) # stride 
    pad = (size - 1) // 2
    modules.add_module('layer_' + str(layer_idx) + '_conv',
                       nn.Conv2d(in_channel, filters, size, stride, pad))

    # module에 add module을 통해 convalution layer에 들어가는 batch norm과 activation을 넣어준다. 
    # batch norm과 activation 값이 포함된 module이기 때문
    if layer_info['batch_normalize'] == '1':
        modules.add_module('layer_' + str(layer_idx) + '_bn',
                           nn.BatchNorm2d(filters))     
    if layer_info['activation'] == 'leaky':
        modules.add_module('layer_' + str(layer_idx) + '_act',
                           nn.LeakyReLU())
    elif layer_info['activation'] == 'relu':
        modules.add_module('layer_' + str(layer_idx) + '_act',
                           nn.ReLU())

def make_shortcut_layer(layer_idx : int, modules : nn.Module):
    modules.add_module('layer_' + str(layer_idx) + '_shortcut', nn.Sequential())

def make_route_layer(layer_idx : int, modules : nn.Module):
    modules.add_module('layer_' + str(layer_idx) + '_route', nn.Sequential())

def make_upsample_layer(layer_idx : int, modules : nn.Module, layer_info : dict):
    stride = int(layer_info['stride'])
    modules.add_module('layer_' + str(layer_idx) + '_upsample', 
                       nn.Upsample(scale_factor = stride, mode = 'nearest'))

class Yololayer(nn.Module):
    def __init__(self, layer_info : dict, in_width : int, in_height : int, is_train : bool):
        super(Yololayer, self).__init__()
        self.n_classes = int(layer_info.get['classes'])
        self.ignore_thresh = float(layer_info['ignore_thresh'])
        self.box_attr = self.n_classes + 5 # box[4] + objectness[1] + class_prob[n_classes]
        # 9개의 anchor에서 어떤 mask index를 사용할 것인가 
        # 각각의 layer에서 다 사용하는 것이 아니고, mask index의 값만큼 사용한다. 
        mask_idxes = [int(x) for x in layer_info['mask'].split(',')]
        anchor_all = [int(x) for x in layer_info['anchors'].split(',')]
        anchor_all = [(anchor_all[i], anchor_all[i + 1]) for i in range(0, len(anchor_all, 2))]
        self.anchor = torch.tensor(anchor_all[x] for x in mask_idxes)
        self.in_width = in_width
        self.in_height = in_height
        # 각각 yolo layer마다 stride, lw, lh 값이 달라진다. 
        self.stride = None
        self.lw = None
        self.lh = None
        self.is_train = is_train

    def forward(self, x):
        # x = input = [N, C, H, W]
        self.lw, self.lh = x.shape[3], x.shape[2]
        self.anchor = self.anchor.to(x.device)
        # stride 값 추출 
        # 1. self.stride = torch.tensor(self.in_width // self.lw)
        # 2. 
        self.stride = torch.tensor([torch.div(self.in_width, self.lw, rounding_mode = 'floor'),
                                    torch.div(self.in_height, self.lh, rounding_mode = 'floor')]).to(x.device)


# 일반적으로 model을 만들 때, nn.Moudule을 상속받아 사용하게 된다. 
class DarkNet53(nn.Module):
    def __init__(self, cfg, param):
        super().__init__()
        self.batch = int(param['batch'])
        self.in_channels = int(param['in_channels'])
        self.in_width = int(param['in_width'])
        self.in_height = int(param['in_height'])
        self.n_classes = int(param['classes'])
        self.module_cfg = parse_model_config(cfg)
        self.module_list = self.set_layer(self.module_cfg)

    def set_layer(self, layer_info):
        module_list = nn.ModuleList()
        in_channels = [self.in_channels] # first channels of input

        for layer_idx, info in enumerate(layer_info):
            modules = nn.Sequential()
            
            if info['type'] == 'convolutional':
                make_conv_layer(layer_idx, modules, info)
                in_channels.append(int(info['filters']))
            # add 하는 부분
            elif info['type'] == 'shortcut':
                make_shortcut_layer(layer_idx, modules)
                in_channels.append(in_channels[-1])
            # concat 하는 부분
            elif info['type'] == 'route':
                make_route_layer(layer_idx, modules)
                layers = [int(y) for y in info['layers'].split(',')]

                if len(layers) == 1:
                    in_channels.append(in_channels[layers[0]])
                # 두 개 layer의 feture을 concat하기 위해 기준은 height와, width가 같아야 한다.  
                # 단 channel 수 는 달라도 된다. concat하게 되면 두 layer의 channel 수를 합친만큼 channel 수를 갖는다. 
                elif len(layers) == 2:
                    in_channels.append(in_channels[layers[0]] + in_channels[layers[1]])
                elif info['type'] == 'upsample':
                    make_upsample_layer(layer_idx, modules, info)
                    in_channels.append(in_channels[-1])
                elif info['type'] == 'yolo':
