from typing import Any
import torch 
from torchvision import transforms as tf

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

import cv2
import numpy as np

from utils.tools import *

def get_transformations(cfg_param = None, is_train = None):
    if is_train:
        data_transform = tf.Compose([AbsoluteLabels(),
                                     DefaultAug(),
                                     ResizeImage(new_size = (cfg_param['in_width'], cfg_param['in_height'])),
                                     RelativeLabel(),
                                     ToTensor()])
    else:
        data_transform = tf.Compose([AbsoluteLabels(),
                                     ResizeImage(new_size = (cfg_param['in_width'], cfg_param['in_height'])),
                                     RelativeLabel(),
                                     ToTensor()])

    return data_transform

# abs box, 절대값으로 되어 있어야 resize와 같은 변형을 해도 original image에 대한 정보를 잃지 않는다. 
class AbsoluteLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        image, label = data
        h, w, _ = image.shape
        # 1, 3은 cx, w / 2, 4는 cy, h
        label[:, [1, 3]] *= w 
        label[:, [2, 4]] *= h

        return image, label
    
# 절대값을 0-1로 normalize, loss function을 계산할 때 0-1 사이의 값을 가지는 값과 비교하기 위해
class RelativeLabel(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        image, label = data
        h, w, _ = image.shape
        # 1, 3은 cx, w / 2, 4는 cy, h
        label[:, [1, 3]] /= w
        label[:, [2, 4]] /= h

        return image, label
    
class ToTensor(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        image, label = data

        # normalize
        # HWC -> CHW
        image = torch.tensor(np.transpose(np.array(image, dtype = float) / 255, (2, 0, 1)), dtype = torch.float32)
        
        labels = torch.FloatTensor(np.array(label))
        
        return image, labels

class ResizeImage(object):
    def __init__(self, new_size, interpolation = cv2.INTER_LINEAR):
        self.new_size = tuple(new_size)
        self.interpolation = interpolation

    def __call__(self, data):
        image, label = data
        image = cv2.resize(image, self.new_size, interpolation = self.interpolation)


# augmentation code를 작성할 때 아래와 같은 template으로 진행
# 공통적으로 필요한 부분이므로 미리 함수 class 형식을 제작하는 것
class ImgAug(object):
    def __init__(self, augmentations = []):
        self.augmentations = augmentations

    def __call__(self, data):
        # unpack data
        image, labels = data

        # convert x, y, w, h to xyxy -> minx, miny, maxx, maxy
        boxes = np.array(labels)
        boxes[:, 1:] = xywh2xyxy_np(boxes[:, 1:])

        # convert bounding box to imageaug format
        bounding_boxes = BoundingBoxesOnImage(
            [BoundingBox(*box[1:], label = box[0]) for box in boxes],
            shape = image.shape
        )

        # apply augmentations
        image, bounding_boxes = self.augmentations(
            image = image,
            bounding_boxes = bounding_boxes
        )

        # 예외 처리, 이미지 밖으로 나가는 bounding box를 clip
        bounding_boxes = bounding_boxes.clip_out_od_image()

        # convert bounding boxes to np.array()
        boxes = np.zeros((len(bounding_boxes), 5))

        for box_idx, box in enumerate(bounding_boxes):
            x1 = box.x1
            y1 = box.y1
            x2 = box.x2
            y2 = box.y2

            # return [x, y, w, h]
            boxes[box_idx, 0] = boxes.label
            boxes[box_idx, 1] = (x1 + x2) / 2
            boxes[box_idx, 2] = (y1 + y2) / 2
            boxes[box_idx, 3] = x2 - x1
            boxes[box_idx, 4] = y2 - y1

        return image, boxes           

class DefaultAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.Sharpen(0.0, 0.1),
            iaa.Affine(rotate = (-0, 0),
            translate_percent = (-0.1, 0.1),
            scale = (0.8, 1.5))])

