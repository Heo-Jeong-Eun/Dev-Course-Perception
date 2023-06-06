import torch
import torchvision
import os, sys
import numpy as np

from torch.utils.data import Dataset
from PIL import Image

class Yolodata(Dataset):
    # formal path
    file_dir = ''
    anno_dir = ''
    file_txt = ''

    # train dataset path
    train_dir = '/Users/1001l1000/Documents/Dev-Course/12-2./KITTI/training'
    train_txt = 'train.txt'

    # eval dataset path 
    valid_dir = '/Users/1001l1000/Documents/Dev-Course/12-2./KITTI/eval'
    valid_txt = 'eval.txt'

    class_str = ['Car', 'Num', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
    num_class = None
    img_data = []

    def __init__(self, is_train = True, transform = None, cfg_param = None):
        super(Yolodata, self).__init__()
        self.is_train = is_train
        self.transform = transform
        self.num_class = cfg_param['classes']

        if self.is_train:
            self.file_dir = self.train_dir + '//JPEG images//'
            self.anno_dir = self.train_dir + '//annotations//'
            self.file_txt = self.train_dir + '//image sets//' + self.train_txt
        else:
            self.file_dir = self.valid_dir + '//JPEG images//'
            self.anno_dir = self.valid_dir + '//annotations//'
            self.file_txt = self.valid_dir + '//image sets//' + self.valid_txt

        img_names = []
        img_data = []

        with open(self.file_txt, 'r', encoding = 'UTF-8', errors = 'ignore') as f:
            # enter -> 공백 변환 
            img_names = [i.replace('\n', '') for i in f.readlines()]

        for i in img_names:
            # image set path를 출력
            print(i)

            # path를 불러온 후 == file name만 가지고 있는 상태이므로 확장자를 더해준다. 
            if os.path.exists(self.file_dir + i + '.jpg'):
                img_data.append(i + '.jpg')
            elif os.path.exists(self.file_dir + i + '.JPG'):
                img_data.append(i + '.JPG')
            elif os.path.exists(self.file_dir + i + '.png'):
                img_data.append(i + '.png')
            elif os.path.exists(self.file_dir + i + '.PNG'):
                img_data.append(i + '.PNG')

        self.img_data = img_data
        # print('data len : {}'.format(len(self.img_data)))

        # print(self.file_dir)
        # print(self.anno_dir)
        # print(self.file_txt)

    # file path를 모아둔 imgae data에서 실제 image file을 읽어오고, txt file도 읽어 들여서 data를 가지고 
    # 학습 input을 넣어줄 수 있게 하는 데이터 전처리를 수행한다. 
    # data loader가 학습할 때마다 get item 함수를 불러온다. 
    # get item per one element in one batch 
    def __getitem__(self, index):
        img_path = self.file_dir + self.img_data[index]

        with open(img_path, 'rb') as f:
            img = np.array(Image.open(img_path).convert('RGB'), dtype = np.uint8)
            # image shape : [H, W, C]
            img_origin_h, img_origin_w = img.shape[:2] 

        if os.path.isdir(self.anno_dir):
            txt_name = self.img_data[index]

            # 확장자 -> txt 파일로 변환
            for ext in ['.png', '.jpg', '.PNG', '.JPG']:
                txt_name = txt_name.replace(ext, 'txt')
            
            anno_path = self.anno_dir + txt_name

            if not os.path.exists(anno_path):
                return 
            
            # 바운딩 박스 
            bbox = []

            with open(anno_path, 'r') as f:
                for line in f.readlines():
                    line = line.replace('\n', '')
                    gt_data = [l for l in line.split(' ')]

                    # skip when abnormal data
                    if len(gt_data) < 5:
                        continue 

                    cls, cx, cy, w, h = float(gt_data[0]), float(gt_data[1]), float(gt_data[2]), float(gt_data[3]), float(gt_data[4])

                    # class, center x, center y, width, height
                    bbox.append([cls, cx, cy, w, h])
            
            bbox = np.array(bbox)

            # skip empty target
            empty_target = False

            if bbox.shape[0] == 0:
                empty_target = True
                bbox = np.array([[0, 0, 0, 0, 0]])

            # data augmentation, 갖고 있는 data를 다양한 type으로 바꿔주어서 성능을 향상시키는 방법
            if self.transform is not None:
                img, bbox = self.transform((img, bbox))

            if not empty_target:
                batch_idx = torch.zeros((bbox.shape[0]))
                target_data = torch.cat((batch_idx.view(-1, 1), torch.tensor(bbox)), dim = 1)
            else:
                return 
            
            return img, target_data, anno_path

        # anno_data가 없는 경우 
        else:
            bbox = np.array([[0, 0, 0, 0, 0]])
            img, _ = self.transform((img, bbox))

            return img, None, None
        
    def __len__(self):
        return len(self.img_data)