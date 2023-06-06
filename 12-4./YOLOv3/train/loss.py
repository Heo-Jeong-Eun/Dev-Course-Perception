import os, sys
import enum

import torch
import torch.nn as nn

from utils.tools import *

class Yololoss(nn.Module):
    def __init__ (self, device, num_class):
        super(Yololoss, self).__init__()
        self.device = device
        self.num_classes = num_class
        self.mseloss = nn.MSELoss().to(device)
        self.bceloss = nn.BCELoss().to(device)
        self.bcelogloss = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([1, 0], device = device)).to(device)

    def compute_loss(self, pred, targets, yolo_layer):
        # lcls = class loss, lbox = box loss, lobj = object loss
        lcls = torch.zeros(1, device = self.device)
        lbox = torch.zeros(1, device = self.device) 
        lobj = torch.zeros(1, device = self.device)

        # get positive targets
        tcls, tbox, tindices, tanchors = self.get_targets(pred, targets, yolo_layer)

        # 3 yolo layers 
        for pidx, pout in enumerate(pred):
            batch_id, anchor_id, gy, gx = tindices[pidx]
            tobj = torch.zeros(pout[..., 0], device = self.device)
            num_targets = batch_id.shape[0]

            if num_targets:
                # [batch_id, anchor_id, grid h, grid_w, box attributes]
                ps = pout[batch_id, anchor_id, gy, gx]
                pxy = torch.sigmoid(ps[..., 0:2])
                pwh = torch.exp(ps[..., 2:4]) * tanchors[pidx]
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_iou(pbox.T, tbox[pidx], xyxy = False)

                # box loss
                # Mean Squared Loss
                # 1. 
                # mse loss를 사용하게 되면 value에 따라 loss가 반영되기 때문에 해당 부분이 커지는 경향이 있어 이 방법을 사용하지 않는 것이 좋다. 
                # loss_wh = self.mseloss(pbox[..., 2:4], tbox[pidx][..., 2:4])
                # loss_xy = self.mseloss(pbox[..., 0:2], tbox[pidx][..., 0:2])
                # 2. 
                # iou range의 평균값을 yolo layer에서 다 더하도록 한다. 
                # iou가 0일 때 loss는 1, iou가 1일 때 loss는 0 / 즉 box가 잘 겹쳐져있다면 bounding box loss = 0, 겹치지 않았다면 loss = 1
                # 겹치는 정도를 판단해 loss range 값이 0-1 사이값이 나오도록 한다. 
                lbox += (1 - iou).mean()

            # objectness
                # gt box와 pred box가 겹치는 경우 -> pos : 1 / 겹치지 않는 경우 -> neg : 0, using iou
                tobj[batch_id, anchor_id, gy, gx] = iou.detach().clamp(0).type(tobj.dtype)
            
                # object 배열이 xwyh, class, object 정보를 다 가지고 있는 정상적인 경우 
                if ps.size(1) - 5 > 1:
                    t = torch.zeros_like(ps[..., 5:], device = self.device)
                    t[range(num_targets), tcls[pidx]] = 1
                    lcls += self.bcelogloss(ps[:, 5:], t)

            lobj += self.bcelogloss(pout[..., 4], tobj)

        # loss weight
        lcls *= 0.05
        lobj *= 1.0
        lbox *= 0.5

        # total loss 
        loss = lcls + lbox + lobj

        # 시각화를 위한 loss list
        loss_list = [[loss.item(), lobj.item(), lcls.item(), lbox.item()]]

        return loss, loss_list

                # assignment -> iou 계산 
                # iou = bbox_iou(pbox, tbox[pidx])

            # print('yolo : {}, shape : {}'.format(pidx, pout.shape))

            # pout shape = [batch, anchors, grid y, grid x, box attributes]
            # the number of boxes in each yolo layer = anchors * grid height * grid width 
            # yolo 0 = 3 * 19 * 19, yolo 1 = 3 * 38 * 38, yolo 2 = 3 * 76 * 76
            # total boxes = 22743 
            # but, real gt objcet num = 10 ~ 20
            
            # positive prediction vs negative prediction
            # pos : neg = 0.01 : 0.99 
            # only in positive pred, we can get box loss and class loss
            # in negative pred, only object loss 

    def get_targets(self, preds, targets, yolo_layer):
        num_anc = 3
        num_targets = targets.shape[0]
        tcls, tboxes, indices, anch = [], [], [], []

        gain = torch.gain(7, device = self.device)
        ai = torch.arange(num_anc, device = targets.device).float().view(num_anc, 1).repeat(1, num_targets)

        # targets shape = [batch id, class id, box cx, box cy, box w, box h, anchor id]
        targets = torch.cat((targets.repeat(num_anc, 1, 1), ai[:, :, None]), 2)

        for yi, yl in enumerate(yolo_layer):
            # yolo layer의 scale에 맞게 각각 anchor를 만들어 준다. 
            anchors = yl.anchor / yl.stride 

            # grid w, grid h
            gain[2:6] = torch.tensor(preds[yi].shape)[[3, 2, 3, 2]]

            # same dim 
            t = targets * gain 

            if num_targets:
                r = t[:, :, 4:6] / anchors[:, None]

                # select ratios less than 4
                j = torch.max(r, 1. / r).max(2)[0] < 4
                t = t[j]
            else:
                t = targets[0]
            
            b, c = t[:, :2].long().T
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            gij = gxy.long()
            gi, gj = gij.T

            # anchor index
            a = t[:, 6].long()

            # add indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))

            # add target box
            tboxes.append(torch.cat((gxy - gij, gwh), dim = 1))

            # add anchor
            anch.append(anchors[a])

            # add class
            tcls.append(c)

        return tcls, tboxes, indices, anchors