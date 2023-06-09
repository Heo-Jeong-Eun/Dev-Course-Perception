import os, sys

import torch
import torch.optim as optim

from utils.tools import *
from train.loss import *

class Trainer:
    def __init__(self, model, train_loader, eval_loader, hparam, device, torch_writer):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.max_batch = hparam['max_batch']
        self.device = device
        self.epoch = 0
        self.iter = 0
        self.yololoss = Yololoss(self.device, self.model.n_classes)
        self.optimizer = optim.SGD(model.parameters(), lr = hparam['lr'], momentum = hparam['momentum'])
        self.scheduler_multistep = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                             milestones = [20, 40, 60],
                                                             gamma = 0.5)
        self.torch_writer = torch_writer
    
    def run_iter(self):
        for i, batch in enumerate(self.train_loader):
            # drop batch when invalid values
            if batch is None:
                continue
            
            imput_image, targets, anno_path = batch 
            imput_image = imput_image.to(self.device, non_blocking = True)

            # forward 시 yolo result를 얻게 되고, 3개의 layer 결과가 출력된다. 
            output = self.model(imput_image)
            loss, loss_list = self.yololoss.compute_loss(output, targets, self.model.yolo_layers)

            # get gradients
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler_multistep.step(self.iter)
            self.iter += 1

            # [loss.item(), lobj.item(), lcls.item(), lbox.item()]
            loss_name = ['total_loss', 'obj_loss', 'cls_loss', 'box_loss']

            if i % 10 == 0:
                print('epoch : {} / iter : {} / lr : {} / loss : {}'.format(self.epoch, self.iter, get_lr(self.optimizer), loss.item()))
                self.torch_writer.add_scalar('lr', get_lr(self.optimizer), self.iter)
                self.torch_writer.add_scalar('total_loss', loss, self.iter)

                for ln, lv in zip(loss_name, loss_list):
                    self.torch_writer.add_scalar(ln, lv, self.iter)
            
        return loss

    def run(self):
        while True:
            self.model.train()

            # loss calculation
            loss = self.run_iter()
            self.epoch += 1

            # save model == check point
            checkpoint_path = os.path.join('./output', 'model_epoch' + str(self.epoch) + '.pth')
            torch.save({'epoch' : self.epoch,
                        'iteration' : self.iter,
                        'model_state_dict' : self.model.state_dict(),
                        'opitmizer_state_dict' : self.optimizer.state_dict(),
                        'loss' : loss}, checkpoint_path)

            # evaluation 
