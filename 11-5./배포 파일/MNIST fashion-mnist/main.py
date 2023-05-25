import sys, os
import argparse
import torch
import torch.nn as nn
from torchvision.datasets import MNIST, FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim

from model.models import *
from loss.loss import *
from util.tools import *


def parse_args():
    parser = argparse.ArgumentParser(description="MNIST")
    parser.add_argument('--mode', dest='mode', help="train / eval / test",
                        default=None, type=str)
    parser.add_argument('--download', dest='download', help="download MNIST dataset",
                        default=False, type=bool)
    parser.add_argument('--output_dir', dest='output_dir', help="output directory",
                        default='./output', type=str)
    parser.add_argument('--checkpoint', dest='checkpoint', help="checkpoint trained model",
                        default=None, type=str)
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    args = parser.parse_args()
    return args

def get_data(name = 'MNIST'):
    my_transform = transforms.Compose([
        transforms.Resize([32,32]),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(1.0,))
    ])
    if name == 'MNIST':
        download_root = "./mnist_dataset"
        train_dataset = MNIST(root=download_root,
                              transform=my_transform,
                              train=True,
                              download=args.download)
        eval_dataset = MNIST(root=download_root,
                             transform=my_transform,
                             train=False,
                             download=args.download)
        test_dataset = MNIST(root=download_root,
                             transform=my_transform,
                             train=False,
                             download=args.download)
    elif name == 'FASHION_MNIST':
        download_root = "./fashionmnist_dataset"
        train_dataset = FashionMNIST(root=download_root,
                                    transform=my_transform,
                                    train=True,
                                    download=args.download)
        eval_dataset = FashionMNIST(root=download_root,
                                    transform=my_transform,
                                    train=False,
                                    download=args.download)
        test_dataset = FashionMNIST(root=download_root,
                                    transform=my_transform,
                                    train=False,
                                    download=args.download)        
    
    return train_dataset,eval_dataset,test_dataset

def main():
    print(torch.__version__)
    
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    
    if torch.cuda.is_available():
        print("gpu")
        device = torch.device("cuda")
    else:
        print("cpu")
        device = torch.device("cpu")
    
    #Get MNIST Dataset
    train_dataset, eval_dataset, test_dataset = get_data(name="FASHION_MNIST")
    
    #Make Dataloader
    train_loader = DataLoader(train_dataset,
                              batch_size=8,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=True,
                              shuffle=True)
    eval_loader = DataLoader(eval_dataset,
                             batch_size=1,
                             num_workers=0,
                             pin_memory=True,
                             drop_last=False,
                             shuffle=False)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             num_workers=0,
                             pin_memory=True,
                             drop_last=False,
                             shuffle=False)
    
    _model = get_model('lenet5')
    
    #LeNet5
    
    if args.mode == "train":
        model = _model(batch = 8, n_classes=10, in_channel=1, in_width=32, in_height=32, is_train=True)
        model.to(device)
        model.train()
        
        #optimizer & scheduler
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9) #weight_decay=0.1  momentum=0.9
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        criterion = get_criterion(crit='mnist', device=device)
        
        loss_list = []
        iter_list = []
        
        epoch = 4
        iter = 0
        for e in range(epoch):
            total_loss = 0
            for i, batch in enumerate(train_loader):
                img = batch[0]
                gt = batch[1]
                
                img = img.to(device)
                gt = gt.to(device)
                
                out = model(img)
                
                loss_val = criterion(out, gt)
                
                #backpropagation
                loss_val.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss_val.item()
                
                if iter % 100 == 0:
                    print("{} epoch {} iter loss : {}".format(e, iter, loss_val.item()))
                    loss_list += [loss_val.item()]
                    iter_list += [iter]
                iter += 1
            
            mean_loss = total_loss / i
            scheduler.step()
            
            print("->{} epoch mean loss : {}".format(e, mean_loss))
            torch.save(model.state_dict(), args.output_dir + "/model_epoch"+str(e)+".pt")
        
        plt.clf()
        plt.plot(iter_list, loss_list, 'b', label='loss')
        plt.savefig('loss.jpg')
        print("Train end")
    elif args.mode == "eval":
        model = _model(batch = 1, n_classes=10, in_channel=1, in_width=32, in_height=32)
        #load trained model
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval() #not train()
        
        acc = 0
        num_eval = 0
        
        for i, batch in enumerate(eval_loader):
            img = batch[0]
            gt = batch[1]
            
            img = img.to(device)
            
            #inference
            out = model(img)
            
            out = out.cpu()

            if out == gt:
                acc += 1
            num_eval += 1
            
        print("Evaluation score : {} / {}".format(acc, num_eval))
    elif args.mode == "test":
        model = _model(batch = 1, n_classes=10, in_channel=1, in_width=1, in_height=1)
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval() #not train()
        
        for i, batch in enumerate(test_loader):
            img = batch[0]
            img = img.to(device)
            
            #inference
            out = model(img)
            out = out.cpu()
            
            print(out)
            
            #show result
            show_img(img.cpu().numpy(), str(out.item()))

if __name__ == "__main__":
    args = parse_args()
    main()