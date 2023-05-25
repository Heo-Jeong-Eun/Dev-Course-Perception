import torch 
import torch.nn as nn
from turtle import down

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description = "MNIST")
    parser.add_argument("--mode", dest = "mode", help = "train / eval / test", default = "None", type = str)
    parser.add_argument("--download", dest = "download", help = "download MNIST", default = "False", type = bool)
    parser.add_argument("--output_dir", dest = "output_dir", help = "output directory", default = "/Users/1001l1000/Documents/Dev-Course/11-4./MNIST//output", type = str)
    parser.add_argument("--checkpoint", dest = "checkpoint", help = "checkpoint trained model", default = "None", type = str)
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    else : args = parser.parse_args()
    
    return args

# torchvision -> https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html?highlight=mnist
def get_data():
    # image resize
    my_transform = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.0,))
    ])
    download_root = "/Users/1001l1000/Documents/Dev-Course/11-4./MNIST/mnist_dataset"
    train_dataset = MNIST(download_root, 
                          transform = my_transform,
                          train = True,
                          download = args.download)
    eval_dataset = MNIST(download_root, 
                          transform = my_transform,
                          train = False,
                          download = args.download)
    test_dataset = MNIST(download_root, 
                        transform = my_transform,
                        train = False,
                        download = args.download)

def main():
    print(torch.__version__)

    # cuda 지원 여부 확인, torch에 tensor data를 어느 저장 공간에 올릴지 정하는 과정이다. 
    if torch.cuda.is_available():
        print("gpu")
        device = torch.device("cuda")
    else:
        print("cpu")
        device = torch.device("cuda")
    
    # get MNIST data set
    train_dataset, eval_dataset, test_dataset = get_data()

if __name__ == '__main__':
    args = parse_args()
    main()