import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import cv2
from network.models import model_selection
from dataset.transform import xception_default_data_transforms,resnet18_transforms
from dataset.mydataset import MyDataset
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as cal_auc
import numpy as np

torch.cuda.set_device(1) # 指定GPU  amax中需要，集群中不需要

def main():
    args = parse.parse_args()
    test_list = args.test_list
    batch_size = args.batch_size
    model_path = args.model_path

    print("test_list: ",test_list)
    print("batch_size: ",batch_size)
    print("model_path: ",model_path)

    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    test_dataset = MyDataset(txt_path=test_list, transform=resnet18_transforms['test'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                              num_workers=4)
    test_dataset_size = len(test_dataset)
    print("test_dataset_size: ",test_dataset_size)
    corrects = 0
    acc = 0

    model = model_selection(modelname='resnet18', num_out_classes=7, dropout=0)

    model.load_state_dict(torch.load(model_path))

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    model = model.cuda()
    model.eval()
    with torch.no_grad():


        for (image, labels) in test_loader:
            image = image.cuda()
            labels = labels.cuda()
            outputs = model(image)

            predicted = torch.argmax(outputs.data, dim=1)

            corrects += torch.sum(predicted == labels.data).to(torch.float32)

            print('Iteration Acc {:.4f}'.format(torch.sum(predicted == labels.data).to(torch.float32) / batch_size))
        acc = corrects / test_dataset_size
        print('Test Acc: {:.4f}'.format(acc))



if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--batch_size', '-bz', type=int, default=32)
    parse.add_argument('--test_list', '-tl', type=str, default='./data_list/jaffe_test.txt')
    parse.add_argument('--model_path', '-mp', type=str, default="./output/fer/best.pkl")
    main()
    print('Hello world!!!')
