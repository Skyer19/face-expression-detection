import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
from network.models import model_selection
from dataset.transform import xception_default_data_transforms,transforms
from dataset.mydataset import MyDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.set_device(1) # 指定GPU  amax中需要，集群中不需要



def main():

    args = parse.parse_args()
    name = args.name
    continue_train = args.continue_train
    train_list = args.train_list
    val_list = args.val_list
    epoches = args.epoches
    batch_size = args.batch_size
    model_name = args.model_name
    model_path = args.model_path
    output_path = os.path.join('./output', name)

    print("--------------------------")
    print("name: ", name)
    print("train_list: ", train_list)
    print("val_list: ", val_list)
    print("epoches: ", epoches)
    print("batch_size: ", batch_size)
    print("model_name: ", model_name)
    print("output_path: ", output_path)
    print("--------------------------")

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    torch.backends.cudnn.enabled = False

    train_dataset = MyDataset(txt_path=train_list, transform=transforms['train'])
    val_dataset = MyDataset(txt_path=val_list, transform=transforms['val'])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                               num_workers=2, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                             num_workers=2, pin_memory=True)

    train_dataset_size = len(train_dataset)
    val_dataset_size = len(val_dataset)


    # -------------initModel------------ #
    # modelname: Can be vgg19 resnet50 resnet18 .,etc
    model = model_selection(modelname='resnet50', num_out_classes=7, dropout=0)

    if continue_train:
        model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999),eps=1e-08)

    best_model_wts = model.state_dict()
    best_acc = 0.0
    iteration = 0

    for epoch in range(epoches):
        print('Epoch {}/{}'.format(epoch + 1, epoches))
        print('-' * 10)

        model.train()
        train_loss = 0.0
        train_corrects = 0.0
        val_loss = 0.0
        val_corrects = 0.0
        for (image, labels) in train_loader:

            iter_loss = 0.0
            iter_corrects = 0.0
            image = image.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(image)
            predicted = torch.argmax(outputs.data, dim=1)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            iter_loss = loss.data.item()
            train_loss += iter_loss

            iter_corrects = (predicted == labels).sum().item()
            train_corrects += iter_corrects

            iteration += 1
            if not (iteration % 10):
                print('iteration {} train loss: {:.4f} Acc: {:.4f}'.format(iteration, iter_loss / batch_size,
                                                                           iter_corrects / batch_size))
        epoch_loss = train_loss / train_dataset_size
        epoch_acc = train_corrects / train_dataset_size
        print('epoch train loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        model.eval()
        with torch.no_grad():
            for (image, labels) in val_loader:
                image = image.to(device)
                labels = labels.to(device)

                outputs = model(image)
                predicted = torch.argmax(outputs.data, dim=1)
                loss = criterion(outputs, labels)
                val_loss += loss.data.item()
                val_corrects += (predicted == labels).sum().item()

            epoch_loss = val_loss / val_dataset_size
            epoch_acc = val_corrects / val_dataset_size

            print('epoch val loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        # scheduler.step()
        optimizer.step()
        torch.save(model.state_dict(), os.path.join(output_path, str(epoch) + '_' + model_name))

    print('Best val Acc: {:.4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), os.path.join(output_path, "best.pkl"))



if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--name', '-n', type=str, default='fer')
    parse.add_argument('--train_list', '-tl', type=str, default='./data_list/jaffe_train.txt')
    parse.add_argument('--val_list', '-vl', type=str, default='./data_list/jaffe_test.txt')
    parse.add_argument('--batch_size', '-bz', type=int, default=32)
    parse.add_argument('--epoches', '-e', type=int, default='10')
    parse.add_argument('--model_name', '-mn', type=str, default='fer.pkl')
    parse.add_argument('--continue_train', type=bool, default=False)
    parse.add_argument('--model_path', '-mp', type=str, default='./output/fer/1_fer.pkl')
    main()
