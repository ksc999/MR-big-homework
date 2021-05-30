from dataset import tiny_caltech35
import torchvision.transforms as transforms
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from model import base_model

from sklearn import manifold,datasets
import matplotlib 
import matplotlib.pyplot as plt

#################################
# purely add noise, and do not fix it
import random
def add_noise(config, train_loader,test_loader, creiteron, p_noise_list):
    acc_list = []
    for p in p_noise_list:
        # initialize model, optimizer, scheduler
        model = base_model(class_num=config.class_num)
        model = model.cuda()
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=0.1, last_epoch=-1)
        # train mode
        model.train()
        for epoch in range(config.epochs):
            for data, label in train_loader:
                data = data.cuda()
                # print(label)
                for i in range(len(label)): # if randn_noise = 1, inject noise
                    rand_noise = np.random.binomial(1, p)   
                    if rand_noise:
                        label[i] = random.randint(0, 34)
                label = label.cuda()
                # print(label)
                output, _ = model(data)
                loss = creiteron(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
        # test mode
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, label in test_loader:
                data = data.cuda()
                label = label.cuda()
                output, _ = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == label).sum()
        accuracy = correct * 1.0 / len(test_loader.dataset)
        # accuracy.cpu()
        acc_list.append(accuracy.cpu())
    plt.plot(p_noise_list, acc_list)
    plt.xlabel('p_noise')
    plt.ylabel('test_accuracy')
    plt.show()
#################################

        

#################################
# use TSNE to draw scatter plot
def draw_dataset_with_TSNE(feature, label):
    feature_numpy = feature.cpu().detach().numpy()
    label_numpy = label.cpu().detach().numpy()

    f = np.array([feature_numpy[i] for i in range(label_numpy.shape[0]) if (label_numpy[i]>=1 and label_numpy[i]<=5)])
    l = np.array([label_numpy[i] for i in range(label_numpy.shape[0]) if (label_numpy[i]>=1 and label_numpy[i]<=5)])
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    feature_tsne = tsne.fit_transform(f)
    x = feature_tsne[:, 0]
    y = feature_tsne[:, 1]
    plt.scatter(x, y, c=l, alpha=0.5)
    plt.show()

#################################

def main(config):


    transform_train = transforms.Compose([
        transforms.Resize(config.image_size, interpolation=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize(config.image_size, interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = tiny_caltech35(transform=transform_train, used_data=['train'])

    # if you want to add the addition set and validation set to train
    # train_dataset = tiny_caltech35(transform=transform_train, used_data=['train', 'val', 'addition'])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    val_dataset = tiny_caltech35(transform=transform_test, used_data=['val'])
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)

    test_dataset = tiny_caltech35(transform=transform_test, used_data=['test'])
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)


    creiteron = torch.nn.CrossEntropyLoss()
    p_noise_list = np.linspace(0, 0.8, 10)
    add_noise(config, train_loader, test_loader, creiteron, p_noise_list)

###############################
# loss and acc curves
    # fig = plt.figure(figsize=(6,4)) 
    # ax1 = fig.add_subplot(111)
    # ax1.plot(train_numbers, train_losses, 'r-', label='loss')
    # ax1.set_xlabel('train_numbers')
    # ax1.set_ylabel('train_losses')
    # ax1.legend(loc='upper left')
    # ax2 = ax1.twinx()
    # ax2.plot(train_numbers, train_accuracies, 'b-', label='acc')
    # ax2.set_ylabel('train_accuracies')
    # ax2.legend(loc='upper right')
    # plt.show()
###############################

    # you can use validation dataset to adjust hyper-parameters
    # val_accuracy = test(val_loader, model)
    # test_accuracy = test(test_loader, model)
    # print('===========================')
    # print("val accuracy:{}%".format(val_accuracy * 100))
    # print("test accuracy:{}%".format(test_accuracy * 100))


def train(config, data_loader, model, optimizer, scheduler, creiteron):
    model.train()
    train_losses = []
###########################
    train_accuracies = []
###########################
    train_numbers = []
    counter = 0
###########################
    draw_feature = list()  # list to store feature from the last epoch
    draw_label = list()     # list to store label from the last epoch
###########################
    for epoch in range(config.epochs):
        for batch_idx, (data, label) in enumerate(data_loader):
            data = data.cuda()
            label = label.cuda()
            output, feature = model(data)
            loss = creiteron(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            counter += data.shape[0]
            accuracy = (label == output.argmax(dim=1)).sum() * 1.0 / output.shape[0]
            if batch_idx % 20 == 0:
                print('Train Epoch: {} / {} [{}/{} ({:.0f}%)] Loss: {:.6f} Accuracy: {:.6f}'.format(
                    epoch, config.epochs, batch_idx * len(data), len(data_loader.dataset),
                                          100. * batch_idx / len(data_loader), loss.item(), accuracy.item()))
                train_losses.append(loss.item())
###########################
                train_accuracies.append(accuracy.item())
###########################
                train_numbers.append(counter)
##############################
# draw scatter plot using features from the last epoch
            if epoch == config.epochs - 1:
                draw_feature.append(feature)
                draw_label.append(label)
##############################
        scheduler.step()
        torch.save(model.state_dict(), './model.pth')
##############################
# convert and reshape features
    draw_feature = torch.stack(draw_feature)
    draw_feature = draw_feature.view(-1, draw_feature.size(-1))
    draw_label = torch.stack(draw_label)
    draw_label = draw_label.view(-1)
# draw scatter plot
    draw_dataset_with_TSNE(draw_feature, draw_label)
##############################
    return train_numbers, train_losses, train_accuracies


def test(data_loader, model):
    model.eval()
    correct = 0
###########################
    draw_feature_test = torch.tensor([])  # list to store feature from the last epoch in the test
    draw_label_test = torch.tensor([])     # list to store label from the last epoch in the test
###########################
    with torch.no_grad():
        index = 0
        for data, label in data_loader:
##############################
            data = data.cuda()
            label = label.cuda()
            output, feature = model(data)
            if index == 0:
                draw_feature_test = feature.clone().detach()
                draw_label_test = label.clone().detach()
            else:
                draw_feature_test = torch.cat((draw_feature_test, feature))
                draw_label_test = torch.cat((draw_label_test, label))
##############################
            pred = output.argmax(dim=1)
            correct += (pred == label).sum()
            index = index + 1
##############################
# draw scatter plot
        draw_dataset_with_TSNE(draw_feature_test, draw_label_test)
##############################
    accuracy = correct * 1.0 / len(data_loader.dataset)
    return accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, nargs='+', default=[112, 112])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--class_num', type=int, default=35)
    parser.add_argument('--learning_rate', type=float, default=0.02)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--milestones', type=int, nargs='+', default=[40, 50])

    config = parser.parse_args()
    main(config)


