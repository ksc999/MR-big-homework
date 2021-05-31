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
# combine simple version of train and test
def simple_train_and_test(epochs, model, creiteron, train_loader, test_loader,
                        optimizer, scheduler, p_noise=0):
    # train mode
    model.train()
    for epoch in range(epochs):
        for data, label in train_loader:
            data = data.cuda()
            for i in range(len(label)): # if randn_noise = 1, inject noise
                rand_noise = np.random.binomial(1, p_noise)   
                if rand_noise:
                    label[i] = random.randint(0, 34)
            label = label.cuda()
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
    return accuracy

# add noise from the data channel, and show the test accuracies
# this process is harder to deal with, compared to the noise adding from source
import random
def add_noise(config, train_loader,test_loader, creiteron, p_noise_list):
    acc_list = []
    for p in p_noise_list:
        # initialize model, optimizer, scheduler
        model = base_model(class_num=config.class_num)
        model = model.cuda()
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=0.1, last_epoch=-1)
        accuracy = simple_train_and_test(config.epochs_before_clean+config.epochs_after_clean, 
                                        model, creiteron, train_loader, test_loader, optimizer, scheduler, p_noise=p)
        acc_list.append(accuracy.cpu())
        print("acc without data clean:", accuracy.item())
        return accuracy.cpu().item()
    # plt.plot(p_noise_list, acc_list)
    # plt.xlabel('p_noise')
    # plt.ylabel('test_accuracy')
    # plt.show()
#################################

#################################
# clean data to tackle the noise from the train set
def clean_data(config, train_loader, test_loader, creiteron):
    model = base_model(class_num=config.class_num)
    model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=0.1, last_epoch=-1)
    # preprocess
    _ = simple_train_and_test(config.epochs_before_clean, model, creiteron, train_loader, test_loader, optimizer, scheduler)
    # train mode
    model.train()
    threhold = 0.05
    for epoch in range(config.epochs_after_clean):
        # dynamic threhold of data cleaning
        if epoch % 5 == 0:
            threhold = threhold + 0.01
        for data, label in train_loader:
            data = data.cuda()
            label = label.cuda()
            output, _ = model(data)
            # use soft max to get propbbility of each label
            output_softmax = torch.softmax(output, dim=1)
            # use torch.sort to get the max and second max term of soft max
            output_softmax_sorted, _ = torch.sort(output_softmax, dim=1, descending=True)
            output_softmax_sorted_and_selected = output_softmax_sorted[:, :2]
            mask = torch.ones_like(label).bool().cuda()
            # if the diff between max and second max is smaller then the dynamic threhold,
            # we see this feature as tarnished one
            for i in range(len(mask)):
                if (output_softmax_sorted_and_selected[i][0] - output_softmax_sorted_and_selected[i][1]) < threhold:
                    mask[i] = False
            label = label[mask]
            # just ensure number of remaining ones won't be so that that cannot even do the mini-batch training
            if len(label) < 2:
                continue
            data = data[mask]
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
    return accuracy
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

    val_dataset = tiny_caltech35(transform=transform_test, used_data=['val'])
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)
    
    test_dataset = tiny_caltech35(transform=transform_test, used_data=['test'])
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)
    
    creiteron = torch.nn.CrossEntropyLoss()

    p_noise_list = [0]
    p_noise_add_to_source = [0.2, 0.3, 0.4, 0.5, 0.6]
    not_clean_acc = []
    clean_acc = []

    for p in p_noise_add_to_source:
        print('p:', p)
        train_dataset = tiny_caltech35(transform=transform_train, used_data=['train'])
        # add noise from the source
        for i in range(len(train_dataset.annotions)):
            noise_rate = np.random.binomial(1, p)
            if(noise_rate):
                train_dataset.annotions[i] = random.randint(0, 34)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
        # use add_noise to get test_acc without cleaning
        # just set p_noise_list as [0]
        acc_1 = add_noise(config, train_loader, test_loader, creiteron, p_noise_list)
        # use clean data to get test_acc with cleaning
        acc_2 = clean_data(config, train_loader, test_loader, creiteron)
        print("acc after clean: ", acc_2.cpu().item())
        not_clean_acc.append(acc_1)
        clean_acc.append(acc_2.cpu().item())

    # visulize of the comparison between clean and not clean
    fig, ax = plt.subplots()
    ax.plot(p_noise_add_to_source, not_clean_acc)
    ax.plot(p_noise_add_to_source, clean_acc)
    ax.legend(['acc without cleaning', 'acc with cleaning'])
    ax.set_xlabel('probability to add noise to source')
    ax.set_ylabel('accuracy')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, nargs='+', default=[112, 112])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--class_num', type=int, default=35)
    parser.add_argument('--learning_rate', type=float, default=0.02)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--epochs_before_clean', type=int, default=20)
    parser.add_argument('--epochs_after_clean', type=int, default=40)
    parser.add_argument('--milestones', type=int, nargs='+', default=[40, 50])

    config = parser.parse_args()
    main(config)


