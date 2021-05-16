import torch.nn as nn


class my_model_2(nn.Module):
    def __init__(self, class_num=35):
        super(my_model_2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, stride=1, dilation=2)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3, 3), padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(3, 3), padding=1, stride=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1, stride=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.relu = nn.ReLU(inplace=True)

        self.avg_pooling = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.drop_out = nn.Dropout(p=0.1)

        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.bn6 = nn.BatchNorm1d(256)
        self.bn7 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, class_num)

    def forward(self, x, need_drop=1):

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.avg_pooling(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.avg_pooling(x)

        res_1 = x
        x = self.relu(self.bn3(self.conv3(x)))
        x = res_1 + x 
        x = self.avg_pooling(x)

        x = self.relu(self.bn4(self.conv4(x)))
        x = self.avg_pooling(x)

        res_2 = x
        x = self.relu(self.bn5(self.conv5(x)))  
        x = res_2 + x

        x = self.GAP(x).squeeze(dim=3).squeeze(dim=2)
        if need_drop:
            x = self.drop_out(x)
        x = self.fc1(self.relu(self.bn6(x)))
        if need_drop:
            x = self.drop_out(x)
        x = self.fc2(self.relu(self.bn7(x)))
        
        return x