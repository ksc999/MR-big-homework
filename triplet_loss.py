import torch
import torch.nn as nn
import torch.nn.functional as F

class MyTripletLoss(nn.Module):

    def __init__(self, margin=0.35):
        # joint to combine CrossEntropyLoss and TripletLoss
        super(MyTripletLoss, self).__init__()
        # self.joint = nn.Parameter(torch.Tensor(1))  
        self.margin = margin
        self.reLU = nn.ReLU(inplace=True)
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
    
    def forward(self, input, label):
        input_normalized = F.normalize(input, p=2, dim=1)
        N = input.size(0)
        # for a single x, compute x*x
        distance = torch.pow(input_normalized, 2).sum(dim=1, keepdim=True).expand(N, N) 
        # compute all pair's distance in the mini batch   
        distance = distance + distance.t()
        distance = torch.addmm(distance, input, input.t(), beta=1, alpha=-2)
        # generate mask to denote whether two points' label are equal
        mask = label.expand(N, N).eq(label.expand(N, N).t())
        # define pos, neg, anchor, where anchor is incorporated implicitly
        for i in range(N):
            pos = distance[i][mask[i]].max().unsqueeze(0)
            if i == 0:
                positive_part = pos
            else:
                positive_part = torch.cat((positive_part, pos))
            neg = distance[i][mask[i]==0].min().unsqueeze(0)
            if i == 0:
                negative_part = neg
            else:
                negative_part = torch.cat((negative_part, neg))
        # generate TripletLoss
        losses = self.reLU(positive_part - negative_part + self.margin)
        my_triplet_loss = losses.mean()
        # add CrossEntropyLoss
        cross_entropy_loss = self.CrossEntropyLoss(input, label) 
        my_loss = my_triplet_loss + cross_entropy_loss
        return my_loss



        
