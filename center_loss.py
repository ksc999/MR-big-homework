import torch
import torch.nn as nn
import torch.nn.functional as F

class CenterLoss(nn.Module):
    def __init__(self, lamda = 0.2):
        super(CenterLoss, self).__init__()
        self.lamda = lamda

    def _get_centers(self, input, label):   # get centers of each label
        label_num = []
        for i in range(35):
            mask_i = label.eq(i)
            input_i = input[mask_i]
            if input_i.size(0) > 0:
                if i == 0:
                    centers = input_i.mean(dim=0).unsqueeze(0).cuda()
                else:
                    centers = torch.cat((centers, input_i.mean(dim=0).unsqueeze(0).cuda())).cuda()
            else:
                if i == 0:
                    centers = torch.zeros((1, input.size(1)), requires_grad=True).cuda()
                else:
                    centers = torch.cat((centers, torch.zeros((1, input.size(1)), requires_grad=True).cuda())).cuda()
            label_num.append(input_i.size(0))
        label_num = torch.as_tensor(label_num).cuda()
        return centers, label_num
    
    def forward(self, input, label):
        normalized_inputs = F.normalize(input)
        centers, label_num = self._get_centers(normalized_inputs, label)
        extented_centers = centers[label]
        extended_label_num = label_num[label]
        difference = torch.pow(normalized_inputs - extented_centers, 2).sum(dim=1)
        center_loss = torch.div(difference, extended_label_num).sum()
        # print(self.lamda * center_loss)
        return self.lamda * center_loss