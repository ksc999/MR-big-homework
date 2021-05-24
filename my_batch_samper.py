import torch
from torch.utils.data import Sampler, RandomSampler, SubsetRandomSampler

class MyBatchSampler(Sampler):
    def __init__(self, all_label, batch_size=64, species_num=16):
        self.all_label = all_label
        self.batch_size = batch_size
        self.species_num = species_num
        self.bootstrap = 2
    
    def __iter__(self):
        num_per_kind = self.batch_size // self.species_num
        for _ in range(self.bootstrap * len(self.all_label) // self.batch_size):          
            species_list = list(torch.randperm(35))[:self.species_num]
            batch = []
            for kind in species_list:
                mask = self.all_label.eq(kind)
                indice = torch.arange(len(self.all_label))
                pure_label_indice = list(indice[mask])
                # print(pure_label_indice)
                lucky_dog = list(SubsetRandomSampler(pure_label_indice))[:num_per_kind]
                batch = batch + lucky_dog
            # print(batch)
            yield batch
    
    def __len__(self):
        return self.bootstrap * len(self.all_label) // self.batch_size