import torch
from torch.utils.data import Dataset

Tensor = torch.FloatTensor

class ToyDataset(Dataset):
    def __init__(self, x_data, y_data):
        super(ToyDataset, self).__init__()

        self.x = torch.Tensor(x_data)
        self.y = torch.Tensor(y_data)      
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)