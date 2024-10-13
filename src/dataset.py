import torch
import torch.nn as nn
from torch.utils.data import (
    Dataset, 
    DataLoader
)
from utils import LoadData

class DatasetForCPE(Dataset):
    
    def __init__(self, data): self.data = data
    
    def __len__(self): return len(self.data)
    
    def __getitem__(self, index): return self.data[index]

class DataModule(nn.Module):
    
    def __init__(self, args):
        
        super().__init__()
        
        self.args = args
        self.num_workers = args.num_workers
        train_data = LoadData(args, 'train').load()
        valid_data = LoadData(args, 'valid').load()
        test_data = LoadData(args, 'test').load()
        self.train_dataset = DatasetForCPE(train_data)
        self.valid_dataset = DatasetForCPE(valid_data)
        self.test_dataset = DatasetForCPE(test_data)
        self.train_dataset_size = args.sampling_num['train']
        self.batch_size = args.batch_size

    def forward(self):
        
        train_dl = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=False,
            pin_memory=True
        )
        val_dl = DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=False,
            pin_memory=True
        )
        test_dl = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=False,
            pin_memory=True
        )
        return train_dl, val_dl, test_dl
