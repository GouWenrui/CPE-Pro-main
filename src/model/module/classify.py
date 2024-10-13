import torch.nn as nn

class ClassificationHead(nn.Module):

    def __init__(self,
                 in_dim, 
                 dropout, 
                 num_labels):
    
        super().__init__()
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(in_dim, in_dim // 2)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.final = nn.Linear(in_dim // 2, num_labels)
        
    def forward(self, x):
        
        x = x.permute(0, 2, 1)
        x = self.pool(x)
        x = x.squeeze(-1) 
        x = self.linear(x)
        x = self.dropout(x)
        x = self.relu(x)
        return self.final(x)
