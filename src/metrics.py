import torch
from torchmetrics.classification import (
    Accuracy,
    Recall,
    Precision,
    F1Score,
    MatthewsCorrCoef,
    BinaryAccuracy,
    BinaryRecall,
    BinaryPrecision,
    BinaryF1Score,
    BinaryMatthewsCorrCoef
)

class Metrics:

    def __init__(self, num_classes, device):
        
        self.num_classes = num_classes
        
        accuracy = Accuracy(
            task="multiclass", num_classes=num_classes).to(device) if num_classes > 2 else BinaryAccuracy().to(device)
        recall = Recall(
            task="multiclass", num_classes=num_classes, average='macro').to(device) if num_classes > 2 else BinaryRecall().to(device)
        precision = Precision(
            task="multiclass", num_classes=num_classes, average='macro').to(device) if num_classes > 2 else BinaryPrecision().to(device)
        f1 = F1Score(
            task="multiclass", num_classes=num_classes, average='macro').to(device) if num_classes > 2 else BinaryF1Score().to(device)
        mcc = MatthewsCorrCoef(
            task="multiclass", num_classes=num_classes).to(device) if num_classes > 2 else BinaryMatthewsCorrCoef().to(device)
        self.metrics_dict = {'acc': accuracy, 'recall': recall, 'precision': precision, 'f1': f1, 'mcc': mcc}
        
        self.device = device
        
    def update(self, pred, target):
        
        if self.num_classes > 2:
            for metric in self.metrics_dict.values():
                metric(pred, target.to(self.device))
        else:
            for metric in self.metrics_dict.values():
                metric(torch.argmax(pred, dim=1), target.to(self.device))
                
    def compute(self):
        
        return {name: metric.compute() for name, metric in self.metrics_dict.items()}
    
    def reset(self):
        
        for metric in self.metrics_dict.values():
            metric.reset()
