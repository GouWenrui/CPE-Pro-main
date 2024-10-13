import torch

class BaseModel(torch.nn.Module):
    '''
    BaseModel is the base class for all models.
    '''
    
    def __init__(self):
        super().__init__()
    
    def forward(self):

        return NotImplementedError
    
    def predict(self):

        return NotImplementedError
    
    def to_device(self):

        return NotImplementedError

class BaseModelOutput:
    '''
    BaseModelOutput contains the output of a BaseModel.
    '''
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.hidden_states = kwargs.get('hidden_states', None)
        self.logits = kwargs.get('logits', None)
        self.loss = kwargs.get('loss', None)
        self.fused_weight = kwargs.get('fused_weight', None)
