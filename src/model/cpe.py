import torch
from torch.nn.functional import (
    binary_cross_entropy_with_logits, 
    cross_entropy,
    one_hot)

import sys
sys.path.append('../')
from model.base import BaseModel, BaseModelOutput
from model.module.gvp.encoder import GVPEncoder
from model.module.sslm.encoder import SSLMEncoder
from model.module.encoder import Intergrate_Encoder
from model.module.pooling import Attention1dPoolingHead
from model.module.classify import ClassificationHead

class CPEPro(BaseModel):
    
    def __init__(self, args):
    
        super().__init__()
        
        self.args = args
        
        if args.use_sslm and args.use_gvp:
            self.encoder = Intergrate_Encoder(args)
        elif args.use_gvp:
            self.encoder = GVPEncoder(args)
        elif args.use_sslm:
            self.encoder = SSLMEncoder(args)
        
        assert args.num_classes > 0, "num_classes should be greater than 0"
        
        if args.atten_pooling:
            self.classification_head = Attention1dPoolingHead(
                hidden_size=self.encoder.output_dim,
                num_labels=args.num_classes,
                dropout=args.linear_dropout)
        else:    
            self.classification_head = ClassificationHead(
                in_dim=self.encoder.output_dim,
                num_labels=args.num_classes,
                dropout=args.linear_dropout
            )

        self.loss_fn = cross_entropy if args.num_classes > 2 else binary_cross_entropy_with_logits
        
        self.num_classes = args.num_classes
        self.use_sslm = args.use_sslm
        self.use_gvp = args.use_gvp
        self.device = args.device
        
    def forward(self, batch, predict=False):
        '''
        Args:
            batch: Tuple of input tensors. 
                struc_seqs: Tensor of shape (batch_size, seq_len), structure sequence
                coords: Tensor of shape (batch_size, L, 3, 3), coordinates
                coord_mask: Tensor of shape (batch_size, L), mask of coordinates
                padding_mask: Tensor of shape (batch_size, L), mask of padding
                confidence: Tensor of shape (batch_size, L), confidence of coordinates
                target: Tensor of shape (batch_size)
        Returns:
            Hidden states, logits and loss        
        '''
        if self.use_sslm and self.use_gvp:
            if predict:
                coords, coord_mask, padding_mask, confidence = self.to_device(*batch[1:], device=self.device)
            else:
                coords, coord_mask, padding_mask, confidence = self.to_device(*batch[1:-1], device=self.device)
            encoder_output = self.encoder(batch[0], coords, coord_mask, padding_mask, confidence)
            input_mask = (~padding_mask).long()
        elif self.use_gvp:
            coords, coord_mask, padding_mask, confidence = self.to_device(*batch[0:-1], device=self.device)
            encoder_output = self.encoder(coords, coord_mask, padding_mask, confidence)
            input_mask = (~padding_mask).long()
        elif self.use_sslm:
            encoder_output, input_mask = self.encoder(batch[0])
        if self.args.atten_pooling:
            logits = self.classification_head(encoder_output, input_mask)
        else:
            logits = self.classification_head(encoder_output)
        
        if not predict:
            target = batch[-1].to(self.device) if self.num_classes > 2 else one_hot(batch[-1], num_classes=self.num_classes).float().to(self.device)
            loss = self.loss_fn(logits, target)
        
        return BaseModelOutput(
            hidden_states=encoder_output,
            logits=logits,
            loss=loss if not predict else None,
            fused_weight = self.encoder.adaptive_fused_weight if self.use_sslm and self.use_gvp else None
        )
    
    def predict(self, batch):
        with torch.no_grad():
            return self(batch, predict=True).logits
    
    def to_device(self, *tensors, device):
        return tuple(tensor.to(device) for tensor in tensors)
    