import torch 
import torch.nn as nn 
from transformers import (
    EsmModel, 
    AutoTokenizer, 
    EsmTokenizer,
    BertModel,
    BertTokenizer)
from torch.nn.functional import (
    binary_cross_entropy_with_logits, 
    cross_entropy,
    one_hot)
from esm.inverse_folding.gvp_utils import unflatten_graph

from model.base import BaseModelOutput
from model.module.padding import padding_with_method
from model.module.gvp.encoder import GVPEncoder
from model.module.classify import ClassificationHead
from model.module.gvp.gvp_utils import flatten_graph

# =================
# Baseline models
# =================
# ESM-1b + GVP
# ESM-1v　+ GVP
# ESM-2 + GVP 
# ProtBert　+ GVP
# SaProt
# =================

class PLMEncoder(nn.Module):
    
    def __init__(self, args):
        
        super().__init__()
        
        self.sequence_max_length = args.sequence_max_length
        
        if args.plm_type == 'esm':
            self.tokenizer = AutoTokenizer.from_pretrained(args.plm_dir)
            self.plm_encoder = EsmModel.from_pretrained(args.plm_dir)
        elif args.plm_type == 'saprot':
            self.tokenizer = EsmTokenizer.from_pretrained(args.plm_dir)
            self.plm_encoder = EsmModel.from_pretrained(args.plm_dir)
        elif args.plm_type == 'prot_bert':
            self.tokenizer = BertTokenizer.from_pretrained(args.plm_dir)
            self.plm_encoder = BertModel.from_pretrained(args.plm_dir)
        else:
            raise ValueError('plm_type must be one of "esm", "saprot", "prot_bert"')
        
        if args.freeze_lm:
            for param in self.plm_encoder.parameters():
                param.requires_grad = False
        
        self.output_dim = self.plm_encoder.config.hidden_size
        self.device = args.device
        
    def forward(self, sequences):
        
        inputs = self.tokenizer(sequences, 
                                return_tensors='pt', 
                                padding=True, 
                                truncation=True, 
                                max_length=self.sequence_max_length).to(self.device)
        outputs = self.plm_encoder(**inputs)
        
        return outputs.last_hidden_state

class PLM_GVP(nn.Module):
    
    def __init__(self, args):
        
        super().__init__()
        
        self.plm_encoder = PLMEncoder(args)
        
        if args.plm_type == 'esm' or args.plm_type == 'prot_bert':
            self.gvp_encoder = GVPEncoder(args)
            self.embed_plm_output = nn.Linear(
                self.plm_encoder.output_dim,
                args.node_hidden_dim_scalar
            )
        
        self.classification_head = ClassificationHead(
            in_dim=self.plm_encoder.output_dim if args.plm_type == 'saprot' else self.gvp_encoder.output_dim,
            dropout=args.linear_dropout,
            num_labels=args.num_classes)
        
        self.plm_type = args.plm_type
        self.device = args.device
        self.num_classes = args.num_classes
        self.adaptive_fused_weight = nn.Parameter(torch.tensor(0.5))
        self.loss_fn = cross_entropy if args.num_classes > 2 else binary_cross_entropy_with_logits 
    
    def forward(self, batch):
        
        plm_output = self.plm_encoder(batch[0])
        
        if self.plm_type != 'saprot':
            coords, coord_mask, padding_mask, confidence = self.to_device(*batch[1:-1], device=self.device)
            node_embeddings, edge_embeddings, edge_index = self.gvp_encoder.get_embeddings(
                coords=coords,
                coord_mask=coord_mask,
                padding_mask=padding_mask,
                confidence=confidence
            )
            node_embeddings = unflatten_graph(node_embeddings, coords.shape[0])
            scalars, vectors=node_embeddings
            embed_seq_rep = self.embed_plm_output(plm_output)
            _, n, _ = scalars.shape
            _, L, _ = embed_seq_rep.shape 
            embed_seq_rep = padding_with_method(embed_seq_rep, n-L, 'mean_std') if L < n else embed_seq_rep[:, :n, :]
            weight = torch.sigmoid(self.adaptive_fused_weight)
            integrated_node_scalars = weight * scalars + (1 - weight) * embed_seq_rep
            node_embeddings = (integrated_node_scalars, vectors)
            node_embeddings = flatten_graph(node_embeddings= node_embeddings)
            encoder_output = self.gvp_encoder.integrate(node_embeddings, edge_embeddings, edge_index, coords)   
        logits = self.classification_head(plm_output if self.plm_type == 'saprot' else encoder_output)
        target = batch[-1].to(self.device) if self.num_classes > 2 else one_hot(batch[-1], 2).float().to(self.device)
        
        return BaseModelOutput(
            hidden_states=plm_output if self.plm_type == 'saprot' else node_embeddings,
            logits=logits,
            loss=self.loss_fn(logits, target),
            fused_weight=self.adaptive_fused_weight
        )
    
    def to_device(self, *tensors, device):
        
        return tuple(tensor.to(device) for tensor in tensors)
        
    