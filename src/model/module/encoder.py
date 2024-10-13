import torch.nn as nn
import torch

from model.module.padding import padding_with_method
from model.module.gvp.encoder import GVPEncoder
from model.module.sslm.encoder import SSLMEncoder
from model.module.gvp.gvp_utils import flatten_graph, unflatten_graph

class Intergrate_Encoder(nn.Module):
    '''
    sequences: list of all sequences, each sequence is a string
    coords: torch.Tensor, shape (batch_size, sequence_max_length, 3)
    coord_mask: torch.Tensor, shape (batch_size, sequence_max_length)
    padding_mask: torch.Tensor, shape (batch_size, sequence_max_length)
    confidence: torch.Tensor, shape (batch_size, sequence_max_length)
    '''
    def __init__(self, args):
        
        super().__init__()

        self.sslm_encoder = SSLMEncoder(args)
        self.embed_sslm_output = nn.Linear(
            self.sslm_encoder.output_dim,
            args.node_hidden_dim_scalar
        )
        
        self.gvp_encoder = GVPEncoder(args)
        self.output_dim = args.node_hidden_dim_scalar + (3 * args.node_hidden_dim_vector)
        
        self.layer_norm = nn.LayerNorm(args.node_hidden_dim_scalar)
        self.adaptive_fused_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, sequences, coords, coord_mask, padding_mask, confidence):
        
        seq_rep, _ = self.sslm_encoder(sequences)
        embed_seq_rep = self.embed_sslm_output(seq_rep)

        node_embeddings, edge_embeddings, edge_index = self.gvp_encoder.get_embeddings(
            coords=coords,
            coord_mask=coord_mask,
            padding_mask=padding_mask,
            confidence=confidence
        )
        node_embeddings = unflatten_graph(node_embeddings, coords.shape[0]) # (batch_size, node, node_embedding_dim) 
        scalars, vectors=node_embeddings
        _, n, _ = scalars.shape
        _, L, _ = embed_seq_rep.shape 
        embed_seq_rep = padding_with_method(embed_seq_rep, n-L, 'mean_std') if L < n else embed_seq_rep[:, :n, :]
        weight = torch.sigmoid(self.adaptive_fused_weight)
        scalars = self.layer_norm(scalars)
        embed_seq_rep = self.layer_norm(embed_seq_rep)
        integrated_node_scalars = weight * scalars + (1 - weight) * embed_seq_rep

        node_embeddings = (integrated_node_scalars, vectors)
        node_embeddings = flatten_graph(node_embeddings=node_embeddings)
        output = self.gvp_encoder.integrate(node_embeddings, edge_embeddings, edge_index, coords)
        
        return output