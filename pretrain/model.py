from transformers import EsmForMaskedLM as SSLMForMaskedLM 
from transformers import EsmConfig as SSLMConfig

import torch.nn as nn

class SSLMForMLM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = SSLMForMaskedLM(
            SSLMConfig.from_pretrained(args.model_path)
            )
    def forward(self, 
                input_ids, 
                attention_mask, 
                labels):
        masked_lm_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = masked_lm_output.loss
        return loss