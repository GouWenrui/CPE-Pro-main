import torch.nn as nn
from transformers import EsmModel as SSLM # "Structure-Sequence" Language Model
from transformers import EsmConfig as SSLMConfig
from transformers import EsmTokenizer as SSLMTokenizer

class SSLMEncoder(nn.Module):
    
    def __init__(self, args):
        
        super().__init__()
        
        if args.load_pretrained:
            self.sslm = SSLM.from_pretrained(args.sslm_dir)
            if args.freeze_lm:
                for param in self.sslm.parameters():
                    param.requires_grad = False
        else:
            self.sslm=SSLM(SSLMConfig.from_pretrained(args.sslm_dir))
        self.tokenizer=SSLMTokenizer.from_pretrained(args.sslm_dir)
        self.sequence_max_length = args.sequence_max_length
        self.output_dim = self.sslm.config.hidden_size
        self.device = args.device
    
    def forward(self, x):
        
        tokenizer_output = self.tokenizer(x,
                                max_length=self.sequence_max_length,
                                padding='max_length',
                                truncation=True,
                                return_tensors='pt').to(self.device)
        
        sslm_output = self.sslm(**tokenizer_output).last_hidden_state
        
        return sslm_output, tokenizer_output['attention_mask']

