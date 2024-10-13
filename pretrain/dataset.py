from transformers import DataCollatorForLanguageModeling
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoTokenizer as SSEMTokenizer
import os
from tqdm import tqdm
from Bio import SeqIO
import numpy as np

def load_ss(struc_seq_file, sampling_num=None):
    struc_seq = []
    with tqdm(total = sampling_num) as pbar:
        pbar.set_description("Loading seq from {}".format(struc_seq_file))
        for record in SeqIO.parse(struc_seq_file, 'fasta'):
            sequence = str(record.seq)
            struc_seq.append(sequence)
            pbar.update(1)
            if sampling_num != None and  len(struc_seq) == sampling_num:
                break
    print("Loaded.")
    return struc_seq

class TokenStatistics:

    def __init__(self, vocab, counts):
        self.vocab = vocab 
        self.counts = counts

    @property
    def weights(self):
        np_counts = np.array(self.counts, dtype=np.float64)
        weights = np_counts / np.sum(np_counts)
        weights = np.round(weights, 2)
        return weights

    @classmethod
    def from_fasta(cls, fasta_file_path):
        sequences = []
        if fasta_file_path.endswith('.fasta'):
            sequences = [str(record.seq) for record in SeqIO.parse(fasta_file_path, 'fasta')]
        else:
            for file in os.listdir(fasta_file_path):
                full_path = os.path.join(fasta_file_path, file)
                data = [str(record.seq) for record in SeqIO.parse(full_path, 'fasta')]
                sequences.extend(data)
        flatten_sequence = "".join(sequences)
        vocab = "ACDEFGHIKLMNPQRSTVWYX"
        counts = [flatten_sequence.count(token) for token in vocab]
        return cls(vocab=vocab, counts=counts)

class DataCollatorForMLM(DataCollatorForLanguageModeling):
    def __init__(self, 
                 tokenizer,
                 modification_probability=0.15,
                 to_mask_ratio=0.8,
                 to_random_token_ratio=0.1,
                 random_tokens=None,
                 random_tokens_weight=None):
        super(DataCollatorForMLM, self).__init__(tokenizer=tokenizer, mlm_probability=modification_probability)
        self.tokenizer = tokenizer
        self.modification_probability = modification_probability
        self.to_mask_ratio = to_mask_ratio
        self.to_random_token_ratio = to_random_token_ratio
        self.original_ration = 1 - to_mask_ratio - to_random_token_ratio
        
        if random_tokens is None: self.random_tokens = None
        else:
            assert random_tokens_weight is not None
            self.random_tokens = torch.tensor([tokenizer.get_vocab()[each] for each in random_tokens], dtype=torch.long)

        if random_tokens_weight is None: self.random_tokens_weight = None
        else:
            assert self.random_tokens is not None
            self.random_tokens_weight = torch.tensor(random_tokens_weight, dtype=torch.float)

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)  
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # to_mask_radio of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # to_random_token_radio of the time, we replace masked input tokens with random word
        # x * (1 - to_mask_ratio) = to_random_token_ratio
        # x = to_random_token_ratio / (1 - to_mask_ratio)
        x = self.to_random_token_ratio / (1 - self.to_mask_ratio)
        indices_random = torch.bernoulli(torch.full(labels.shape, x)).bool() & masked_indices & ~indices_replaced

        if self.random_tokens_weight is not None:
            total = labels.numel()
            random_words = torch.multinomial(
                self.random_tokens_weight,
                total,
                replacement=True
            )
            random_words = self.random_tokens[random_words].reshape_as(labels)
            inputs[indices_random] = random_words[indices_random]
        else:
            random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
            inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

class SSLMDataset(Dataset): # Structured Sequence Dataset
    def __init__(self, 
                 args,
                 tokenizer):
        self.args = args
        self.sequences = load_ss(args.struc_seq_file, sum(args.sampling_num.values()))            
        self.tokenizer = tokenizer
        self.max_length = args.max_length
        if args.use_token_statistics:
            token_statistics = TokenStatistics.from_fasta(args.struc_seq_file)
            self.data_collator = DataCollatorForMLM(
                tokenizer=tokenizer,
                random_tokens=token_statistics.vocab,
                random_tokens_weight=token_statistics.weights,
                modification_probability=args.mlm_method['mlm_probability'],
                to_mask_ratio=args.mlm_method['mask_radio'],
                to_random_token_ratio=args.mlm_method['random_radio'],
                )
        else:
            self.data_collator = DataCollatorForMLM(
                tokenizer=tokenizer,
                modification_probability=args.mlm_method['mlm_probability'],
                to_mask_ratio=args.mlm_method['mask_radio'],
                to_random_token_ratio=args.mlm_method['random_radio'],
                )
            
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        inputs = self.tokenizer(sequence, 
                                max_length=self.max_length,
                                truncation=True,
                                padding='max_length',
                                return_tensors='pt')
        input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
        input_ids, labels = self.data_collator.torch_mask_tokens(input_ids)
        input_ids = input_ids.squeeze(0)
        attention_mask = attention_mask.squeeze(0)
        labels = labels.squeeze(0)
        return (input_ids, attention_mask, labels)

class DataModule(nn.Module):
    def __init__(self, args):
        super(DataModule, self).__init__()
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        dataset = SSLMDataset(
            args,
            tokenizer=SSEMTokenizer.from_pretrained(args.model_path),
        )
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [args.sampling_num['train'], args.sampling_num['valid'], args.sampling_num['test']]    
        )
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
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )
        test_dl = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )
        return train_dl, val_dl, test_dl