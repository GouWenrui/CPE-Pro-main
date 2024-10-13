from torch.optim import AdamW
from argparse import ArgumentParser
from tqdm import tqdm
import torch
from datetime import datetime
import random
import numpy as np
import os
import sys
sys.path.append('../')

from dataset import DataModule
from model import SSLMForMLM
from src.utils import seed_everything, EarlyStopping, str2dict

def create_parser():
    parser = ArgumentParser(description='Pretrain SSLM')
    parser.add_argument('--model_path', type=str, default=None, help='model path')
    parser.add_argument('--struc_seq_file', type=str, default=None, help='struc seq file')
    parser.add_argument('--use_token_statistics', action="store_true", default=False, help='use token statistics')
    parser.add_argument('--max_length', type=int, default=256, help='max length')
    parser.add_argument('--sampling_num', type=str2dict, default={'train': 98400, 'valid': 5460, 'test': 5460}, help='sampling num')
    parser.add_argument('--num_workers', type=int, default=4, help='num workers')
    parser.add_argument('--mlm_method', type=str2dict, default={'mlm_probability': 0.25, 'mask_radio': 0.9, 'random_radio': 0.0}, help='mlm method')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='epochs')
    parser.add_argument('--patience', type=int, default=5, help='patience')
    parser.add_argument('--indicator_larger', action="store_true", default=False)
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--seed', type=int, default=3407, help='seed')
    parser.add_argument('--save_model', action="store_true", default=False, help='save model')
    return parser

def main():
    
    parser = create_parser()
    args = parser.parse_args()
    
    seed_everything(args.seed)

    data_module = DataModule(args)
    train_dl, val_dl, test_dl = data_module()
    
    model = SSLMForMLM(args)
    model_size = sum(p.numel() for p in model.parameters()) // 1_000_000
    model_pretrained_size = sum(p.numel() for p in model.parameters() if p.requires_grad) // 1_000_000
    model_name = f'SSLM_t{model.encoder.config.num_hidden_layers}_{model_size}M'
    model.to(args.device)
   
    print(f'========Pretrain SSLM========')
    print(f'= Device: {args.device} ')
    print(f'= Model: {model_name}')
    print(f'= Model size: {model_size}M')
    print(f'= Train parameters: {model_pretrained_size}M')
    print(f'= Epochs: {args.epochs}')
    print(f'= Batch size: {args.batch_size}')
    print(f'= Learning rate: {args.lr}')
    print(f'=============================')
    
    print(f'Pretraining...')
    es = EarlyStopping(patience=args.patience)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.00005)
    val_perplexity = []
    for i in range(args.epochs):
        with tqdm(total=len(train_dl)) as pbar:
            train_loss = []
            for batch_idx, batch in enumerate(train_dl):
                pbar.set_description(f'batch{batch_idx+1}/{len(train_dl)}')
                batch = [tensor.to(args.device) for tensor in batch]
                loss = model(*batch)
                train_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                pbar.set_postfix({'current loss': sum(train_loss)/len(train_loss)})
                pbar.update(1)
        print(f'Epoch {i+1}/{args.epochs}: Loss is {sum(train_loss)/len(train_loss)}')
        val_loss = []
        v_p = []
        best_p = float('inf')
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dl):
                batch = [tensor.to(args.device) for tensor in batch]
                loss = model(*batch)
                val_loss.append(loss.item())
                v_p.append(torch.exp(loss).item())
        val_perplexity.extend(v_p)
        
        avg_val_perplexity = sum(v_p) / len(v_p)
        print(f'Loss: {sum(val_loss)/len(val_loss)}; Perplexity: {avg_val_perplexity}')
        if avg_val_perplexity < best_p:
            best_p = avg_val_perplexity
            best_model_weights = model.state_dict()
        if es.early_stopping(avg_val_perplexity): break
    test_perplexity = []
    for _, batch in enumerate(test_dl):
        batch = [tensor.to(args.device) for tensor in batch]
        loss = model(*batch)
        test_perplexity.append(torch.exp(loss).item())
    print(f'Test perplexity is {sum(test_perplexity)/len(test_perplexity)}')
    print(f'========Pretrain Finished========')
    
    filename = f'{model_name}-pretrain-epoch{args.epochs}-9-0-1-log.csv'
    import csv
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Validation Perplexity'])
        for perplexity in val_perplexity:
            writer.writerow([perplexity])
    print(f"log has been saved to {filename}")
    if args.save_model:
        save_path = f"checkpoints/{model_name}_{datetime.now().strftime('%Y%m%d%H')}.pth"
        torch.save(best_model_weights, save_path)
        print(f'Model saved at {save_path}.')

if __name__ == '__main__':
    main()