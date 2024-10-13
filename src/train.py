import gc
import torch
from tqdm import tqdm
from accelerate import Accelerator
from argparse import ArgumentParser
import csv
from copy import deepcopy
import sys
sys.path.append('../')
sys.path.append('../../')
from model.cpe import CPEPro
from dataset import DataModule
from utils import seed_everything, EarlyStopping, str2dict
from metrics import Metrics
from plm import PLM_GVP



def create_parser():
    
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument('--use_sslm',action="store_true", default=False)
    parser.add_argument('--use_gvp',action="store_true", default=False)
    parser.add_argument('--use_plm',action="store_true", default=False)
    parser.add_argument('--plm_type',type=str, choices=['esm', 'saprot', 'prot_bert'], default='saprot')
    parser.add_argument('--sampling_num', type=str2dict, default={'train': 20000, 'valid': 2000, 'test': 2000})
    parser.add_argument('--process_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--foldseek', type=str, default=None)
    parser.add_argument('--label_mapping', type=str2dict, default={"crystal": 0,"alphafold": 1, "omegafold": 2, "esmfold": 3})
    parser.add_argument('--file_path', type=str, default=None)
    # train
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--early_stop_patience', type=int, default=5)
    parser.add_argument('--indicator_larger', action="store_true", default=False)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--save_trained_model', action="store_true", default=False)
    # gvp_module
    parser.add_argument('--top_k_neighbors', type=int, default=3)
    parser.add_argument('--node_hidden_dim_scalar', type=int, default=128)
    parser.add_argument('--node_hidden_dim_vector', type=int, default=128)
    parser.add_argument('--edge_hidden_dim_scalar', type=int, default=128)
    parser.add_argument('--edge_hidden_dim_vector', type=int, default=128)
    parser.add_argument('--num_encoder_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed_gvp_output_dim', type=int, default=128)
    parser.add_argument('--aa_max_len', type=int, default=256)
    # lm_module
    parser.add_argument('--load_pretrained', action="store_true", default=False)
    parser.add_argument('--plm_dir', type=str, default=None)
    parser.add_argument('--sslm_dir', type=str, default=None)
    parser.add_argument('--sequence_max_length', type=int, default=256)
    parser.add_argument('--freeze_lm', action="store_true", default=False)
    # pooling
    parser.add_argument('--atten_pooling', action="store_true", default=False)
    parser.add_argument('--linear_dropout', type=float, default=0.25)
    parser.add_argument('--num_classes', type=int, default=4)
    args = parser.parse_args()
    
    return args

class Trainer(torch.nn.Module):
    
    def __init__(self, model, args):
        
        super().__init__()
        
        self.args = args
        self.model = model
        self.epoch = args.epoch
        self.accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)        
        self.es = EarlyStopping(args.early_stop_patience, args.indicator_larger)
        self.metrics = Metrics(args.num_classes, args.device)
        
        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_all_params = sum(p.numel() for p in model.parameters()) // 1e6
        print(f'>> Model Size: {model_all_params}M')
        if args.use_plm:
            if args.plm_type == 'saprot':
                self.info=f"{args.plm_dir.split('/')[-1]}_epoch{args.epoch}_lr{args.lr}_{args.num_classes}Classes_{args.sampling_num['train'] / 1e3}K_{model_params//1e3}K"
            else:
                self.info=f"PLM-GVP_{args.plm_dir.split('/')[-1]}_epoch{args.epoch}_lr{args.lr}_{args.num_classes}Classes_{args.sampling_num['train'] / 1e3}K_{model_params//1e6}M"
        else:
            if args.load_pretrained:
                self.info=f"CPEPro[gvp-{args.use_gvp}&Pt_SSLM-{args.use_sslm}&AM-pooling-{args.atten_pooling}]_epoch{args.epoch}_lr{args.lr}_{args.num_classes}Classes_{args.sampling_num['train'] / 1e3}K_{model_params//1e6}M"
            else:
                self.info=f"CPEPro[gvp-{args.use_gvp}&SSLM-{args.use_sslm}&AM-pooling-{args.atten_pooling}]_epoch{args.epoch}_lr{args.lr}_{args.num_classes}Classes_{args.sampling_num['train'] / 1e3}K_{model_params//1e6}M"
        
        self.optimizer =  torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epoch, eta_min=0.00005)
        
        self.model_state_dict = None    
        self.save_trained_model = args.save_trained_model
        
        self.log = {'train_loss': [], 
                    'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_mcc': [],
                    'test_acc': [], 'test_precision': [], 'test_recall': [], 'test_f1': [], 'test_mcc': []}
        
    def forward(self, data_module):
        
        train_dl, val_dl, test_dl = data_module()
        
        self.model, self.optimizer, train_dl = self.accelerator.prepare(
            self.model, self.optimizer, train_dl
        )
        print(f'>> Training {self.info}...')
        for epoch in range(self.epoch):
            tl = []; vl = []
            with tqdm(total=len(train_dl)) as pbar:
                pbar.set_description(f'Training Epoch {epoch+1}/{self.epoch}')
                for batch_idx, batch in enumerate(train_dl):
                    tl.append(self.training_step(batch, batch_idx))
                    pbar.set_postfix({'current loss': sum(tl)/len(tl)})
                    pbar.update(1)
            self.log['train_loss'].extend(tl)
            print(f">> Epoch {epoch+1} Loss: {sum(tl)/len(tl)}")
            
            self.model.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_dl):
                    vl.append(self.validation_step(batch, batch_idx))
                metrics = self.metrics.compute()
                self.metrics.reset()
                self.log['val_loss'].extend(vl)
                self.log['val_acc'].append(metrics['acc'])
                self.log['val_precision'].append(metrics['precision'])
                self.log['val_recall'].append(metrics['recall'])
                self.log['val_f1'].append(metrics['f1'])
                self.log['val_mcc'].append(metrics['mcc'])
                
                print(f">> Valid loss: {sum(vl)/len(vl)}")
                print(f">> acc: {metrics['acc']}; precision: {metrics['precision']}; recall: {metrics['recall']}; f1: {metrics['f1']}; mcc: {metrics['mcc']}")
                
                if metrics['acc'] >= max(self.log['val_acc']):
                    self.model_state_dict = deepcopy(self.model.state_dict())
            if self.es.early_stopping(metrics['acc']): print(f'>> Early stop at epoch {epoch+1}'); break
        
        print(f'>> Training finished.\n>> Testing...')
        self.model.load_state_dict(self.model_state_dict)        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dl):
                self.test_step(batch, batch_idx)
            metrics = self.metrics.compute()
            self.log['test_acc'].append(metrics['acc'])
            self.log['test_precision'].append(metrics['precision'])
            self.log['test_recall'].append(metrics['recall'])
            self.log['test_f1'].append(metrics['f1'])
            self.log['test_mcc'].append(metrics['mcc'])
            print(f">> acc: {metrics['acc']}; precision: {metrics['precision']}; recall: {metrics['recall']}; f1: {metrics['f1']}; mcc: {metrics['mcc']}")         
        self.save_log()
        if self.save_trained_model: self.save_model()
        gc.collect()
    
    def training_step(self, batch, batch_idx):
        with self.accelerator.accumulate(self.model):
            loss = self.model(batch).loss
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
        return round(loss.item(), 4)    

    def validation_step(self, batch, batch_idx):
        output = self.model(batch)
        self.metrics.update(output.logits, batch[-1])
        
        return round(output.loss.item(), 4)
    
    def test_step(self, batch, batch_idx):
        self.metrics.update(self.model(batch).logits, batch[-1].to(self.args.device))

    def save_model(self):
        save_path=f"checkpoint/{self.info}_{self.args.sslm_dir.split('/')[-1]}_model.pth"
        torch.save(self.model_state_dict, save_path)
        print(f'model saved at {save_path}')

    def save_log(self):
        if 'CPEPro' in self.info:
            filename = f"train_log/CPE-Pro/{self.info}_{self.args.sslm_dir.split('/')[-1]}_log.csv"
        else:
            filename = f'train_log/plm/{self.info}_log.csv'
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['step/epoch', 'train_loss', 'val_loss', 
                      'val_acc', 'val_precision', 'val_recall', 'val_f1', 'val_mcc',
                      'test_acc', 'test_precision', 'test_recall', 'test_f1', 'test_mcc']
            writer.writerow(header)
            for i in range(len(self.log['train_loss'])):
                epoch = i + 1
                train_loss = self.log['train_loss'][i] if i < len(self.log['train_loss']) else ''
                val_loss = self.log['val_loss'][i] if i < len(self.log['val_loss']) else ''
                val_acc = self.log['val_acc'][i] if i < len(self.log['val_acc']) else ''
                val_precision = self.log['val_precision'][i] if i < len(self.log['val_precision']) else ''
                val_recall = self.log['val_recall'][i] if i < len(self.log['val_recall']) else ''
                val_f1 = self.log['val_f1'][i] if i < len(self.log['val_f1']) else ''
                val_mcc = self.log['val_mcc'][i] if i < len(self.log['val_mcc']) else ''
                test_acc = self.log['test_acc'][0] if i == 0 else ''
                test_precision = self.log['test_precision'][0] if i == 0 else ''
                test_recall = self.log['test_recall'][0] if i == 0 else ''
                test_f1 = self.log['test_f1'][0] if i == 0 else ''
                test_mcc = self.log['test_mcc'][0] if i == 0 else ''
                writer.writerow([epoch, train_loss, val_loss, 
                                 val_acc, val_precision, val_recall, val_f1, val_mcc, 
                                 test_acc, test_precision, test_recall, test_f1, test_mcc])
    
    


if __name__ == '__main__':
    
    args = create_parser()
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    seed_everything(args.seed)
    data_module = DataModule(args)
    model = CPEPro(args) if not args.use_plm else PLM_GVP(args)
    trainer = Trainer(model.to(args.device), args)
    trainer(data_module)

    
    

    
