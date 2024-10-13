from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from Bio import SeqIO
import os
from tqdm import tqdm
from transformers import (
    EsmModel,
    EsmTokenizer,
    AutoTokenizer,
    BertModel,
    BertTokenizer
)

def load_pdb_names(path):
    import os
    names = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.pdb'):
                names.append(file)
    return names

def get_seq(foldseek, path, chains: list = None, process_id: int = 0):
        """
        Args:
            foldseek: Binary executable file of foldseek
            path: Path to pdb file
            chains: Chains to be extracted from pdb file. If None, all chains will be extracted.
            process_id: Process ID for temporary files. This is used for parallel processing.

        Returns: sequence dictionary: {chain: sequence}
        """
        assert os.path.exists(foldseek), f"Foldseek not found: {foldseek}"
        assert os.path.exists(path), f"Pdb file not found: {path}"

        tmp_save_path = f"get_struc_seq_{process_id}.tsv"
        cmd = f"{foldseek} structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
        os.system(cmd)

        seq_dict = {}
        name = os.path.basename(path)
        with open(tmp_save_path, "r") as r:
            for line in r:
                desc, seq, struct_seq = line.split("\t")[:3]
                name_chain = desc.split(" ")[0]
                chain = name_chain.replace(name, "").split("_")[-1]
                if chains is None or chain in chains:
                    if chain not in seq_dict:
                        combined_seq = "".join([a + b.lower() for a, b in zip(seq, struct_seq)])
                        seq_dict[chain] = (seq, struct_seq, combined_seq)

        os.remove(tmp_save_path)
        os.remove(tmp_save_path + ".dbtype")
        return seq_dict   

def model_encoder(model_type, lm_dir, seqs, device='cuda'):
    if model_type == 'esm' or model_type == 'sslm':
        tokenizer = AutoTokenizer.from_pretrained(lm_dir)
        model = EsmModel.from_pretrained(lm_dir).to(device)
    elif model_type == 'saprot':
        tokenizer = EsmTokenizer.from_pretrained(lm_dir)
        model = EsmModel.from_pretrained(lm_dir).to(device)
    elif model_type == 'prot_bert':
        tokenizer = BertTokenizer.from_pretrained(lm_dir)
        model = BertModel.from_pretrained(lm_dir).to(device)
    params = sum(p.numel() for p in model.parameters()) // 1e6
    print(f'Params: {params}M')
    print('Model loaded')
    rep_list = []
    with torch.no_grad():
        with tqdm(total=len(seqs)) as pbar:
            for i in range(0, len(seqs)):
                tokenizer_output = tokenizer(seqs[i],
                                            max_length=128,
                                            padding='max_length',
                                            truncation=True,
                                            return_tensors='pt').to(device)
                model_output = model(input_ids=tokenizer_output['input_ids'],
                                     attention_mask=tokenizer_output['attention_mask'])
                rep_list.append(model_output["last_hidden_state"].cpu())
                pbar.update(1)              
    rep = torch.cat(rep_list, dim=0)
    return rep

def cluster(rep, labels, model_name=None):
    rep = torch.tensor(rep, dtype=torch.float32)
    rep = rep.permute(0, 2, 1)
    rep = nn.AdaptiveAvgPool1d(1)(rep)
    rep = rep.squeeze(-1)
    tsne = TSNE(n_components=2, 
                perplexity=30,
                n_iter=3000,
                random_state=0)
    tsne_results = tsne.fit_transform(rep)
    plt.figure(figsize=(5, 3))
    custom_palette = ["#A38EB7", "#8FD1C1"]
    sns.scatterplot(x=tsne_results[:,0], 
                    y=tsne_results[:,1], 
                    hue=labels, 
                    palette=custom_palette)
    plt.title(f'{model_name}')
    plt.legend()
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig(f'{model_name}_cluster.png', dpi=500, bbox_inches='tight')

def main(args):
    
    alpha_pdbs = load_pdb_names(args.alpha_pdb_path)
    beta_pdbs = load_pdb_names(args.beta_pdb_path)
    seqs = []; labels = []
    
    with tqdm(total=len(alpha_pdbs)) as pbar:
        for i in range(len(alpha_pdbs)):
            pbar.set_description(f'{i+1}/{len(alpha_pdbs)}')
            seq_dict = get_seq(foldseek=args.foldseek_path,
                                    path=os.path.join(args.alpha_pdb_path, alpha_pdbs[i]))
            if len(seq_dict) == 0:
                continue
            seqs.append(next(iter(seq_dict.values()))[1])
            pbar.update(1)
    labels.extend(['all-α helical']*len(seqs))
    alpha_len = len(seqs)
    
    with tqdm(total=len(beta_pdbs)) as pbar:
        for i in range(len(beta_pdbs)):
            pbar.set_description(f'{i+1}/{len(beta_pdbs)}')
            seq_dict = get_seq(foldseek=args.foldseek_path,
                                    path=os.path.join(args.beta_pdb_path, beta_pdbs[i]))
            if len(seq_dict) == 0:
                continue
            seqs.append(next(iter(seq_dict.values()))[0])
            pbar.update(1)
    beta_len = len(seqs) - alpha_len
    labels.extend(['all-β sheet']*beta_len)
    
    rep = model_encoder(model_type=args.model_type, 
                        lm_dir=args.lm_dir, 
                        seqs=seqs,
                        device=args.device)
    
    cluster(rep, labels, model_name=args.model_name)

if __name__ == '__main__':
    
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('--foldseek_path', type=str, required=True, default=None)
    parser.add_argument('--alpha_pdb_path', type=str, required=True)
    parser.add_argument('--beta_pdb_path', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--lm_dir', type=str, required=True) 
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    main(args)
    
    

