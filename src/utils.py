import os
import sys
import random
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import ast

from esm.data import BatchConverter, Alphabet
from typing import Sequence, Tuple, List
sys.path.append('../')
from model.module.gvp.util import load_coords

class LoadData:

    def __init__(self, args, run_mode):
        
        assert args.use_sslm or args.use_gvp or args.use_plm, 'At least one of use_sslm, use_gvp, use_plm should be True'
        assert not (args.use_sslm and args.use_plm), 'use_sslm and use_plm cannot be True at the same time'
        
        self.sampling_num = args.sampling_num[run_mode]
        self.each_class_num = args.sampling_num[run_mode] // len(args.label_mapping)
        assert args.sampling_num[run_mode] % len(
            args.label_mapping) == 0, 'sampling_num should be a multiple of the number of classes'

        assert isinstance(args.file_path,
                          str), 'file_path should be a string that points to the folder containing pdb files.'
        
        self.pdb_path = args.file_path + run_mode
        self.label_mapping = args.label_mapping
        self.pdb_files = self.sample_pdb_files(self.get_file_name_list(self.pdb_path))
        
        if args.use_gvp:
            assert args.aa_max_len != None, "The maximum length of `coords_max_shape` should be provided."
            self.coords_max_shape = [args.aa_max_len, 3, 3]
            self.confidence_max_shape = [args.aa_max_len]

        self.args = args
        self.use_sslm = args.use_sslm
        self.use_gvp = args.use_gvp
        self.use_plm = args.use_plm
        self.plm_type = args.plm_type
        self.file_path = args.file_path
        self.foldseek = args.foldseek
        self.load_coords = load_coords

    def load(self):
        
        data = []
        if self.use_sslm and self.use_gvp:
            coords, coord_mask, padding_mask, confidence, struc_seqs = self.get_structure_from_pdb(self.pdb_path,
                                                                                                  self.pdb_files)
            for i in range(coords.size()[0]):    
                label = self.get_label(self.pdb_files[i])
                data.append((struc_seqs[i], coords[i], coord_mask[i], padding_mask[i], confidence[i], label))

        elif self.use_sslm:
            with tqdm(total=len(self.pdb_files)) as pbar:
                pbar.set_description(">> Loading 'structure-sequence' using foldseek from {}".format(self.pdb_path))
                for file in self.pdb_files:
                    struc_seq = self.get_struc_seq(
                        foldseek=self.foldseek, 
                        path=os.path.join(self.pdb_path, file),
                        process_id=self.args.process_id)["A"]
                    data.append((struc_seq, self.get_label(file)))
                    pbar.update(1)
            print('>> ========= Loaded ==========')

        elif self.use_plm:
            if self.plm_type == 'esm' or self.plm_type == 'prot_bert':
                coords, coord_mask, padding_mask, confidence, aa_seqs = self.get_structure_from_pdb(self.pdb_path,self.pdb_files)
                for i in range(coords.size()[0]):
                    label = self.get_label(self.pdb_files[i])
                    data.append((aa_seqs[i], coords[i], coord_mask[i], padding_mask[i], confidence[i], label))
            elif self.plm_type == 'saprot':
                with tqdm(total=len(self.pdb_files)) as pbar:
                    pbar.set_description(">> Loading 'Structure Aware' sequences using from {}".format(self.pdb_path))
                    for file in self.pdb_files:
                        combined_seq = self.get_struc_seq(
                            foldseek=self.foldseek, 
                            path=os.path.join(self.pdb_path, file),
                            process_id=self.args.process_id,
                            combine=True)["A"]
                        data.append((combined_seq, self.get_label(file)))
                        pbar.update(1)
                print('>> ========= Loaded ==========')
            else:
                raise ValueError("plm_type should be either 'esm' or 'prot_bert' or 'saprot'")
            

        elif self.use_gvp:
            self.pdb_files = self.sample_pdb_files(self.get_file_name_list(self.pdb_path))
            coords, coord_mask, padding_mask, confidence = self.get_structure_from_pdb(self.pdb_path, self.pdb_files)
            for i in range(coords.size()[0]):
                data.append(
                    (coords[i], coord_mask[i], padding_mask[i], confidence[i], self.get_label(self.pdb_files[i])))
        
        return data

    def get_file_name_list(self, path):
        file_name_list = []
        for file in os.listdir(path):
            file_name_list.append(file)
        return file_name_list

    def sample_pdb_files(self, pdb_files):
        grouped_files = {label: [] for label, _ in self.label_mapping.items()}
        for pdb_file in pdb_files:
            for label, _ in self.label_mapping.items():
                if f"-{label}" in pdb_file:
                    if len(grouped_files[label]) == self.each_class_num: continue
                    grouped_files[label].append(pdb_file)
                    break
        resampled_files = []
        for _, files in grouped_files.items():
            resampled_files.extend(files)
            
        return resampled_files

    def get_struc_seq(self, foldseek, path, chains: list = ["A"], process_id: int = 0, combine: bool = False):
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
                desc, seq, struc_seq = line.split("\t")[:3]
                name_chain = desc.split(" ")[0]
                chain = name_chain.replace(name, "").split("_")[-1]
                if chains is None or chain in chains:
                    if chain not in seq_dict:
                        if combine:
                            combined_seq = "".join([a + b.lower() for a, b in zip(seq, struc_seq)])
                        seq_dict[chain] = combined_seq if combine else struc_seq

        os.remove(tmp_save_path)
        os.remove(tmp_save_path + ".dbtype")
        
        return seq_dict

    def get_label(self, file_name):
        for key, value in self.label_mapping.items():
            if key in file_name: return value
        return ValueError("No corresponding label found for {}'s data.".format(file_name))

    def get_structure_from_pdb(self, path, file_name):
        '''
        Load protein structure from pdb file
        :param file_name:
        :return: structure
        '''
        if self.use_gvp and not self.use_sslm: self.sampling_num = len(file_name) if len(
            file_name) < self.sampling_num else self.sampling_num
        confident = None
        raw_batch = []
        with tqdm(total=self.sampling_num) as pbar:
            if self.use_sslm:
                struc_seqs = []
                pbar.set_description(">> Loading structures and 'structure-sequence' from {}".format(path))
            if self.use_plm:
                aa_seqs = []
                pbar.set_description(">> Loading structures and AA sequence from {}".format(path))
            else:
                pbar.set_description(">> Loading structures from {}".format(path))
            for index in range(self.sampling_num):
                file_path = os.path.join(path, file_name[index])
                if self.use_sslm:
                    struc_seqs.append(self.get_struc_seq(
                        foldseek=self.foldseek, 
                        path=file_path,
                        process_id=self.args.process_id)["A"])
                coords, seq = self.load_coords(file_path, ["A"])
                if self.plm_type == 'prot_bert': seq = ' '.join(seq)
                if self.use_plm: aa_seqs.append(seq)
                raw_batch.append((coords, confident, seq))
                pbar.update(1)
        print('>> ========= Loaded ==========')
        alphabet = Alphabet.from_architecture("invariant_gvp")
        converter = CoordBatchConverter(alphabet)
        coords, coord_mask, padding_mask, confidence = converter(
            raw_batch=raw_batch,
            coords_max_shape=self.coords_max_shape,
            confidence_max_shape=self.confidence_max_shape)
        
        if self.use_gvp and self.use_sslm:
            return coords, coord_mask, padding_mask, confidence, struc_seqs
        
        if self.use_gvp and self.use_plm:
            return coords, coord_mask, padding_mask, confidence, aa_seqs
        
        return coords, coord_mask, padding_mask, confidence

class CoordBatchConverter(BatchConverter):

    def __call__(self, 
                 raw_batch: Sequence[Tuple[Sequence, str]], 
                 coords_max_shape, 
                 confidence_max_shape,
                 device=None):
        """
        Args:
            raw_batch: List of tuples (coords, confidence, seq)
            In each tuple,
                coords: list of floats, shape L x 3 x 3
                confidence: list of floats, shape L; or scalar float; or None
                seq: string of length L
        Returns:
            coords: Tensor of shape batch_size x L x 3 x 3
            confidence: Tensor of shape batch_size x L
            strs: list of strings
            tokens: LongTensor of shape batch_size x L
            padding_mask: ByteTensor of shape batch_size x L
        """
        self.alphabet.cls_idx = self.alphabet.get_idx("<cath>")
        batch = []
        for coords, confidence, seq in raw_batch:
            if confidence is None:
                confidence = 1.
            if isinstance(confidence, float) or isinstance(confidence, int):
                confidence = [float(confidence)] * len(coords)
            if seq is None:
                seq = 'X' * len(coords)
            batch.append(((coords, confidence), seq))
        coords_and_confidence, strs, tokens = super().__call__(batch)
        coords = [
            F.pad(torch.tensor(cd), (0, 0, 0, 0, 1, 1), value=np.inf)
            for cd, _ in coords_and_confidence
        ]
        confidence = [
            F.pad(torch.tensor(cf), (1, 1), value=-1.)
            for _, cf in coords_and_confidence
        ]
        coords = self.collate_dense_tensors(coords, pad_v=np.nan, max_shape=coords_max_shape)
        confidence = self.collate_dense_tensors(confidence, pad_v=-1., max_shape=confidence_max_shape)
        if device is not None:
            coords = coords.to(device)
            confidence = confidence.to(device)
            tokens = tokens.to(device)
        padding_mask = torch.isnan(coords[:, :, 0, 0])
        coord_mask = torch.isfinite(coords.sum(-2).sum(-1))
        confidence = confidence * coord_mask + (-1.) * padding_mask
        
        return coords, coord_mask, padding_mask, confidence

    def from_lists(self, coords_list, confidence_list=None, seq_list=None, device=None):
        """
        Args:
            coords_list: list of length batch_size, each item is a list of
            floats in shape L x 3 x 3 to describe a backbone
            confidence_list: one of
                - None, default to highest confidence
                - list of length batch_size, each item is a scalar
                - list of length batch_size, each item is a list of floats of
                    length L to describe the confidence scores for the backbone
                    with values between 0. and 1.
            seq_list: either None or a list of strings
        Returns:
            coords: Tensor of shape batch_size x L x 3 x 3
            confidence: Tensor of shape batch_size x L
            strs: list of strings
            tokens: LongTensor of shape batch_size x L
            padding_mask: ByteTensor of shape batch_size x L
        """
        batch_size = len(coords_list)
        if confidence_list is None:
            confidence_list = [None] * batch_size
        if seq_list is None:
            seq_list = [None] * batch_size
        raw_batch = zip(coords_list, confidence_list, seq_list)
        
        return self.__call__(raw_batch, device)

    @staticmethod
    def collate_dense_tensors(samples, pad_v, max_shape=None):
        """
        Takes a list of tensors with the following dimensions:
            [(d_11,       ...,           d_1K),
             (d_21,       ...,           d_2K),
             ...,
             (d_N1,       ...,           d_NK)]
        and stack + pads them into a single tensor of:
        (N, max_i=1,N { d_i1 }, ..., max_i=1,N {diK})
        """
        if len(samples) == 0:
            return torch.Tensor()
        if len(set(x.dim() for x in samples)) != 1:
            raise RuntimeError(
                f"Samples has varying dimensions: {[x.dim() for x in samples]}"
            )
        (device,) = tuple(set(x.device for x in samples))
        result = torch.empty(
            len(samples), *max_shape, dtype=samples[0].dtype, device=device
        )
        result.fill_(pad_v)
        for i in range(len(samples)):
            result_i = result[i]
            t = samples[i]
            slices = tuple(slice(0, min(dim, max_dim)) for dim, max_dim in zip(t.shape, max_shape))
            result_i[slices] = t[slices]

        return result

def seed_everything(seed_value):
    
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def str2dict(value):
    """Convert string to dictionary."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError("Dictionary value is not valid.")
    
class EarlyStopping:
    
    def __init__(self, 
                 patience=10,
                 indicator_larger_better=True):
        
        self.patience = patience
        self.counter = 0
        self.larger_better = indicator_larger_better
        if indicator_larger_better:
            self.best = -np.inf
        else:
            self.best = np.inf

    def early_stopping(self, current_indicator):
        
        update = (current_indicator > self.best) if self.larger_better else (current_indicator < self.best)
        
        if update:
            self.best = current_indicator
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience