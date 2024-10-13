from requirements import *

class LoadData:
        
        def __init__(self, file_path, max_len):
            
            
            if not os.path.exists(file_path):
                return ValueError(f'{file_path} not found')
            if os.path.isfile(file_path) and file_path.endswith('.pdb'):
                self.files = [file_path]
            elif os.path.isdir(file_path):
                self.files = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.pdb')]
            else:
                return ValueError(f'{file_path} is not a valid pdb file or directory')
            
            self.files_name = [os.path.basename(f) for f in self.files]
            self.max_len = max_len
            
        def load(self):
            data = []; seqs = []
            for file in self.files:
                seqs.append(self.get_struc_seq(args.foldseek, file)['A'])
            coords, coord_mask, padding_mask, confidence = self.get_structure()
            for i in range(coords.size()[0]):
                    data.append((seqs[i], coords[i], coord_mask[i], padding_mask[i], confidence[i]))
            print(f'{len(data)} structures loaded.')
            return self.files_name, data                

        def get_struc_seq(self,
                          foldseek,
                          path,
                          chains: list = ["A"],
                          process_id: int = 0) -> dict:
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
                for  line in r:
                    desc, _, struc_seq = line.split("\t")[:3]
                    name_chain = desc.split(" ")[0]
                    chain = name_chain.replace(name, "").split("_")[-1]       
                    if chains is None or chain in chains:
                        if chain not in seq_dict:
                            seq_dict[chain] = struc_seq
            
            os.remove(tmp_save_path)
            os.remove(tmp_save_path + ".dbtype")
            return seq_dict

        def get_structure(self):
            '''
            Args:
            Returns:
                coords: torch.tensor, shape=(batch_size, L, 3, 3)
                coord_mask: torch.tensor, shape=(batch_size, L)
                padding_mask: torch.tensor, shape=(batch_size, L)
                confidence: torch.tensor, shape=(batch_size, L)
            '''
            def load_structure(fpath, chain=["A"]):
                """
                    Args:
                        fpath: filepath to either pdb or cif file
                        chain: the chain id or list of chain ids to load
                    Returns:
                        biotite.structure.AtomArray
                """
                if fpath.endswith('cif'):
                    with open(fpath) as fin:
                        pdbxf = pdbx.PDBxFile.read(fin)
                    structure = pdbx.get_structure(pdbxf, model=1)
                elif fpath.endswith('pdb'):
                    with open(fpath) as fin:
                        pdbf = pdb.PDBFile.read(fin)
                    structure = pdb.get_structure(pdbf, model=1)
                bbmask = filter_backbone(structure)
                structure = structure[bbmask]
                all_chains = get_chains(structure)
                if len(all_chains) == 0:
                    raise ValueError('No chains found in the input file.')
                if chain is None:
                    chain_ids = all_chains
                elif isinstance(chain, list):
                    chain_ids = chain
                else:
                    chain_ids = [chain] 
                for chain in chain_ids:
                    if chain not in all_chains:
                        raise ValueError(f'Chain {chain} not found in input file')
                chain_filter = [a.chain_id in chain_ids for a in structure]
                structure = structure[chain_filter]
                return structure
            
            def get_atom_coords_residuewise(atoms: List[str], struct: biotite.structure.AtomArray):
                """
                Args:
                    atoms: List of atom names to extract coordinates for, e.g., ["N", "CA", "C"].
                    struct: A biotite AtomArray representing the molecular structure.
                """
                def filterfn(s, axis=None):
                    """
                    Args:
                        s: biotite AtomArray
                        axis: None
                    Returns: coords
                    """
                    filters = np.stack([s.atom_name == name for name in atoms], axis=1)
                    sum_check = filters.sum(0)
                    if not np.all(sum_check <= np.ones(filters.shape[1])):
                        raise RuntimeError("The structure contains multiple atoms with the same name.")
                    first_occurrence_index = filters.argmax(0)
                    coords = s[first_occurrence_index].coord
                    coords[sum_check == 0] = float("nan")
                    
                    return coords

                return biotite.structure.apply_residue_wise(struct, struct, filterfn)

            def extract_coords_from_structure(structure: biotite.structure.AtomArray):
                """
                Args:
                    structure: An instance of biotite AtomArray
                Returns:
                    Tuple (coords, seq)
                        - coords is an L x 3 x 3 array for N, CA, C coordinates
                        - seq is the extracted sequence
                """
                coords = get_atom_coords_residuewise(["N", "CA", "C"], structure)
                residue_identities = get_residues(structure)[1]
                seq = ''.join([ProteinSequence.convert_letter_3to1(r) for r in residue_identities])
                return coords, seq
            
            confident = None
            raw_batch = []
            coords_max_shape = [self.max_len, 3, 3]
            confidence_max_shape = [self.max_len]
            for file in self.files:
                coords,seq = extract_coords_from_structure(load_structure(file))
                raw_batch.append((coords, confident, seq))
            alphabet = Alphabet.from_architecture("invariant_gvp")
            converter = CoordBatchConverter(alphabet)
            coords, coord_mask, padding_mask, confidence = converter(
                raw_batch=raw_batch,
                coords_max_shape=coords_max_shape,
                confidence_max_shape=confidence_max_shape)
            return coords, coord_mask, padding_mask, confidence
            
class PredictOutput:
    
    def __init__(self, num_classes: int, label2class: dict):
        self.num_class = num_classes
        self.label2class = label2class
        self.logits = torch.tensor([]).to(args.device)

    def update(self, logits):
        self.logits = torch.cat([self.logits, logits], dim=0)
    
    def get(self):
        print(self.logits)
        confidences = F.softmax(self.logits, dim=-1)
        print(confidences)
        predicted_classes = torch.argmax(confidences, dim=-1)
        results = []
        for class_id, confidence in zip(predicted_classes, confidences):
            confidence_value = confidence[class_id].item()
            results.append({'class': self.label2class[class_id.item()], 'confidence': confidence_value})
        return results

def predict_structure_origin(args):
    
    def dataloader(data):
        dataset = DatasetForCPE(data)
        return DataLoader(
            dataset=dataset,
            batch_size=len(dataset),
            num_workers=args.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        ) 

    def load_model():
        if not os.path.exists(args.model_weight_path):
            return ValueError(f"{args.model_weight_path} not found")
        model = CPEPro(args)
        # model.load_state_dict(torch.load(args.model_weight_path))
        pretrained_dict = torch.load(args.model_weight_path)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        model.eval()
        print(f'Model loaded, predicting...')
        return model.to(args.device)

    def save_output(results, files, output_file):
        with open(output_file, 'w') as f:
            f.write("Name\tOrigin\tConfidence\n")
            for file_name, result in zip(files, results):
                line = f"{file_name}\t{result['class']}\t{result['confidence']:.4f}\n"
                f.write(line)
        print(f'Output saved at {output_file}.')
    files, data = LoadData(args.file_path, args.sequence_max_length).load()
    model = load_model()
    output = PredictOutput(args.num_classes, args.label2Class)
    for batch in dataloader(data):
        output.update(model.predict(batch))
    save_output(output.get(), files, args.output_file)
           
if __name__ == '__main__':

    parser = ArgumentParser()

    # foldseek
    parser.add_argument('--foldseek', type=str, default=None,
                        help='Path to the foldseek executable.')
    
    # model
    parser.add_argument('--model_weight_path', type=str, default=None,
                        help='Path to the model weights.')
    parser.add_argument('--use_gvp', action='store_true', default=True,
                        help='Whether to use GVP (Geometric Vector Perceptron).')
    parser.add_argument('--use_sslm', action='store_true', default=True,
                        help='Whether to use SSLM (Self-Supervised Learning Model).')
    parser.add_argument('--freeze_lm', action='store_true', default=True,
                        help='Whether to freeze the language model.')
    
    # gvp
    parser.add_argument('--top_k_neighbors', type=int, default=3,
                        help='Number of top neighbors to consider.')
    parser.add_argument('--node_hidden_dim_scalar', type=int, default=128,
                        help='Hidden dimension scalar for nodes.')
    parser.add_argument('--node_hidden_dim_vector', type=int, default=128,
                        help='Hidden dimension vector for nodes.')
    parser.add_argument('--edge_hidden_dim_scalar', type=int, default=128,
                        help='Hidden dimension scalar for edges.')
    parser.add_argument('--edge_hidden_dim_vector', type=int, default=128,
                        help='Hidden dimension vector for edges.')
    parser.add_argument('--num_encoder_layers', type=int, default=3,
                        help='Number of encoder layers.')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate.')
    parser.add_argument('--embed_gvp_output_dim', type=int, default=128,
                        help='Output dimension of GVP embedding.')
    
    # sslm
    parser.add_argument('--load_pretrained', action='store_true', default=True,
                        help='Whether to load pretrained model.')
    parser.add_argument('--sslm_dir', type=str, default=None,
                        help='Directory containing SSLM data.')
    parser.add_argument('--sequence_max_length', type=int, default=256,
                        help='Maximum sequence length.')
    parser.add_argument('--linear_dropout', type=float, default=0.15,
                        help='Linear layer dropout.')
    
    # task
    parser.add_argument('--num_classes', type=int, default=4,
                        help='Number of classes in the classification task.')
    parser.add_argument('--label2Class', type=dict, default={0: 'CRYSTAL', 1: 'ALPHAFOLD', 2: 'OMEGAFOLD', 3: 'ESMFOLD'},
                        help='Mapping from labels to class names.')
    
    # validation
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run the model on.')
    parser.add_argument('--atten_pooling', action='store_true', default=True,
                        help='Whether to use attention pooling.')

    parser.add_argument('--file_path', type=str, help='Path to the input file', 
                        default=None)
    parser.add_argument('--output_file', type=str, help='Path to the output file', 
                        default=None)
    
    args = parser.parse_args()
    predict_structure_origin(args)