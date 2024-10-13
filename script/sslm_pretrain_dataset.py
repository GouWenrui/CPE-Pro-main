import os
import pandas as pd
import biotite.structure.io as bsio
from tqdm import tqdm

def calc_plddt(dataset_input=None, pdb_file=None, out_file="plddt.csv", type="protein"):
    if dataset_input is not None:
        if type == "protein":
            out_info = {"pdb": [], "plddt": []}
        elif type == "residue":
            out_info = {"pdb": []}
        pdbs = os.listdir(dataset_input)
        print(f"Calculating pLDDT for {len(pdbs)} structures...")
        with tqdm(total=len(pdbs)) as pbar:
            for pdb in pdbs:
                pdb_file = os.path.join(dataset_input, pdb)
                struct = bsio.load_structure(pdb_file, extra_fields=["b_factor"])
                if type == "protein":
                    plddt = struct.b_factor.mean()
                    out_info["pdb"].append(pdb)
                    out_info["plddt"].append(plddt)
                elif type == "residue":
                    out_info["pdb"].append(pdb)
                    res_list = []
                    for res, plddt in zip(struct.res_id, struct.b_factor):
                        if res not in res_list:
                            if not out_info.get(res):
                                out_info[res] = []
                            res_list.append(res)
                            out_info[res].append(plddt)
                pbar.update(1)
        pd.DataFrame(out_info).to_csv(out_file, index=False)
    else:
        struct = bsio.load_structure(pdb_file, extra_fields=["b_factor"])
        if type == "protein":
            plddt = struct.b_factor.mean()
        elif type == "residue":
            res_list = []
            plddt = []
            for res, b_factor in zip(struct.res_id, struct.b_factor):
                if res not in res_list:
                    res_list.append(res)
                    plddt.append(b_factor)
        print(plddt)
        
def get_struc_seq(foldseek, path, chains: list = ["A"], process_id: int = 0):
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
                desc, _, struc_seq = line.split("\t")[:3]
                name_chain = desc.split(" ")[0]
                chain = name_chain.replace(name, "").split("_")[-1]
                if chains is None or chain in chains:
                    if chain not in seq_dict:
                        seq_dict[chain] = struc_seq

        os.remove(tmp_save_path)
        os.remove(tmp_save_path + ".dbtype")
        return seq_dict

def filter_by_plddt(csv_file, plddt_threshold=0.9):
    print(f"Filtering structures with pLDDT > {plddt_threshold}...")
    plddt_list = []
    df = pd.read_csv(csv_file)
    with tqdm(total=len(df)) as pbar:
        for i in range(len(df)):
            pdb = df.loc[i, "pdb"]
            plddt = df.loc[i, "plddt"]
            if plddt > plddt_threshold:
                plddt_list.append({"pdb": pdb, "plddt": plddt})
            pbar.update(1)
    
    plddt_df = pd.DataFrame(plddt_list)
    plddt_df.to_csv("plddt_filtered.csv", index=False)
    return len(plddt_list)

def struc_seq_fasta(pdbs_path, csv_file, out_file, foldseek):
    df = pd.read_csv(csv_file)
    pdbs = df["pdb"]
    
    headers = []
    sequences = []
    
    with tqdm(total=len(pdbs)) as pbar:
        for pdb in pdbs:
            struc_seq = get_struc_seq(
                foldseek=foldseek,
                path=os.path.join(pdbs_path, pdb),
                chains=["A"],
                process_id=0,
            )["A"]
            headers.append(f'>{pdb}')
            sequences.append(struc_seq)
            pbar.update(1)
    
    with open(out_file, "w") as w:
        for i in range(len(headers)):
            w.write(headers[i] + '\n' + sequences[i] + '\n')      

