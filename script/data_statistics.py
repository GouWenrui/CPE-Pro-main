from Bio import SeqIO

def count_sequences_in_fasta(file_path):
    with open(file_path, "r") as file:
        sequences = list(SeqIO.parse(file, "fasta"))
        return len(sequences)

def sequence_max_length(file_path):
    length_num = 0
    with open(file_path, "r") as file:
        sequences = list(SeqIO.parse(file, "fasta"))
        max_length = max(len(sequence.seq) for sequence in sequences)
        mean_length = sum(len(sequence.seq) for sequence in sequences) / len(sequences)
        length_distribution = {}
        for sequence in sequences:
            length = len(sequence.seq)
            if length > 256: 
                length_num += 1
            bin_key = (length // 50) * 50
            if bin_key not in length_distribution:
                length_distribution[bin_key] = 0
            length_distribution[bin_key] += 1
        most_common_length_range = max(length_distribution, key=length_distribution.get)
        return max_length, mean_length, most_common_length_range, length_num 

def count_pdb_files(directory):
    import os
    pdb_files = [file for file in os.listdir(directory) if file.endswith(".pdb")]
    return len(pdb_files)