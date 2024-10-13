import os
import sys
sys.path.append('../')
sys.path.append('../src')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from typing import Sequence, Tuple, List
import biotite
import biotite.structure
from biotite.structure.io import pdbx, pdb
from biotite.structure.residues import get_residues
from biotite.structure import filter_backbone
from biotite.structure import get_chains
from biotite.sequence import ProteinSequence
from esm.data import Alphabet
from argparse import ArgumentParser


from src.utils import CoordBatchConverter, seed_everything
from src.model.cpe import CPEPro
from src.dataset import DatasetForCPE
