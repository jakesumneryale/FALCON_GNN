'''
Created by Jacob Sumner for the purposes of writing his own code to modify
the current version of GNN-DOVE
'''

## Imports
import numpy as np
import pandas as pd
import Bio
from Bio import PDB
from Bio.PDB import PDBParser
import matplotlib.pylab as plt
import os
from os import listdir
from os.path import isfile, join
from pypdb import *
import pickle


def get_c_alpha_coords(protein_file):
	'''
	Creates a biopython protein object from
	the protein file and extracts the C-alpha coordinates
	from the file and returns them in a numpy array
	so that they can be used by the code.
	C-alpha coordinates come from each residue, so the 
	protein will be coarse-grained.
	'''

	pass


def create_adjacency_matrix(protein_len):
	'''
	Because the protein model is coarse-grained to
	only the residues, the adjacency matrix is very simple to create
	and will be a heuristic to fill in the adjacency matrix
	based on the length of the protein. Self-edges will be
	included in the adjacency matrix
	'''

	pass


def get_seqvec_features_for_protein(protein_sequence):
	'''
	Creates the SeqVec features for each of the 
	amino acids in the protein sequence. The size of the
	outputted vector will be (L, 1024), where L is the 
	length of the protein.
	'''

	pass