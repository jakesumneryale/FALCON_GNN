##############
# Created by Jacob Sumner for the purposes of writing his own code to modify
# the current version of GNN-DOVE
##############

## Imports
import numpy as np
import Bio
from Bio import PDB
from Bio.PDB import PDBParser
import os
from os import listdir
from os.path import isfile, join
import pickle
from bio_embeddings.embed import SeqVecEmbedder



letter_transfer = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
		'GLY': 'G', 'HIS': 'H', 'ILE': "I", 'LYS': "K", 'LEU': "L",
	   'MET': "M", 'ASN': "N", 'PRO': "P", 'GLN': "Q", 'ARG': "R",
	   'SER': "S", 'THR': "T", 'VAL': "V", 'TRP': "W", 'TYR': 'Y'}


def get_c_alpha_coords(protein_file):
	"""
	Creates a biopython protein object from
	the protein file and extracts the C-alpha coordinates
	from the file and returns them in a numpy array
	so that they can be used by the code.
	C-alpha coordinates come from each residue, so the 
	protein will be coarse-grained.
	"""
	global letter_transfer
	parser = PDBParser()
 
	pdb_name = protein_file.split("/")[-1].split(".")[0]
	structure  = parser.get_structure(pdb_name, protein_file)
	temp_seq = ""
	temp_coordlist = []
	for model in structure:
		chain_list = []
		for chain in model: 
			for residues in chain:
				count = 0
				if residues.get_resname() not in "UNK":
					temp_seq += letter_transfer[residues.get_resname()] ## appends the single letter AA code
				for atom in residues:
					if atom.get_name() in "CA":
						temp_coordlist.append(atom.get_coord())
						break
	return temp_seq, np.array(temp_coordlist)


def create_adjacency_matrix(protein_len):
	'''
	Because the protein model is coarse-grained to
	only the residues, the adjacency matrix is very simple to create
	and will be a heuristic to fill in the adjacency matrix
	based on the length of the protein. Self-edges will be
	included in the adjacency matrix
	'''

	adj_mat = np.zeros((protein_len, protein_len))
	for i in range(protein_len):
		for j in range(i,protein_len): ## only focus on half of the matrix to save loops
			if i == protein_len-1 and j == protein_len-1:
				## Break condition
				adj_mat[i,j] = 1
				break
			## They are the same node - self edge
			elif i==j:
				adj_mat[i,j] = 1
			## They are covalently bonded - edge at i,j and j,i - symmetric
			elif j==i+1:
				adj_mat[i,j] = 1
				adj_mat[j,i] = 1
	return adj_mat


def get_seqvec_features_for_protein(protein_sequence):
	'''
	Creates the SeqVec features for each of the 
	amino acids in the protein sequence. The size of the
	outputted vector will be (L, 1024), where L is the 
	length of the protein.
	'''

	embedder = SeqVecEmbedder()

	embedding = embedder.embed("SEQVENCE")

	return embedding 

def main():
	test_file = "/Users/jakesumner/Documents/GitHub/GNN_DOVE/example/input/correct.pdb"

	## Test get_c_alpha_coords

	temp_seq, temp_coords = get_c_alpha_coords(test_file)
	temp_adj = create_adjacency_matrix(len(temp_seq))
	temp_emb = get_seqvec_features_for_protein(temp_seq)
	potato = 1


if __name__ == '__main__':
	main()