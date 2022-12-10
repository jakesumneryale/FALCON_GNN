import os
from data_processing.Extract_Monomers import Extract_Monomers
from rdkit.Chem.rdmolfiles import MolFromPDBFile
from data_processing.Feature_Processing import get_atom_feature
import numpy as np
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from scipy.spatial import distance_matrix
import data_processing.Prepare_Input_Helper
from bio_embeddings.embed import SeqVecEmbedder
import torch


def get_seqvec_features_for_protein(protein_sequence, embedder):
    '''
    Creates the SeqVec features for each of the 
    amino acids in the protein sequence. The size of the
    outputted vector will be (L, 1024), where L is the 
    length of the protein.
    '''

    embedding = embedder.embed(protein_sequence)
    embedding = torch.tensor(embedding).sum(dim=0) # Tensor with shape [L,1024]
    return embedding

def Prepare_Input(structure_path, receptor_units, capri_rank = None):
    embedder = SeqVecEmbedder() 
    # extract the interface region
    root_path=os.path.split(structure_path)[0]
    receptor_path, ligand_path, receptor_seq, ligand_seq = Extract_Monomers(structure_path, receptor_units)
    receptor_mol = MolFromPDBFile(receptor_path, sanitize=False)
    ligand_mol = MolFromPDBFile(ligand_path, sanitize=False)
    receptor_count = receptor_mol.GetNumAtoms()
    ligand_count = ligand_mol.GetNumAtoms()

    ## Change the features to be SeqVec in the morning!
    receptor_feature = get_seqvec_features_for_protein(receptor_seq, embedder)
    ligand_feature = get_seqvec_features_for_protein(ligand_seq, embedder)

    # get receptor adj matrix
    c1 = receptor_mol.GetConformers()[0]
    d1 = np.array(c1.GetPositions())
    adj1 = GetAdjacencyMatrix(receptor_mol) + np.eye(receptor_count)
    # get ligand adj matrix
    c2 = ligand_mol.GetConformers()[0]
    d2 = np.array(c2.GetPositions())
    adj2 = GetAdjacencyMatrix(ligand_mol) + np.eye(ligand_count)
    # combine analysis
    H = np.concatenate([receptor_feature, ligand_feature], 0)
    agg_adj1 = np.zeros((receptor_count + ligand_count, receptor_count + ligand_count))
    agg_adj1[:receptor_count, :receptor_count] = adj1
    agg_adj1[receptor_count:, receptor_count:] = adj2  # array without r-l interaction
    dm = distance_matrix(d1, d2)
    agg_adj2 = np.copy(agg_adj1)
    agg_adj2[:receptor_count, receptor_count:] = np.copy(dm)
    agg_adj2[receptor_count:, :receptor_count] = np.copy(np.transpose(dm))  # with interaction array
    # node indice for aggregation
    valid = np.zeros((receptor_count + ligand_count,))
    valid[:receptor_count] = 1
    input_file=os.path.join(root_path,"Input.npz")
    # sample = {
    #     'H': H.tolist(),
    #     'A1': agg_adj1.tolist(),
    #     'A2': agg_adj2.tolist(),
    #     'V': valid,
    #     'key': structure_path,
    # }
    if capri_rank == None:
        np.savez(input_file,  H=H, A1=agg_adj1, A2=agg_adj2, V=valid)
    else: 
        np.savez(input_file,  H=H, A1=agg_adj1, A2=agg_adj2, V=valid)
    return input_file