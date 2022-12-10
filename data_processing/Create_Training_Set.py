## Created by Jake Sumner so the training data from Dockground 1.0 can be
## processed for training

from data_processing.Prepare_Input_Jake import Prepare_Input


def parse_through_all_files(dirloc, rmsd_dict):
    '''
    Parses through all files in the directories for the
    decoy datasets
    '''
    structure_master = {}
    os.chdir(dirloc)
    dirs = os.walk(dirloc)
    parser = PDBParser()
    for ele in dirs:
        dirname = ele[0]
        if ele[0].endswith("decoys"):
            ## This is the correct decoys directory
            os.chdir(ele[0]) ## change directory to correct one
            #folder_name = ele[0].split("/")[11]
            #rec_chains, lig_chains = [len(tmp) for tmp in folder_name.split("_")[1].split(":")]
            master_pdb_id = ele[0][100:104]
            decoy_pdbs = ele[2]
            temp_decoy_dict = {}
            for decoy in decoy_pdbs:
                pdb_name = decoy.split(".")[0]
                structure  = parser.get_structure(pdb_name, decoy)
                temp_reslist = []
                temp_coordlist = []
                for model in structure:
                    chain_list = []
                    for chain in model:
                        for residues in chain:
                            count = 0
                            if residues.get_resname() not in "UNK":
                                temp_reslist.append(residues.get_resname()) ## appends the single letter AA code
                            for atom in residues:
                                if atom.get_name() in "CA":
                                    temp_coordlist.append(atom.get_coord())
                                    break
                temp_decoy_dict[pdb_name] = [temp_reslist, np.array(temp_coordlist)]
            structure_master[master_pdb_id] = temp_decoy_dict
    return structure_master