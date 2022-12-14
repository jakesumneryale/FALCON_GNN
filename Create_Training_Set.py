## Created by Jake Sumner so the training data from Dockground 1.0 can be
## processed for training

from data_processing.Prepare_Input_Jake import Prepare_Input
import pickle
import os
from data_processing.Single_Dataset import Single_Dataset
from torch.utils.data import DataLoader
from data_processing.collate_fn import collate_fn_Jake

def get_capri_rank(rmsds):
    '''
    Takes the RMSDs and determines whether the model is
    acceptable or not according to CAPRI standards
    '''
    r_rmsd, l_rmsd, i_rmsd, fnat, fnon_nat = rmsds
    capri_rank = None
    if fnat < 0.1 or (l_rmsd > 10 and i_rmsd > 4.0):
        ## Unacceptable Capri Rank
        capri_rank = 0 
    else:
        ## Accetable Capri Rank
        capri_rank = 1
    return capri_rank

pdb_only_list = ["2bnq", "2btf"]

def get_capri_rank_2(pdb_name):
    '''
    Takes the RMSDs and determines whether the model is
    acceptable or not according to CAPRI standards
    '''
    rank = int(pdb_name[7])
    if rank > 3:
        ## Unacceptable Capri Rank
        capri_rank = 0 
    else:
        ## Accetable Capri Rank
        capri_rank = 1
    return capri_rank

pdb_only_list = ["2bnq", "2btf"]

def parse_through_all_files(dirloc, rmsd_dict):
    '''
    Parses through all files in the directories for the
    decoy datasets
    '''
    npz_filelist = []
    os.chdir(dirloc)
    dirs = os.walk(dirloc)
    for ele in dirs:
        dirname = ele[0]
        if ele[0].endswith("decoys") and not ele[0].endswith("_decoys"):
            ## This is the correct decoys directory
            os.chdir(ele[0]) ## change directory to correct one
            #folder_name = ele[0].split("/")[11]
            #rec_chains, lig_chains = [len(tmp) for tmp in folder_name.split("_")[1].split(":")]
            decoy_pdbs = sorted(ele[2])
            temp_decoy_dict = {}
            master_pdb_id = decoy_pdbs[0][:4]

            if master_pdb_id not in pdb_only_list:
                ## Skips PDBs that have already been processed
                continue

            ## Gets the length of the rec subunits from the file name
            rec_subunits = len(ele[0].split("/")[-2].split("_")[1])
            for decoy in decoy_pdbs[4:]:
                if decoy.endswith(".pdb"):
                    try:
                        structure_path = os.path.join(ele[0], decoy)
                        capri_rank = get_capri_rank(rmsd_dict[master_pdb_id][decoy.split(".")[0]])
                        npz_filelist.append(Prepare_Input(structure_path, rec_subunits, capri_rank = capri_rank))
                    except:
                        print(f"FAILED ON MODEL {master_pdb_id} {decoy}")
    return npz_filelist

def parse_through_all_files_2(dirloc):
    '''
    Parses through all files in the directories for the
    decoy datasets
    '''
    npz_filelist = []
    os.chdir(dirloc)
    dirs = os.walk(dirloc)
    for ele in dirs:
        decoy_pdbs = sorted(ele[2])
        temp_decoy_dict = {}

        ## Gets the length of the rec subunits from the file name
        rec_subunits = 1
        for decoy in decoy_pdbs:
            if decoy.endswith(".pdb"):
                try:
                    structure_path = os.path.join(ele[0], decoy)
                    capri_rank = get_capri_rank_2(decoy)
                    npz_filelist.append(Prepare_Input(structure_path, rec_subunits, capri_rank = capri_rank))
                except:
                    print(f"FAILED ON MODEL {decoy}")
                 
        
    return npz_filelist

'''
Make sure to make this program so that it can create the entire dataset along with labels that it stores
in the dataset - need to modify the collate_fn.py so that it can handle the label input as well.
Check the train function to see if there is anything vital there for guidance on how to include the
label.
'''

def main():
    decoy_path = r"/mnt/c/Users/jaket/Documents/GNN_DOVE_DATA/dockground_set_2"
    os.chdir(r"/mnt/c/Users/jaket/Documents/GNN_DOVE_DATA/")
    #rmsd_dict = pickle.load(open("dockground_decoys_rmsds.pickle", "rb"))
    npz_filelist = parse_through_all_files_2(decoy_path)

    ## Save the list to make a dataset out of later!
    os.chdir(decoy_path)
    file_obj = open("dockground_2_npz_list.pickle", "wb")
    pickle.dump(npz_filelist, file_obj)
    # dataset = Single_Dataset(npz_filelist)
    # dataloader = DataLoader(dataset, 1, shuffle=False,
    #                         num_workers=4,
    #                         drop_last=False, collate_fn=collate_fn_Jake)
    # potato = 1
     

if __name__ == '__main__':
    main()