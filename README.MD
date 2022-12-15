#################

Read Me File for Jake Sumner's Final Project for CPSC 583

#################

For the final project of the class, I forked the DOVE-GNN code from the Kihara Lab, the original authors of the code. I then proceeded to heavily edit much of the code for the final project so that I would work for what I was trying to do. Similarly, I created many new files in the process, both for scripting and data analysis.

A list of the programs that I edited extensively:

data_processing/collate_fn.py (used for the dataloader)
data_processing/Extract_Monomers.py (used for generating the dataset from the protein structures)
data_processing/Prepare_Input_Helper.py (created in an abandoned attempt at generating the data)
data_processing/Prepare_Input_Jake.py (modified extensively to allow me to generated the input data for FALCON-GNN)
model/GNN_Model.py (modified in order to adapt the GNN model and its parameters to what I needed to do to get the best model peformance)
ops/argparser.py (minorly modified to add an option for inputting receptor subunits)
ops/random_split.py (added in from the PyTorch Source code to allow for random split testing for initial model assessment)
train/Train_FALCON_GNN.py (the script I created to train FALCON-GNN. I iteratively changed it to adopt different model architectures).
Create_Training_Set.py (created to generate the training set data that I ended up using for all of my training and testing)
FALCON_GNN_Data_Analysis.ipynb (Jupyter Notebook I created to analyze the data for the write up of the final project)
FALCON_GNN_envrionment.yml (environment file so that anyone can use the same environment I did to run the code)
GNN Course Final Project Dataset Analysis.ipynb (A notebook I created to do some preliminary analysis on the datasets ultimately used in the final project)
main.py (modified so that I could run the code I wrote myself and train FALCON-GNN easily from the command line)


If you have any questions about the code, it should be decently commented. Otherwise, feel free to email me and ask.


###################

USAGE

###################

In order to get the final results for the Dockground set 2 data as seen in the paper, one
