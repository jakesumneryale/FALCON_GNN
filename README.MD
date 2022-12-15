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

** Note: the directory 'best_models' contains the PyTorch parameter files for the fold models I generated thorugh training and subsequently analyzed in the results section of the final paper.


If you have any questions about the code, it should be decently commented. Otherwise, feel free to email me and ask.


###################

USAGE

###################

In order to get the final results for the Dockground set 2 data as seen in the paper, you must first download the Dockground Set 2 data from my google drive, which can be done using this link: https://drive.google.com/file/d/126eDDuc-GihaI3_CiuwbXxZRV4cHr0lu/view?usp=sharing. If for some reason that doesn't work, please email me. 

Next, please make sure that you fork the code from my GitHub. The project can be found at this link: https://github.com/jakesumneryale/FALCON_GNN. You can set up the environment needed to run the code with the aptly named "FALCON_GNN_envrionment.yml" file included in the GitHub repository. 

When your environment is set up and the data is downloaded, please make sure to go to the main directory of the FALCON_GNN code and open up the "FALCON_GNN_Data_Analysis.ipynb file". Run the code blocks starting from the beginning until you get to the 'Actually testing the model section', and stop.

Scroll down to the section titled "Testing the model on the Dockground Set 2 Dataset" and run the first code block. Next, go to each of the code blocks with the comment at the top "## Jake Parameters 2..." and change the 'model_path' to the path on your machine to the .../GitHub/FALCON_GNN/best_model/<fold.pt> file. Then change the 'input_path' to wherever you downloaded and stored the Dockground set 2 data.

Once all of these factors are taken care of, you can run the code and reproduce the results!

**NOTE: I did not provide instructions to re-create the results for the first half of the results becuase the dataset is around 40 GB and that is too large. If needed, I can provide a link to that data and instructions on how to get those results. The only reason I didn't include it is because the dataset was so large it would make testing very cumbersome. Thank you.
