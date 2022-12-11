## Created for training by Jake Sumner for the CPSC 583 final project
## Training methods will be used in the model, but will be tweaked
## To fit the desires of the project

import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import random_split
import numpy
from os import listdir
from os.path import isfile, join
from data_processing.collate_fn import collate_fn_Jake

def initialize_model(model, device, load_save_file=False):
    if load_save_file:
        model.load_state_dict(torch.load(load_save_file))
    else:
        for param in model.parameters():
            if param.dim() == 1:
                continue
                nn.init.constant(param, 0)
            else:
                #nn.init.normal(param, 0.0, 0.15)
                nn.init.xavier_normal_(param)

    if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
      model = nn.DataParallel(model)
    model.to(device)
    return model

def init_model(model_path,params):
    model = GNN_Model(params)
    print('    Total params: %.10fM' % (count_parameters(model) / 1000000.0))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = initialize_model(model, device)
    state_dict = torch.load(model_path, map_location = device)
    model.load_state_dict(state_dict['state_dict'])
    model.eval()
    return model,device

def Train_Model(dataloader,device,model, loss_fn, optimizer):
	Final_pred = []
    for batch_idx, sample in enumerate(dataloader):
        H, A1, A2, V, Y, Atom_count = sample
        batch_size = H.size(0)
        H, A1, A2, V, Y = H.to(device), A1.to(device), A2.to(device), V.to(device), Y.to(device)
        pred= model.train_model((H, A1, A2, V, Y, Atom_count), device)
        pred1 = pred.detach().cpu().numpy()
        Final_pred += list(pred1)
    return Final_pred


def train_falcon_gnn(input_path, params):
	'''
	Trains the GNN according to Jake's parameters.
	FALCON_GNN stands for Fucking Awesome Linking
	Cohort Of Nottingham, which I just made up
	'''
	learning_rate = 1e-4
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	loss_fn = nn.CrossEntropyLoss()

	## Get all the NPZ files from the input_path

	list_npz = [f for f in listdir(input_path) if isfile(join(input_path, f)) and if f.endswith(".npz")]

    dataset = Single_Dataset(list_npz)
    train_data, test_data = random_split(dataset, [0.75, 0.25])
    train_loader = DataLoader(dataset, 200, shuffle=False,
                            num_workers=params['num_workers'],
                            drop_last=False, collate_fn=collate_fn_Jake)

    test_loader = DataLoader(dataset, 200, shuffle=False,
                            num_workers=params['num_workers'],
                            drop_last=False, collate_fn=collate_fn_Jake)


