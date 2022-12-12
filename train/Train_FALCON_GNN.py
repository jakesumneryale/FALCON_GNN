## Created for training by Jake Sumner for the CPSC 583 final project
## Training methods will be used in the model, but will be tweaked
## To fit the desires of the project

import os
import torch
import torch.optim as optim
import torch.nn as nn
# from torch.utils.data import random_split
from torch.utils.data import DataLoader
import numpy
from os import listdir
from os.path import isfile, join
from data_processing.collate_fn import collate_fn_Jake
from time import time
from data_processing.Single_Dataset import Single_Dataset
from ops.random_split import random_split
from model.GNN_Model import GNN_Model
import pickle


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

def init_model(params):
	model = GNN_Model(params)
	print('	Total params: %.10fM' % (count_parameters(model) / 1000000.0))
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = initialize_model(model, device)
	# state_dict = torch.load(model_path, map_location = device)
	# model.load_state_dict(state_dict['state_dict'])
	model.eval()
	return model,device

def Train_Model(dataloader,device,model, loss_fn, optimizer):
	total_loss = 0
	model.train()

	for batch_idx, sample in enumerate(dataloader):
		optimizer.zero_grad()
		H, A1, A2, V, Y, Atom_count = sample
		batch_size = H.size(0)
		H, A1, A2, V, Y = H.to(device), A1.to(device), A2.to(device), V.to(device), Y.to(device)
		pred = model.train_model((H, A1, A2, V, Atom_count), device)
		# pred1 = pred.detach().cpu().numpy()
		# Final_pred += list(pred1)
		loss = loss_fn(pred, Y)
		loss.backward()
		optimizer.step()
		total_loss += float(loss)
	return total_loss



def train_falcon_gnn(input_path, params):
	'''
	Trains the GNN according to Jake's parameters.
	FALCON_GNN stands for Fucking Awesome Linking
	Cohort Of Nottingham, which I just made up
	'''
	## Get all the NPZ files from the input_path

	list_npz = [f for f in listdir(input_path) if isfile(join(input_path, f)) and f.endswith(".npz")]
	fold1_label = ['1a2k', '1e96', '1he1', '1he8', '1wq1', '1f6m', '1ma9', '2btf', '1g20', '1ku6', '1t6g', '1ugh', '1yvb', '2ckh', '3pro']
	fold2_label = ['1akj', '1p7q', '2bnq', '1dfj', '1nbf', '1r4m', '1xd3', '2bkr', '1gpw', '1hxy', '1u7f', '1uex', '1zy8', '2goo', '1ewy']
	fold3_label = ['1avw', '1bth', '1bui', '1cho', '1ezu', '1ook', '1oph', '1ppf', '1tx6', '1xx9', '2fi4', '2kai', '1r0r', '2sni', '3sic']
	fold4_label = ['1bvn', '1tmq', '1f51', '1fm9', '1a2y', '1g6v', '1gpq', '1jps', '1wej', '1l9b', '1s6v', '1w1i', '2a5t', '3fap']

	train_data_list = []
	test_data_list = []
	for file in list_npz:
		if file[:4] in fold1_label or file[:4] in fold3_label or file[:4] in fold4_label:
			train_data_list.append(file)
		elif file[:4] in fold2_label:
			test_data_list.append(file)

	train_data = Single_Dataset(train_data_list)
	test_data = Single_Dataset(test_data_list)
	#train_data, test_data = random_split(dataset, [0.75, 0.25])

	BATCH_SIZE = 10

	train_loader = DataLoader(train_data, BATCH_SIZE, shuffle=False,
							num_workers=params['num_workers'],
							drop_last=False, collate_fn=collate_fn_Jake)

	test_loader = DataLoader(test_data, BATCH_SIZE, shuffle=False,
							num_workers=params['num_workers'],
							drop_last=False, collate_fn=collate_fn_Jake)

	## Initialize Model

	model, device = init_model(params)

	learning_rate = 5e-5
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	loss_fn = nn.BCELoss()

	EPOCHS = 30  # number of training epoch

	best_valid_loss = float('inf')

	os.chdir(input_path)
	train_loss_list = []

	for epoch in range(EPOCHS):

		start_time = time()  # record the start time

		#train_loss = train(model, train_loader, optimizer, loss_fn)
		train_loss = Train_Model(train_loader,device,model, loss_fn, optimizer)

		end_time = time()

		epoch_time = end_time - start_time

		print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_time}s')
		print(f'\tTrain Loss: {train_loss:.3f}')
		train_loss_list.append(train_loss)
	 
	os.chdir('/mnt/c/Users/jaket/Documents/GNN_DOVE_DATA')
	torch.save(model.state_dict(), 'full_train_DG1_random_batch_jake_params_3.pt')  # saving model's parameters
	pickle.dump(train_loss_list, open("full_train_DG1_random_batch_jake_params_3_loss.pickle", "wb"))
	#torch.save(test_loader, 'full_train_DG1_random_batch_jake_params_2_test_data.dl')




