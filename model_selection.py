from satori.models import AttentionNet
import torch.nn.functional as F
import csv
import math
import random
import gzip
import torch
from sklearn import metrics
import multiprocessing
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os
import argparse
import warnings
from satori.experiment import load_datasets


warnings.filterwarnings("ignore")
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


def dinucshuffle(sequence):
	b = [sequence[i:i+2] for i in range(0, len(sequence), 2)]
	random.shuffle(b)
	d = ''.join([str(x) for x in b])
	return d


def logsampler(a, b):
		x = np.random.uniform(low=0, high=1)
		y = 10**((math.log10(b)-math.log10(a))*x + math.log10(a))
		return y


def sqrtsampler(a, b):

		x = np.random.uniform(low=0, high=1)
		y = (b-a)*math.sqrt(x)+a
		return y

#######create full code for the exeriment run and test, just one file and start writin a document#############
def Calibration(arg_space):
	print("Start")
	best_AUC = 0
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print(device)
	model_dir = arg_space.directory

	#print(params)
	#---------test code-------#
	#for batch in train_loader:
	#    pdb.set_trace()
	#    print(batch[0][:10])
	#--------test code-------#
	# device='cpu'
	params = {}
	learning_steps_list=[2000, 4000, 6000, 8000, 10000]
	for number in range(50):
		# hyper-parameters
		print(number)
		num_of_multiheads_list = [1, 2, 4, 8]
		single_headsize_list = [16, 32, 64]
		multihead_size_list = [50, 100, 200]
		params['num_multiheads'] = random.choice(num_of_multiheads_list)
		params['singlehead_size'] = random.choice(single_headsize_list)#SingleHeadSize
		params['multihead_size'] = random.choice(multihead_size_list) #MultiHeadSize
		params['use_pooling'] = random.choice([False, False])
		params['pooling_val'] = random.choice([6, 4, 2])
		params['readout_strategy']  = "normalize"
		params['embd_kmersize'] = 3
		params['use_RNN'] = False
		params['use_CNN'] = random.choice([True, True])
		params['CNN1_useexponential'] = random.choice([True, False])
		params['use_posEnc'] = False
		params['use_CNNpool'] = random.choice([True, True])
		params['RNN_hiddensize'] = 100
		CNN_filters_list = [[200] , [200, 100], [200, 100, 50]]
		CNN_filtersize_list = [13, 15, 16, 20]
		CNN_poolsize_list = [6, 8, 10]
		CNN_padding_list = [6, 4]

		params['CNN_filters'] = random.choice(CNN_filters_list)
		first_cnn = random.choice(CNN_filtersize_list)
		params['CNN_filtersize'] = [(first_cnn - (2*i))  for i in range(len(params['CNN_filters']))]
		params['CNN_poolsize'] = random.choice(CNN_poolsize_list)
		params['CNN_padding'] = random.choice(CNN_padding_list)
		numLabels = 2
		reuseWeightsQK = False
		params['input_channels']  = 4#number of channels, one hot encoding
		params['relativeAttn'] = random.choice([False, True])
		params['sqrt_dist'] = random.choice([False, True])
		params['Learnable_relativeAttn'] = random.choice([False, True])
		params['mixed_attn'] = random.choice([False, True])  
		params['mixed_attn_config'] = random.choice([0.5, 0.25, 0.75]) 
		params['batch_size']  = random.choice([32, 64, 128, 256])
		params['num_attnlayers'] = 1

		train_loader, train_indices, test_loader, test_indices, valid_loader, valid_indices, output_dir = load_datasets(arg_space, params['batch_size'])


		# RNN_hidden_size_list=[20,50,80,100]
		# RNN_hidden_size=random.choice(RNN_hidden_size_list)
		# dropoutList=[0,0.15,0.3,0.45,0.6] 
		# dropprob=random.choice(dropoutList)
		# hidden_list=[True,False]
		# hidden=random.choice(hidden_list)
		# xavier_List=[True,True,False] 
		# xavier=random.choice(xavier_List)
		# hidden_size_list=[32,64]
		# hidden_size=random.choice(hidden_size_list)
		optim_list=['SGD','Adagrad','Adam']
		optim=random.choice(optim_list)
		learning_rate=logsampler(0.005,0.5) 
		momentum_rate=sqrtsampler(0.95,0.99)  
		# sigmaConv=logsampler(10**-6,10**-2)   
		# sigmaNeu=logsampler(10**-3,10**-1) 
		# sigmaRNN=logsampler(10**-4,10**-1) 
		weightDecay=logsampler(10**-10,10**-1) 
		criterion = nn.CrossEntropyLoss(reduction='mean')
		params["learning_rate"] = learning_rate
		params["optim"] = optim
		params["momentum_rate"] = momentum_rate
		params["weight_decay"] = weightDecay

		# nummotif_list=[16]
		# nummotif1=random.choice(nummotif_list)	
		model_auc=[[],[],[]]
		print(params) 
		for kk in range(3):
			model = AttentionNet(numLabels, params, device=device).to(device)

			#model = Network(nummotif1,motiflen,RNN_hidden_size,hidden_size,hidden,dropprob,sigmaConv,sigmaNeu,sigmaRNN,xavier).to(device)
			if optim=='SGD':
				optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=momentum_rate,nesterov=True
											,weight_decay=weightDecay)
			elif optim == 'Adam':
				optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
			else:
				optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate,weight_decay=weightDecay)

			# train_loader=train_dataloader[kk]
			# valid_loader=valid_dataloader[kk]
			
			
			learning_steps=0
			while learning_steps<=40000:
			   
				auc=[]
				model.train()
				for batch_idx, (headers, seqs, data, target)  in enumerate(train_loader):
					
					data = data.to(device, dtype=torch.float)
					target = target.to(device, dtype=torch.long)

					# Forward pass
					output, _ = model(data)          
					loss = criterion(output,target)
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()
					learning_steps+=1

					if learning_steps% 2000==0:
						
							
							
							
						with torch.no_grad():
							model.eval()
							auc=[]
							for j, (headers, seqs, data1, target1) in enumerate(valid_loader):
								data1 = data1.to(device, dtype=torch.float)
								target1 = target1.to(device, dtype=torch.long)
								
								# Forward pass
								output, _ = model(data1)
								softmax = torch.nn.Softmax(dim=1)
								pred = softmax(output)
								
								pred=output.cpu().detach().numpy()#.reshape(output.shape[0])
								labels=target1.cpu().numpy()#.reshape(output.shape[0])
								if output.shape[0]>60:
									try:
										auc.append(metrics.roc_auc_score(labels, pred[:,1]))
									except ValueError:
										print('NaN found', params)
										auc.append(0.0)
							print(np.mean(auc))
							model_auc[kk].append(np.mean(auc))
							
							model.train()
						
		
		print('                   ##########################################               ')

		for n in range(5):
			AUC=(model_auc[0][n]+model_auc[1][n]+model_auc[2][n])/3
			#print(AUC)
			if AUC>best_AUC:
				best_AUC=AUC
				best_learning_steps=learning_steps_list[n]
				best_LearningRate=learning_rate
				best_LearningMomentum=momentum_rate

				best_num_multiheads = params['num_multiheads']
				best_singlehead_size = params['singlehead_size'] 
				best_multihead_size = params['multihead_size']
				best_use_pooling = params['use_pooling'] 
				best_pooling_val = params['pooling_val'] 
				best_readout_strategy = params['readout_strategy']
				best_embd_kmersize = params['embd_kmersize'] 
				best_use_RNN = params['use_RNN'] 
				best_use_CNN = params['use_CNN']
				best_CNN1_useexponential = params['CNN1_useexponential']
				best_use_posEnc = params['use_posEnc'] 
				best_use_CNNpool = params['use_CNNpool'] 
				best_RNN_hiddensize = params['RNN_hiddensize']
				best_weightDecay=weightDecay
				best_optim=optim

				best_CNN_filters = params['CNN_filters']
				best_CNN_filtersize = params['CNN_filtersize'] 
				best_CNN_poolsize = params['CNN_poolsize'] 
				best_CNN_padding = params['CNN_padding']
				best_batch_size = params['batch_size']  


	print('best_AUC=',best_AUC)            
	print('best_learning_steps=',best_learning_steps)      
	print('best_LearningRate=',best_LearningRate)
	print('best_LearningMomentum=',best_LearningMomentum)
	print('best_num_multiheads=',best_num_multiheads)
	print('best_singlehead_size=',best_singlehead_size)
	print('best_multihead_size=',best_multihead_size)
	print('best_use_pooling',best_use_pooling)
	print('best_weightDecay=',weightDecay)
	print('best_pooling_val=',best_pooling_val)
	print('best_readout_strategy=',best_readout_strategy)
	print('best_use_CNN=',best_use_CNN)
	print('best_optim=',best_optim)
	print('best_CNN_filters=',best_CNN_filters)
	print('best_CNN_filtersize=',best_CNN_filtersize)
	print('best_CNN_poolsize=',best_CNN_poolsize)
	print('best_CNN_padding=',best_CNN_padding)
	print('best_use_CNNpool', best_use_CNNpool)
	print('best_batch_size=',best_batch_size)
	
	best_hyperparameters = {'best_learning_steps': best_learning_steps,'best_LearningRate':best_LearningRate,'best_LearningMomentum':best_LearningMomentum,
			 'best_num_multiheads':best_num_multiheads, 'best_singlehead_size':best_singlehead_size,'best_multihead_size':best_multihead_size,'best_use_pooling':best_use_pooling,
							 'best_weightDecay':best_weightDecay,'best_pooling_val':best_pooling_val,'best_use_CNN':best_use_CNN,'best_CNN_filters':best_CNN_filters,'best_optim':best_optim,
							 'best_CNN_filtersize':best_CNN_filtersize,'best_CNN_poolsize':best_CNN_poolsize, 'best_CNN_padding' : best_CNN_padding, 'best_batch_size': best_batch_size }
	torch.save(best_hyperparameters, model_dir+'best_hyperpamarameters.pth')
	return best_hyperparameters