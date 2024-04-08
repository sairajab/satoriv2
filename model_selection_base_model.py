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

def setup_seed(seed):
    random.seed(seed)                          
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True 
    
#######create full code for the exeriment run and test, just one file and start writin a document#############
def Calibration(arg_space):
    print("Start")
    best_AUC = 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model_dir = arg_space.directory
    params = {}
    #learning_steps_list=[2000, 4000, 6000, 8000, 10000]
    params["input_channels"] = 4#number of original input channels (for DNA = 4)
    
    #################FIXED PARAMS####################
    params["use_embeddings"] =False#Use embeddings if True otherwise use one-hot encoded input
    params["embd_window"] =5#embedding window size
    params["embd_size"] =50#embedding size word2vec model
    params["embd_kmersize"] = 3#size of kmer in word2vec model
    params["use_pooling"] = False#Use pooling at single head level
    params["pooling_val"] = 4#if use pooling at single head level, size of pooling
    params["readout_strategy"] = "normalize"#read out layer type/strategy
    params["use_RNN"] = False#use RNN in the model
    params["RNN_hiddensize"] = 100#RNN hidden size
    params["use_CNN"] =True #use CNN layer
    params["use_posEnc"] = False#use positional encoding
    params["use_CNNpool"] = True#use CNN pooling
    params["CNN_poolsize"] = 4#CNN maxpool size
    params["CNN_padding"] = 0#CNN padding, need to be determined based on input length and filter size
    params["get_CNNout"] = True#get first CNN layer output (useful for motif analysis)
    params["get_seqs"] = True#get sequences for the test set examples (useful for motif analysis)
    params["get_pattn"] = True#get Attention value
    params["sqrt_dist"] = False##Sqrt of Relative Attention
    params["residual_connection"] = False##
    params['num_attnlayers'] =1 #number of attention layers
    params['mixed_attn'] = False
    params['mixed_attn_config'] = 0.25

    # entropy_reg_value|0.01#entropy regularization value
    epochs = [30, 40, 50, 60]
    attn_type = [True, False] #true for relative attn
    learnable_dist = [False] 
    entropy_loss = [True, False]
    
    for attn in attn_type:
        
        if not attn:
            learnable_dist = [False]
            
        for dist in learnable_dist:
            for entr in entropy_loss:
                for number in range(50):
                    # hyper-parameters
                    print(number)
                    
                    num_epoch = random.choice(epochs)
                    num_of_multiheads_list = [2, 4, 8]
                    single_headsize_list = [32, 64, 128]
                    multihead_size_list = [100, 200]
                    params['num_multiheads'] = random.choice(num_of_multiheads_list)
                    params['singlehead_size'] = random.choice(single_headsize_list)#SingleHeadSize
                    params['multihead_size'] = random.choice(multihead_size_list) #MultiHeadSize
                    params['CNN1_useexponential'] = random.choice([True, False])
                    CNN_filters_list = [[200] , [300], [400]]
                    CNN_filtersize_list = [13, 15, 17, 19]


                    params['CNN_filters'] = random.choice(CNN_filters_list)
                    first_cnn = random.choice(CNN_filtersize_list)
                    params['CNN_filtersize'] = [random.choice(CNN_filtersize_list)]
                    numLabels = 2
                    reuseWeightsQK = False
                    params['relativeAttn'] = attn
                    params['Learnable_relativeAttn'] = dist

                    params['batch_size']  = random.choice([32, 64, 128, 256])
                    params["entropy_loss"] = entr
                    params["entropy_reg_value"] = random.uniform(0.005, 0.05) #entropy regularization value

                    train_loader, train_indices, test_loader, test_indices, valid_loader, valid_indices, output_dir = load_datasets(arg_space, params['batch_size'])

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
                    print("Epochssss ", num_epoch)
                    for kk in range(3):
                        setup_seed(kk)
                        model = AttentionNet(arg_space, params, device=device).to(device)

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
                        loss = 0
                        
                        for ep in range(num_epoch):
                        
                            auc=[]
                            model.train()
                            for batch_idx, (headers, seqs, data, target)  in enumerate(train_loader):
                                
                                data = data.to(device, dtype=torch.float)
                                target = target.to(device, dtype=torch.long)

                                # Forward pass
                                outputs, PAttn = model(data)

                                if entr:
                                    epsilon = 1e-10
                                    PAttn = PAttn + epsilon
                                    attention_entropy = -torch.sum(PAttn * torch.log(PAttn), dim=-1)
                                    entropy_loss = attention_entropy.mean()
                                    loss = criterion(outputs, target) + (params["entropy_reg_value"]  * entropy_loss)
                                else:
                                    loss = criterion(outputs, target)
                                    
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()
                            learning_steps+=1
                            print("Loss ", loss)

                            if ep% 5==0:
        
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
                                            #if output.shape[0]>30:
                                            try:
                                                    auc.append(metrics.roc_auc_score(labels, pred[:,1]))
                                            except ValueError:
                                                    print('NaN found', params)
                                                    auc.append(0.0)
                                        print(ep, np.mean(auc))
                                        model_auc[kk].append(np.mean(auc))
                                        
                                        model.train()
                                    
                    
                    print('                   ##########################################               ')

                    for n in range(num_epoch//5):
                        AUC=(model_auc[0][n]+model_auc[1][n]+model_auc[2][n])/3
                        #print(AUC)
                        if AUC>best_AUC:
                            best_AUC=AUC
                            best_learning_steps=num_epoch
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
    
    print(params)
    
    best_hyperparameters = {'best_learning_steps': best_learning_steps,'best_LearningRate':best_LearningRate,'best_LearningMomentum':best_LearningMomentum,
             'best_num_multiheads':best_num_multiheads, 'best_singlehead_size':best_singlehead_size,'best_multihead_size':best_multihead_size,'best_use_pooling':best_use_pooling,
                             'best_weightDecay':best_weightDecay,'best_pooling_val':best_pooling_val,'best_use_CNN':best_use_CNN,'best_CNN_filters':best_CNN_filters,'best_optim':best_optim,
                             'best_CNN_filtersize':best_CNN_filtersize,'best_CNN_poolsize':best_CNN_poolsize, 'best_CNN_padding' : best_CNN_padding, 'best_batch_size': best_batch_size }
    torch.save(best_hyperparameters, model_dir+'best_hyperpamarameters.pth')
    return best_hyperparameters
