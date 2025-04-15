from satori.modelsold import AttentionNet
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
from argparse import ArgumentParser


warnings.filterwarnings("ignore")
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


def parseArgs():
    """Parse command line arguments
    
    Returns
    -------
    a : argparse.ArgumentParser
    
    """
    parser = ArgumentParser(description='Model Search script.')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', 
                        default=False, help="verbose output [default is quiet running]")
    parser.add_argument('-o','--outDir', dest='directory', type=str,
                        action='store', help="output directory", default='')
    parser.add_argument('-m','--mode', dest='mode', type=str,
                        action='store', help="Mode of operation: train or test.", default='train')     
    parser.add_argument('--deskload', dest='deskLoad',
                        action='store_true', default=False,
                        help="Load dataset from desk. If false, the data is converted into tensors and kept in main memory (not recommended for large datasets).")  
    parser.add_argument('-w','--numworkers', dest='numWorkers', type=int,
                        action='store', help="Number of workers used in data loader. For loading from the desk, use more than 1 for faster fetching.", default=1)        
    parser.add_argument('--splitperc', dest='splitperc', type=float, action='store',
                        help="Pecentages of test, and validation data splits, eg. 10 for 10 percent data used for testing and validation.", default=10)
    parser.add_argument('--motifanalysis', dest='motifAnalysis',
                        action='store_true', default=False,
                        help="Analyze CNN filters for motifs and search them against known TF database.")
    parser.add_argument('--filtersanalysis', dest='filterAnalysis',
                        action='store_true', default=False,
                        help="Analyze CNN filters for motifs based on annotation file.")
    parser.add_argument('--scorecutoff', dest='scoreCutoff', type=float,
                        action='store', default=0.65,
                        help="In case of binary labels, the positive probability cutoff to use.")
    parser.add_argument('--tomtompath', dest='tomtomPath',
                        type=str, action='store', default=None,
                        help="Provide path to where TomTom (from MEME suite) is located.") 
    parser.add_argument('--database', dest='tfDatabase', type=str, action='store',
                        help="Search CNN motifs against known TF database. Default is Human CISBP TFs.", default=None)
    parser.add_argument('--annotate',dest='annotateTomTom',type=str,action='store',
                        default=None, help="Annotate tomtom motifs. The value of this variable should be path to the database file used for annotation. Default is None.")                   
    parser.add_argument('-i','--interactions', dest='featInteractions',
                        action='store_true', default=False,
                        help="Self attention based feature(TF) interactions analysis.")
    parser.add_argument('--interactionanalysis', dest='interactionsAnalysis',
                        action='store_true', default=False,
                        help="interactions analysis with ground truth values")
    
    parser.add_argument('-b','--background', dest='intBackground', type=str,
                        action='store', default=None,
                        help="Background used in interaction analysis: shuffle (for di-nucleotide shuffled sequences with embedded motifs.), negative (for negative test set). Default is not to use background (and significance test).")
    parser.add_argument('--attncutoff', dest='attnCutoff', type=float,
                        action='store', default=0.04,
                        help="Attention cutoff value. For a given interaction, it should have an attention value at least as high as this value across all examples.") #In human promoter DHSs data analysis, lowering the cutoff leads to more TF interactions. 
    parser.add_argument('--fiscutoff', dest='fisCutoff', type=float,
                        action='store', default=0,
                        help="FIS score cutoff value. For a given interaction, it should have an FIS score at least as high as this value across all examples.") 
    parser.add_argument('--intseqlimit', dest='intSeqLimit', type=int,
                        action='store', default = -1,
                        help="A limit on number of input sequences to test. Default is -1 (use all input sequences that qualify).")
    parser.add_argument('-s','--store', dest='storeInterCNN',
                        action='store_true', default=False,
                        help="Store per batch attention and CNN outpout matrices. If false, the are kept in the main memory.")
    parser.add_argument('--numlabels', dest='numLabels', type=int,
                        action='store', default = 2,
                        help="Number of labels. 2 for binary (default). For multi-class, multi label problem, can be more than 2. ")
    parser.add_argument('--tomtomdist', dest='tomtomDist', type=str,
                        action='store', default = 'ed',
                        help="TomTom distance parameter (pearson, kullback, ed etc). Default is euclidean (ed). See TomTom help from MEME suite.")
    parser.add_argument('--tomtompval', dest='tomtomPval', type=float,
                        action='store', default = 0.05,
                        help="Adjusted p-value cutoff from TomTom. Default is 0.05.")
    parser.add_argument('--testall', dest='testAll',
                        action='store_true', default=False,
                        help="Test on the entire dataset (default False). Useful for interaction/motif analysis.")
    parser.add_argument('--useall', dest='useAll',
                        action='store_true', default=False,
                        help="Use all examples in multi-label problem instead of using precision based example selection.  Default is False.")
    parser.add_argument('--precisionlimit', dest='precisionLimit', type=float,
                        action='store', default = 0.50,
                        help="Precision limit to use for selecting examples in case of multi-label problem.")
    parser.add_argument('--attrbatchsize', dest='attrBatchSize', type=int,
                        action='store', default = 12,
                        help="Batch size used while calculating attributes for FIS scoring. Default is 12.")
    parser.add_argument('--method', dest='methodType', type=str,
                        action='store', default='SATORI',
                        help="Interaction scoring method to use; options are: SATORI, FIS, or BOTH. Default is SATORI.")
    parser.add_argument('inputprefix', type=str,
                        help="Input file prefix for the bed/text file and the corresponding fasta file (sequences).")

    parser.add_argument('--gt_pairs', dest='pairs_file',
                        action='store', default='',
                        help="Path to groud truth pairs file")
    parser.add_argument('--finetune_model', dest='finetune_model_path',
                        action='store', default=None,
                        help="Path to the pre-trained model ")
    parser.add_argument('--set_seed', dest='set_seed',
                        action='store_true', default=False,
                        help="Set seed or not")  

    parser.add_argument('--seed', dest='seed',
                        action='store', type=int, default=0,
                        help="Seed to intialize model")   
    parser.add_argument('--motifweights', dest='load_motif_weights',
                        action='store_true', default=False,
                        help="Load weights of first CNN from motifs PWM and freeze them, by default false hence weights are randomly intialized and trained.).")  

    args = parser.parse_args()

    return args


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
    dataset_name = "Simulated"
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
    
    params["CNN_padding"] = 0#CNN padding, need to be determined based on input length and filter size
    params["get_CNNout"] = True#get first CNN layer output (useful for motif analysis)
    params["get_seqs"] = True#get sequences for the test set examples (useful for motif analysis)
    params["get_pattn"] = True#get Attention value
    params["sqrt_dist"] = False##Sqrt of Relative Attention
    params["residual_connection"] = False##
    params['num_attnlayers'] =1 #number of attention layers
    params['mixed_attn'] = False
    params['mixed_attn_config'] = 0.25
    params['multiple_linear'] = False
    params['sei'] = False
    params['cnn_attention_block'] = False
    params['exp_name'] = dataset_name

    # entropy_reg_value|0.01#entropy regularization value
    epochs = [30, 40, 50, 60]
    attn_type = [False] #add true for relative attn
    learnable_dist = [False] 
    entropy_loss = [False] #True 
    
    for attn in attn_type:
        
        if not attn:
            learnable_dist = [False]
            
        for dist in learnable_dist:
            for entr in entropy_loss:
                for number in range(30):
                    # hyper-parameters
                    print(number)
                    
                    num_epoch = random.choice(epochs)
                    num_of_multiheads_list = [2, 4, 8]
                    single_headsize_list = [32, 64, 128]
                    multihead_size_list = [100, 200]
                    params['num_multiheads'] = random.choice(num_of_multiheads_list)
                    params['singlehead_size'] = random.choice(single_headsize_list)#SingleHeadSize
                    params['multihead_size'] = random.choice(multihead_size_list) #MultiHeadSize
                    params['CNN1_useexponential'] = random.choice([False, False])#no Exponential
                    CNN_filters_list = [[200] , [300], [400]]
                    CNN_filtersize_list = [13, 15, 17, 19]


                    params['CNN_filters'] = random.choice(CNN_filters_list)
                    first_cnn = random.choice(CNN_filtersize_list)
                    params['CNN_filtersize'] = [random.choice(CNN_filtersize_list)]
                    numLabels = 2
                    reuseWeightsQK = False
                    params['relativeAttn'] = attn
                    params['Learnable_relativeAttn'] = dist

                    params['batch_size']  = random.choice([32, 64, 128])
                    params["entropy_loss"] = entr
                    params["entropy_reg_value"] = random.uniform(0.005, 0.05) #entropy regularization value

                    optim_list=['SGD','Adagrad','Adam']

                    optim=random.choice(optim_list)
                    learning_rate=logsampler(0.005,0.05) 
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
                    list_ln = [512, 1024, 2048]
                    params['linear_layer_size'] = random.choice(list_ln)
                    params["CNN_poolsize"] = random.choice([4, 6, 8])#CNN maxpool size
                    # nummotif_list=[16]
                    # nummotif1=random.choice(nummotif_list)    
                    model_auc=[[],[],[]]
                    print(params) 
                    print("Epochssss ", num_epoch)
                    train_loader, train_indices, test_loader, test_indices, valid_loader, valid_indices, output_dir, seq_len = load_datasets(arg_space, params['batch_size'], dataset_name)#

                    for kk in range(3):
                        #setup_seed(kk)
                        model = AttentionNet(arg_space, params, device=device,seq_len=seq_len).to(device)
                        print(model)

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
                                                    #print('NaN found', params)
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


def hyperparameter_selection():
    
    arg_space = parseArgs()
    

    best_Res = Calibration(arg_space)

    print(best_Res)
    
if __name__ == "__main__":
    hyperparameter_selection()