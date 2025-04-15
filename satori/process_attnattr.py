import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch
import time
from Bio.SeqUtils import gc_fraction as GC
from multiprocessing import Pool
from scipy.stats import mannwhitneyu
from sklearn import metrics
from statsmodels.stats.multitest import multipletests
from torch.backends import cudnn
#local imports
from satori.models import AttentionNet
from satori.utils import get_popsize_for_interactions, get_intr_filter_keys
from satori.process_attention import analyze_interactions
import torch.nn.functional as F
import torch.nn as nn
### Global variables ###
Filter_Intr_Keys = None
#tp_pos_dict = None
#######################

class ModelWrapper(nn.Module):
    def __init__(self, original_model):
        super(ModelWrapper, self).__init__()
        self.original_model = original_model

    def forward(self, x, attr= None, targets=None):
        logits, _ , PAttn = self.original_model(x, attr)
        if targets is not None:
            # If targets are provided, filter the outputs as needed
            #print(logits[0,:])
            logits = logits * targets
        #print(targets[0,:],logits[0,:])
        logits = torch.sum(logits, dim = 1)
        return logits, PAttn



def get_filters_in_individual_seq(sdata):
	header,num_filters,filter_data_dict,CNNfirstpool = sdata
	s_info_dict = {}
	for j in range(0,num_filters):
		filter_data = filter_data_dict['filter'+str(j)] #np.loadtxt(motif_dir+'/filter'+str(j)+'_logo.fa',dtype=str)
		for k in range(0,len(filter_data),2):
			hdr = filter_data[k].split('_')[0]
			if hdr == header:
				pos = int(filter_data[k].split('_')[-2]) #-2 because the format is header_num_pos_activation
				pooled_pos = int(pos/CNNfirstpool)
				key = pooled_pos#header+'_'+str(pooled_pos)
				if key not in s_info_dict:
					s_info_dict[key] = ['filter'+str(j)]
				else:
					if 'filter'+str(j) not in s_info_dict[key]:
						s_info_dict[key].append('filter'+str(j))
	return {header: s_info_dict}


def get_filters_in_seq_dict(all_seqs,motif_dir,num_filters,CNNfirstpool,numWorkers=1):
	filter_data_dict = {}
	for i in range(0,num_filters):
		filter_data = np.loadtxt(motif_dir+'/filter'+str(i)+'_logo.fa',dtype=str)
		filter_data_dict['filter'+str(i)] = filter_data
	seq_info_dict = {}
	sdata = []
	for i in range(0,all_seqs.shape[0]):
		header = all_seqs[i][0]
		sdata.append([header,num_filters,filter_data_dict,CNNfirstpool])
	with Pool(processes = numWorkers) as pool:
		result = pool.map(get_filters_in_individual_seq,sdata,chunksize=1)
		for subdict in result:
			seq_info_dict.update(subdict)
	return seq_info_dict

def tiana_compute_integrated_gradients(net, data, PAttn, m=20, numlabels=2, target=None):
    # Shape of PAttn is now (128, 4, 145, 145)
    batch_size, num_heads, height, width = PAttn.shape
    batch_att = PAttn

    # Initialize grad_tensor (list of tensors for each head)
    grad_tensor = [torch.zeros_like(batch_att[:, h], requires_grad=False) for h in range(num_heads)]

    # Initialize baseline
    baseline = [torch.zeros_like(batch_att[:, h]) for h in range(num_heads)]

    # Loop over the range of m
    for k in range(m):
        # Compute the scaled input for each head
        input_att = [baseline[h] + ((k / m) * (batch_att[:, h] - baseline[h])) for h in range(num_heads)]
        
        # Ensure input_att is watched for gradients
        for h in range(num_heads):
            input_att[h].requires_grad_(True)  # Watch the gradient for each head's attention tensor

        if numlabels == 2:
            outputs, _, PAttn = net(data, input_att)
            pred = torch.unbind(outputs[torch.arange(target.size(0)), target])
        else:
            outputs, PAttn = net(data, input_att, target)
            pred = torch.unbind(outputs)

        # Accumulate the gradients for each head
        for h in range(num_heads):
            gradient = torch.autograd.grad(pred, PAttn[:, h], retain_graph=True)[0]
            grad_tensor[h] = grad_tensor[h] + gradient

    # Compute the attribution tensor for each head
    attr = [(grad_tensor[h] * ((batch_att[:, h] - baseline[h]) / m)) for h in range(num_heads)]

    # Concatenate the attention results across heads
    PAttn_all = torch.cat(attr, dim=1).cpu().detach().numpy()

    return PAttn_all


def compute_integrated_gradients(net, data, PAttn, m=20,numlabels=2, target=None):
    number_of_heads = len(PAttn)
    batch_att = PAttn

    # Initialize grad_tensor
    grad_tensor = [torch.zeros_like(PAttn[0], requires_grad=False) for _ in range(number_of_heads)]

    # Initialize baseline
    baseline = [torch.zeros_like(PAttn[0]) for _ in range(number_of_heads)]

    # Loop over the range of m
    for k in range(m):
        input_att = [baseline[h] + ((k / m) * (batch_att[h] - baseline[h])) for h in range(number_of_heads)]
        for h in range(number_of_heads):
            input_att[h].requires_grad_(True)  # Ensure input_att is watched for gradients

        if numlabels == 2:
            outputs, _, PAttn = net(data, input_att)
            pred = torch.unbind(outputs[torch.arange(target.size(0)), target])

        else:
            outputs,PAttn = net(data, input_att,target)
            pred = torch.unbind(outputs)
            
        for h in range(number_of_heads):
            gradient = torch.autograd.grad(pred, PAttn[h], retain_graph=True)[0]
            grad_tensor[h] = grad_tensor[h] + gradient

    # Compute attr_tensor
    attr = [(grad_tensor[h] * ((batch_att[h] - baseline[h]) / m)) for h in range(number_of_heads)]
    PAttn_all = torch.cat(attr, dim=1).cpu().detach().numpy()
    #print(np.asarray([i>=0.5 for i in pred_out[0,:].cpu()]))


    return PAttn_all

def evaluateRegularBatchAttnAttr(net, device, iterator, criterion, out_dirc, getAttnAttr = None):
    running_loss = 0.0
    valid_auc = []
    net.eval()
    CNNlayer = net.layer1[0:3]  # first conv layer without the maxpooling part
    CNNlayer.eval()
    roc = np.asarray([[], []]).T
    PAttn_all = {}
    per_batch_labelPreds = {}
    per_batch_CNNoutput = {}
    per_batch_testSeqs = {}
    per_batch_info = {}
    i = 0 
    with torch.enable_grad():
        for batch_idx, (headers, seqs, data, target) in enumerate(iterator):
            data, target = data.to(device, dtype=torch.float), target.to(
                    device, dtype=torch.long)
            outputs,PAttnTensor, PAttn = net(data)  # dist           
            loss = criterion(outputs, target)             
            softmax = torch.nn.Softmax(dim=1)
            labels=target.cpu().numpy()
            pred_out = softmax(outputs)
            start_time = time.time() 
			#For tiana
			#PAttn_all[batch_idx] = tiana_compute_integrated_gradients(net, data, PAttn, target=target)
   
            PAttn_all[batch_idx] = compute_integrated_gradients(net, data, PAttn, target=target)
            end_time = time.time() 

            print("Time Taken: %d seconds"%round(end_time-start_time))

            if i == 0 :
                output_dir = out_dirc
                if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                with open(output_dir+'/PAttn-'+str(batch_idx)+'.pckl', 'wb') as f:
                            pickle.dump(PAttnTensor.cpu().detach().numpy(), f)

                with open(output_dir+'/PAttr-'+str(batch_idx)+'.pckl', 'wb') as f:
                            pickle.dump(PAttn_all[batch_idx], f)
                            
                with open(output_dir+'/predictions-'+str(batch_idx)+'.pckl', 'wb') as f:
                            pickle.dump(pred_out.cpu().detach().numpy(), f)
                i = i + 1
                    
            pred = pred_out.cpu().detach().numpy()
            label_pred = np.column_stack((labels, pred[:, 1]))
            per_batch_labelPreds[batch_idx] = label_pred
            roc = np.row_stack((roc, label_pred))
            try:
                valid_auc.append(metrics.roc_auc_score(labels, pred[:, 1]))
            except:
                valid_auc.append(0.0)
            running_loss += loss.item()
            outputCNN = CNNlayer(data).cpu().detach().numpy()
            outputCNN = CNNlayer(data)

            per_batch_CNNoutput[batch_idx] = outputCNN.cpu(
                        ).detach().numpy()
            per_batch_testSeqs[batch_idx] = np.column_stack(
                        (headers, seqs))  

    labels = roc[:, 0]
    preds = roc[:, 1]
    valid_auc = metrics.roc_auc_score(labels, preds)
    return running_loss/len(iterator), valid_auc, roc, PAttn_all, per_batch_labelPreds, per_batch_CNNoutput, per_batch_testSeqs

	
	
def evaluateRegularBatchMCAttnAttr(net,  net_wrapper,device, iterator, criterion, out_dirc, getAttnAttr = None):
    running_loss = 0.0
    valid_auc = []
    net.eval()
    roc = np.asarray([[], []]).T
    PAttn_all = {}
    all_labels = []
    all_preds = []
    running_loss = 0.0
    valid_auc = []
    net.eval()
    CNNlayer = net.layer1[0:3]  # first conv layer without the maxpooling part
    CNNlayer.eval()
    roc = np.asarray([[], []]).T
    PAttn_all = {}
    per_batch_labelPreds = {}
    per_batch_CNNoutput = {}
    per_batch_testSeqs = {}
    per_batch_info = {}
    i = 0 
    with torch.enable_grad():
        for batch_idx, (headers, seqs, data, target) in enumerate(iterator):
            data, target = data.to(device, dtype=torch.float), target.to(
                    device, dtype=torch.float)
            outputs,PAttnTensor, PAttn = net(data)  # dist
            loss = criterion(outputs, target)
            labels=target.cpu().numpy()
            sigmoid = torch.nn.Sigmoid()
            pred_out = sigmoid(outputs)
            start_time = time.time()
            #For tiana
            PAttn_all[batch_idx] = tiana_compute_integrated_gradients(net, data, PAttn, target=target)
   
            #PAttn_all[batch_idx] = compute_integrated_gradients(net_wrapper, data, PAttn, numlabels=100, target=target)
            end_time = time.time() 
            print("Time Taken: %d seconds"%round(end_time-start_time))
            if i == 0 :
                    output_dir = out_dirc
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    with open(output_dir+'/PAttn-'+str(batch_idx)+'.pckl', 'wb') as f:
                            pickle.dump(PAttnTensor.cpu().detach().numpy(), f)

                    with open(output_dir+'/PAttr-'+str(batch_idx)+'.pckl', 'wb') as f:
                            pickle.dump(PAttn_all[batch_idx], f)
                            
                    with open(output_dir+'/predictions-'+str(batch_idx)+'.pckl', 'wb') as f:
                            pickle.dump(pred_out.cpu().detach().numpy(), f)
                    with open(output_dir+'/labels-'+str(batch_idx)+'.pckl', 'wb') as f:
                            pickle.dump(labels, f)        
                            
                    i = i + 1
                    
            pred = pred_out.cpu().detach().numpy()
            all_labels += labels.tolist()
            all_preds += pred.tolist()
            label_pred = {'labels': labels, 'preds': pred}
            per_batch_labelPreds[batch_idx] = label_pred
            running_loss += loss.item()
            outputCNN = CNNlayer(data).cpu().detach().numpy()
            outputCNN = CNNlayer(data)

            per_batch_CNNoutput[batch_idx] = outputCNN.cpu(
                        ).detach().numpy()
            per_batch_testSeqs[batch_idx] = np.column_stack(
                        (headers, seqs))  
            
    return running_loss/len(iterator), valid_auc, roc, PAttn_all, per_batch_labelPreds, per_batch_CNNoutput, per_batch_testSeqs


def score_individual_head(data):
	count,PAttn,header,seq_inf_dict,k,ex,params,tomtom_data,attn_cutoff,sequence_len,CNNfirstpool,num_filters,motif_dir,feat_size = data
	global Filter_Intr_Keys
	
	filter_Intr_Attn = np.ones(len(Filter_Intr_Keys))*-1
	filter_Intr_Pos = np.ones(len(Filter_Intr_Keys)).astype(int)*-1
	
	y_ind = count#(k*params['batch_size']) + ex
	attn_mat = abs(PAttn[ex,:,:])
	
	attn_mat = np.asarray([attn_mat[feat_size*i:feat_size*(i+1), :] for i in range(0,params['num_multiheads'])]) 
	attn_mat = np.max(attn_mat, axis=0) #out of the 8 attn matrices, get the max value at the corresponding positions
	for i in range(0, attn_mat.shape[0]):
		if i not in seq_inf_dict:
			continue
		for j in range(0, attn_mat.shape[1]):
			if j not in seq_inf_dict:
				continue
			if i==j:
				continue
			max_loc = [i,j]#attn_mat[i,j]
			
			pos_diff = CNNfirstpool * abs(max_loc[0]-max_loc[1])
			
			KeyA = i #seq_inf_dict already is for the current header and we just need to specify the Pooled position
			KeyB = j 
			
			attn_val = attn_mat[i,j]
			
			all_filters_posA = seq_inf_dict[KeyA]
			all_filters_posB = seq_inf_dict[KeyB]
			
			for keyA in all_filters_posA:
				for keyB in all_filters_posB:
					if keyA == keyB:
						continue
					intr = keyA+'<-->'+keyB
					rev_intr = keyB+'<-->'+keyA
					if intr in Filter_Intr_Keys:
						x_ind = Filter_Intr_Keys[intr]
					elif rev_intr in Filter_Intr_Keys:
						x_ind = Filter_Intr_Keys[rev_intr]

					if attn_val > filter_Intr_Attn[x_ind]:#[y_ind]:
						filter_Intr_Attn[x_ind] = attn_val #[y_ind] = attn_val
					filter_Intr_Pos[x_ind] = pos_diff#[y_ind] = pos_diff
						
	return y_ind,filter_Intr_Attn,filter_Intr_Pos

def process_attnattr(experiment_blob, intr_dir, params, argSpace, Filter_Intr_Keys=None, device=None, tp_pos_dict={},LabelPreds=None, seq_limit = -1, attn_cutoff = 0.25, numWorkers=1, for_background=False):
	motif_dir = experiment_blob['motif_dir_neg'] if for_background else experiment_blob['motif_dir_pos']
	criterion = experiment_blob['criterion']

	tomtom_data = np.loadtxt(motif_dir+'/tomtom/tomtom.tsv',dtype=str,delimiter='\t')
	if for_background:
		test_loader = experiment_blob['test_loader_bg'] if argSpace.intBackground=='shuffle' else experiment_blob['test_loader']
	else:
		test_loader =  experiment_blob['test_loader']
	saved_model_dir = experiment_blob['saved_model_dir']
	optimizer = experiment_blob['optimizer']
	
	if not os.path.exists(intr_dir):
		os.makedirs(intr_dir)

	num_labels = argSpace.numLabels
	pos_score_cutoff = argSpace.scoreCutoff
	sequence_len = experiment_blob['seq_len']
 
	net = AttentionNet(num_labels, params, device=device,seq_len=sequence_len, genPAttn=True, getAttngrad=True).to(device)
	# ## For tiana only
	# pssm = 'motif_pssm.npy'
	# # load pssm
	# with open(pssm, 'rb') as f:
	# 	motif_array = np.load(f)
		
	# #number of total tf (half of pssm)
	# ntf = motif_array.shape[-1]//2
		
	# # motif_size
	# motif_size = motif_array.shape[0]
		
	# # padding size
	# if ntf % 4 == 0:
	# 	npad = 0
	# elif ntf % 4 != 0:
	# 	npad = 4 - ntf % 4
	# # Example usage
	# net = TianaModel(num_tf=ntf, pad_size=npad, max_len=motif_size, pssm_path='motif_pssm.npy', seq_size=sequence_len).to(device)
	try:    
		checkpoint = torch.load(saved_model_dir+'/model')
		net.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch = checkpoint['epoch']
		loss = checkpoint['loss']
	except:
	    raise Exception("No pre-trained model found! Please run with --mode set to train.")
	
	model = net.to(device)
	model.eval()
	torch.backends.cudnn.enabled=False


	num_filters = params['CNN_filters'][0]
	CNNfirstpool = params['CNN_poolsize'] 
	CNNfiltersize = params['CNN_filtersize'][0]
	batchSize = params['batch_size']
	output_dir = experiment_blob['output_dir']

	col_index = 0
 
	if num_labels == 2:
		res_test = evaluateRegularBatchAttnAttr(net, device, test_loader, criterion, output_dir ,getAttnAttr = True)
	else:
		wrapper = ModelWrapper(net)
		res_test = evaluateRegularBatchMCAttnAttr(net, wrapper, device, test_loader, criterion, output_dir ,getAttnAttr = True)

	Prob_Attention_All = res_test[3]
	if not for_background and LabelPreds == None:
		LabelPreds = res_test[4]
	Seqs = res_test[-1]
	numPosExamples,numNegExamples = get_popsize_for_interactions(argSpace, LabelPreds, batchSize)


	Filter_Intr_Attn = np.ones((len(Filter_Intr_Keys),numPosExamples))*-1
	Filter_Intr_Pos = np.ones((len(Filter_Intr_Keys),numPosExamples)).astype(int)*-1

	Filter_Intr_Attn_neg = np.ones((len(Filter_Intr_Keys),numNegExamples))*-1
	Filter_Intr_Pos_neg = np.ones((len(Filter_Intr_Keys),numNegExamples)).astype(int)*-1		
	numExamples = numPosExamples
	count = 0
	seq_info_dict_list = []	
	for k in range(0,len(Prob_Attention_All)): #going through all batches
		start_time = time.time()
		PAttn = Prob_Attention_All[k]
		feat_size = PAttn.shape[-1]
		per_batch_labelPreds = LabelPreds[k]
		headers = np.asarray(Seqs[k][:,0])  
		if for_background:	
			numExamples = numNegExamples

		if col_index >= numExamples:
				break

		
		if num_labels == 2:
			if for_background:
				tp_indices = [i for i in range(0,per_batch_labelPreds.shape[0]) if (per_batch_labelPreds[i][0]==0 and per_batch_labelPreds[i][1]<(1-pos_score_cutoff))]
			else:
				tp_indices = [i for i in range(0,per_batch_labelPreds.shape[0]) if (per_batch_labelPreds[i][0]==1 and per_batch_labelPreds[i][1]>pos_score_cutoff)]		
		else:
			if argSpace.useAll == True:
				tp_indices = [i for i in range(0,per_batch_labelPreds['labels'].shape[0])]
			else:
				tp_indices=[] 
				TPs = {}
				if not for_background:                                                                                                                                                                                                                                                                                                                                              
					batch_labels = per_batch_labelPreds['labels']                                                                                                                                                                         
					batch_preds = per_batch_labelPreds['preds']   
					for e in range(0,batch_labels.shape[0]):                                                                                                                                                                        
						ex_labels = batch_labels[e].astype(int)                                                                                                                                                                     
						ex_preds = batch_preds[e]                                                                                                                                                                                   
						ex_preds = np.asarray([i>=0.5 for i in ex_preds]).astype(int)    
						prec = metrics.precision_score(ex_labels, ex_preds)         
						if prec >= argSpace.precisionLimit:   
							TP = [i for i in range(0,ex_labels.shape[0]) if (ex_labels[i]==1 and ex_preds[i]==1)] #these are going to be used in calculating attributes: average accross only those columns which are true positives                                                                                                                               
							tp_indices.append(e)
							tp_pos_dict[headers[e]] = TP
							TPs[e] = TP
				else:
					for h_i in range(0, len(headers)):
						header = headers[h_i]
						if header in tp_pos_dict:
							tp_indices.append(h_i)
							TPs[h_i] = tp_pos_dict[header]
		Seqs_tp = Seqs[k][tp_indices]
		print('generating sequence position information...')
		seq_info_dict = get_filters_in_seq_dict(Seqs_tp,motif_dir,num_filters,CNNfirstpool,numWorkers=numWorkers)
		seq_info_dict_list.append([seq_info_dict, tp_indices])
		print('Done!')

		fdata = []
		for ex in tp_indices:
			header = np.asarray(Seqs[k])[ex][0]
			fdata.append([count,PAttn,header,seq_info_dict[header],k,ex,params,tomtom_data,attn_cutoff,sequence_len,CNNfirstpool,num_filters,motif_dir,feat_size])
			count += 1
			col_index += 1
			if count == seq_limit:
				break
			if col_index >= numExamples:
				break

		with Pool(processes = numWorkers) as pool:
			result = pool.map(score_individual_head, fdata, chunksize=1)
		for element in result:
			bid = element[0]
			if for_background == False:
				Filter_Intr_Pos[:,bid] = element[2]
				Filter_Intr_Attn[:,bid] = element[1]
			else:
				Filter_Intr_Pos_neg[:,bid] = element[2]
				Filter_Intr_Attn_neg[:,bid] = element[1]
		
		end_time = time.time()
		if argSpace.verbose:	
			print("Done for Batch: ",k, "Sequences Done: ",count, "Time Taken: %d seconds"%round(end_time-start_time))
	pop_size = count * params['num_multiheads'] #* int(np.ceil(attn_cutoff)) #total sequences tested x # multi heads x number of top attn scores allowed
	return tp_pos_dict, Filter_Intr_Pos, Filter_Intr_Attn, Filter_Intr_Pos_neg, Filter_Intr_Attn_neg, LabelPreds

def analyze_attr_interactions(argSpace,Filter_Intr_Attn,Filter_Intr_Pos,Filter_Intr_Attn_neg,Filter_Intr_Pos_neg, Interact_dir, tomtom_data, plot_dist=True): 
	if plot_dist:
		resMain = Filter_Intr_Attn[Filter_Intr_Attn!=-1]                                                                                                                                               
		resBg = Filter_Intr_Attn_neg[Filter_Intr_Attn_neg!=-1]
		with open(Interact_dir+'/attnattr_score_main_bg.pckl','wb') as f:
			pickle.dump([resMain,resBg],f)
		resMainHist = np.histogram(resMain,bins=20)
		resBgHist = np.histogram(resBg,bins=20)
		plt.plot(resMainHist[1][1:],resMainHist[0]/sum(resMainHist[0]),linestyle='--',marker='o',color='g',label='main')
		plt.plot(resBgHist[1][1:],resBgHist[0]/sum(resBgHist[0]),linestyle='--',marker='x',color='r',label='background')
		plt.legend(loc='best',fontsize=10)
		plt.savefig(Interact_dir+'/normalized_AttnAttr_scores_distributions.pdf')
		plt.clf()

		plt.hist(resMain,bins=20,color='g',label='main')
		plt.hist(resBg,bins=20,color='r',alpha=0.5,label='background')
		plt.legend(loc='best',fontsize=10)
		plt.savefig(Interact_dir+'/AttnAttr_scores_distributions.pdf')
		plt.clf()
		
		Bg_MaxMean = []
		Main_MaxMean = []
		for entry in Filter_Intr_Attn:
			try:
				Main_MaxMean.append([np.max(entry[entry!=-1]),np.mean(entry[entry!=-1])])
			except:
				continue	
		for entry in Filter_Intr_Attn_neg:
			try:
				Bg_MaxMean.append([np.max(entry[entry!=-1]),np.mean(entry[entry!=-1])])
			except:
				continue
			
		Bg_MaxMean = np.asarray(Bg_MaxMean)
		Main_MaxMean = np.asarray(Main_MaxMean)
		
		plt.hist(Main_MaxMean[:,0],bins=20,color='g',label='main')
		plt.hist(Bg_MaxMean[:,0],bins=20,color='r',alpha=0.5,label='background')
		plt.legend(loc='best',fontsize=10)
		plt.savefig(Interact_dir+'/Attn_scores_distributions_MaxPerInteraction.pdf')
		plt.clf()
		
		plt.hist(Main_MaxMean[:,1],bins=20,color='g',label='main')
		plt.hist(Bg_MaxMean[:,1],bins=20,color='r',alpha=0.5,label='background')
		plt.legend(loc='best',fontsize=10)
		plt.savefig(Interact_dir+'/Attn_scores_distributions_MeanPerInteraction.pdf')
		plt.clf()
	
	attnLimits = [argSpace.attnCutoff] if argSpace.attnCutoff==0.12 else [0.12, argSpace.attnCutoff]
	attnCutoff = np.mean(Filter_Intr_Attn[Filter_Intr_Attn!=-1])
 
	if argSpace.attnCutoff == 0.04:
 
		attnLimits = [attnCutoff]
	else: 
		attnLimits = [argSpace.attnCutoff]
	for attnLimit in attnLimits:
		pval_info = []#{}
		for i in range(0,Filter_Intr_Attn.shape[0]):                                                                                                                                                   
			pos_attn = Filter_Intr_Attn[i,:]                                                                                                                                                              
			pos_attn = pos_attn[pos_attn!=-1]#pos_attn[pos_attn>0.04] #pos_attn[pos_attn!=-1]                                                                                                                                                                   
			neg_attn = Filter_Intr_Attn_neg[i,:]                                                                                                                                                          
			neg_attn = neg_attn[neg_attn!=-1]#neg_attn[neg_attn>0.04] #neg_attn[neg_attn!=-1] 
			num_pos = len(pos_attn)
			num_neg = len(neg_attn)
			if len(pos_attn) <= 1:# or len(neg_attn) <= 1:
				continue
			if len(neg_attn) <= 1: #if just 1 or 0 values in neg attn, get a vector with all values set to 0 (same length as pos_attn)
				neg_attn = np.asarray([0 for i in range(0,num_pos)])
			if np.max(pos_attn) < attnLimit: # 
				continue
			pos_posn = Filter_Intr_Pos[i,:]  
			pos_posn_mean = pos_posn[np.argmax(Filter_Intr_Attn[i,:])] #just pick the max
			neg_posn = Filter_Intr_Pos_neg[i,:]  
			neg_posn_mean = neg_posn[np.argmax(Filter_Intr_Attn_neg[i,:])] #just pick the max
			stats,pval = mannwhitneyu(pos_attn,neg_attn,alternative='greater')#ttest_ind(pos_d,neg_d)#mannwhitneyu(pos_d,neg_d,alternative='greater')   
			pval_info.append([i, pos_posn_mean, neg_posn_mean,num_pos,num_neg, stats,pval])#pval_dict[i] = [i,stats,pval]                                                                                                                                                              
			#if i%100==0:                                                                                                                                                                               
			#	print('Done: ',i) 
		pval_info = np.asarray(pval_info)
		res_final = pval_info#[pval_info[:,-1]<0.01] #can be 0.05 or any other threshold #For now, lets take care of this in post processing (jupyter notebook)
		res_final_int = []   
		#print("Res final " , res_final.shape[0])                                                                                                                                                                                                                                                                                                                                               
		for i in range(0,res_final.shape[0]):                                                                                                                                                          
			value = int(res_final[i][0])                                                                                                                                                               
			pval = res_final[i][-1]
			pp_mean = res_final[i][1]
			np_mean = res_final[i][2]  
			num_pos = res_final[i][3]
			num_neg = res_final[i][4]         
			stats = res_final[i][-2]                                                                                                                                                         
			for key in Filter_Intr_Keys:                                                                                                                                                               
				if Filter_Intr_Keys[key] == value:                                                                                                                                                     
					res_final_int.append([key,value,pp_mean,np_mean,num_pos,num_neg,stats,pval])  
		
		res_final_int = np.asarray(res_final_int) 
		qvals = multipletests(res_final_int[:,-1].astype(float), method='fdr_bh')[1] #res_final_int[:,1].astype(float)
		res_final_int = np.column_stack((res_final_int,qvals))
		
		final_interactions = [['filter_interaction','example_no','motif1','motif1_qval','motif2','motif2_qval','mean_distance','mean_distance_bg','num_obs','num_obs_bg','pval','adjusted_pval']]
		for entry in res_final_int:                                                                                                                                                                    
			f1,f2 = entry[0].split('<-->')                                                                                                                                                             	                                                                                                                                                                      
			m1_ind = np.argwhere(tomtom_data[:,0]==f1)                                                                                                                                                 
			m2_ind = np.argwhere(tomtom_data[:,0]==f2)                                                                                                                                                 
			#print(m1_ind,m2_ind)
			if len(m1_ind) == 0 or len(m2_ind) == 0:
				continue
			m1 = tomtom_data[m1_ind[0][0]][1]
			m2 = tomtom_data[m2_ind[0][0]][1]
			m1_pval = tomtom_data[m1_ind[0][0]][5]
			m2_pval = tomtom_data[m2_ind[0][0]][5]
			final_interactions.append([entry[0],entry[1],m1,m1_pval,m2,m2_pval,entry[2],entry[3],entry[4],entry[5],entry[-2],entry[-1]])
			#print(entry[-1],m1,m2,entry[0])
		np.savetxt(Interact_dir+'/interactions_attnLimit-'+str(attnLimit)+'.txt',final_interactions,fmt='%s',delimiter='\t')
		with open(Interact_dir+'/processed_results_attnLimit-'+str(attnLimit)+'.pckl','wb') as f:
			pickle.dump([pval_info,res_final_int],f)
		print("Done for Attention Cutoff Value: ",str(attnLimit))


def infer_intr_ATTNATTR(experiment_blob, params, argSpace, device=None):
    
 
	global Filter_Intr_Keys
	output_dir = experiment_blob['output_dir']
	intr_dir = output_dir + '/Interactions_ATTNATTR'
 
	batchSize = params['batch_size']
	num_labels = argSpace.numLabels

	Filter_Intr_Keys = get_intr_filter_keys(params['CNN_filters'][0]) 
    
	tp_pos_dict, Filter_Intr_Pos, Filter_Intr_Attn, _,_ , LabelsPred= process_attnattr(experiment_blob, intr_dir, params, argSpace, Filter_Intr_Keys=Filter_Intr_Keys, device=device)
	_,_,_, Filter_Intr_Pos_neg, Filter_Intr_Attn_neg,_ = process_attnattr(experiment_blob, intr_dir, params, argSpace, Filter_Intr_Keys=Filter_Intr_Keys, device=device,LabelPreds=LabelsPred, tp_pos_dict=tp_pos_dict, for_background=True)

	motif_dir_pos = experiment_blob['motif_dir_pos']
	tomtom_data = np.loadtxt(motif_dir_pos+'/tomtom/tomtom.tsv',dtype=str,delimiter='\t')

	analyze_attr_interactions(argSpace,Filter_Intr_Attn, Filter_Intr_Pos,Filter_Intr_Attn_neg,Filter_Intr_Pos_neg, intr_dir, tomtom_data)
