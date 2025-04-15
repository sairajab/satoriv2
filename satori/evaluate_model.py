import os
import pickle
import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Any
import torch.nn.functional as F
import time
import torch.nn as nn

def evaluateRegularMC(net, device, iterator, criterion, out_dirc,  ent_loss=False, entropy_reg_weight=0.005, getPAttn=False, storePAttn=False, getCNN=False, storeCNNout=False, getSeqs=False, motifweights = False,getAttnAttr = False, classlabel=1):
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
    if getCNN:
        if motifweights:
                CNNlayer = net.layer1[0:1]
        else:
            CNNlayer = net.layer1[0:3]   # first conv layer without the maxpooling part
        CNNlayer.eval()
    roc = np.asarray([[], []]).T
    PAttn_all = {}
    per_batch_labelPreds = {}
    per_batch_CNNoutput = {}
    per_batch_testSeqs = {}
 
    with torch.no_grad():
        for batch_idx, (headers, seqs, data, target) in enumerate(iterator):
            data, target = data.to(device, dtype=torch.float), target.to(
                    device, dtype=torch.float)
            outputs,PAttn = net(data)
            if ent_loss:
                epsilon = 1e-10
                PAttn = PAttn + epsilon
                feat_size = PAttn.shape[-1]
                attention_entropy = -torch.sum(PAttn * torch.log(PAttn), dim=-1)
                reshaped_tensor = attention_entropy.view(PAttn.shape[0], -1, feat_size).mean(dim=2)

                per_seq_entropy_loss = reshaped_tensor.sum(dim=1)
                entropy_loss = per_seq_entropy_loss.mean() 
                loss = criterion(outputs, target) + \
                        (entropy_reg_weight * entropy_loss)
            else:
                    loss = criterion(outputs, target)
            labels = target.cpu().numpy()
            sigmoid = torch.nn.Sigmoid()
            pred = sigmoid(outputs).cpu().detach().numpy()
            all_labels += labels.tolist()
            all_preds += pred.tolist()
            label_pred = {'labels': labels, 'preds': pred}
            per_batch_labelPreds[batch_idx] = label_pred
            if getPAttn == True:
                if storePAttn == True:
                    output_dir = out_dirc
                    if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                    with open(output_dir+'/PAttn_batch-'+str(batch_idx)+'.pckl', 'wb') as f:
                            pickle.dump(PAttn.cpu().detach().numpy(), f)
                    PAttn_all[batch_idx] = output_dir+'/PAttn_batch-' + \
                            str(batch_idx)+'.pckl'  # paths to the pickle PAttention
                else:
                    PAttn_all[batch_idx] = PAttn.cpu().detach().numpy()
                    
            if getCNN == True:
                outputCNN = CNNlayer(data)
                if storeCNNout == True:
                    output_dir = out_dirc
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    with open(output_dir+'/CNNout_batch-'+str(batch_idx)+'.pckl', 'wb') as f:
                        pickle.dump(outputCNN.cpu().detach().numpy(), f)
                    per_batch_CNNoutput[batch_idx] = output_dir + \
                            '/CNNout_batch-'+str(batch_idx)+'.pckl'
                else:
                    per_batch_CNNoutput[batch_idx] = outputCNN.cpu(
                        ).detach().numpy()
                    
            if getSeqs == True:
                    per_batch_testSeqs[batch_idx] = np.column_stack(
                        (headers, seqs))
            running_loss += loss.item()
                
    for j in range(0, len(all_labels[0])):
        cls_labels = np.asarray(all_labels)[:, j]
        pred_probs = np.asarray(all_preds)[:, j]
        auc_score = metrics.roc_auc_score(cls_labels.astype(int), pred_probs)
        valid_auc.append(auc_score)
    return running_loss/len(iterator), valid_auc, roc, PAttn_all, per_batch_labelPreds, per_batch_CNNoutput, per_batch_testSeqs


def evaluateRegular(net, device, iterator, criterion, out_dirc, ent_loss=False, entropy_reg_weight=0.005, getPAttn=False, storePAttn=False, getCNN=False, storeCNNout=False, getSeqs=False, motifweights = False, getAttnAttr = None):
    running_loss = 0.0
    net.eval()
    if getCNN:
        if motifweights:
            CNNlayer = net.layer1[0:1]
        else:
            CNNlayer = net.layer1[0:3]   # first conv layer without the maxpooling part
        CNNlayer.eval()
    roc = np.asarray([[], []]).T
    PAttn_all = {}
    per_batch_labelPreds = {}
    per_batch_CNNoutput = {}
    per_batch_testSeqs = {}
    i = 0
    with torch.no_grad():

        for batch_idx, (headers, seqs, data, target) in enumerate(iterator):

            data, target = data.to(device, dtype=torch.float), target.to(
                    device, dtype=torch.long)
            outputs,PAttn = net(data)  # dist
            if ent_loss:
                epsilon = 1e-10
                PAttn = PAttn + epsilon
                feat_size = PAttn.shape[-1]
                attention_entropy = -torch.sum(PAttn * torch.log(PAttn), dim=-1)
                reshaped_tensor = attention_entropy.view(PAttn.shape[0], -1, feat_size).mean(dim=2)

                per_seq_entropy_loss = reshaped_tensor.sum(dim=1)
                entropy_loss = per_seq_entropy_loss.mean() 

                loss = criterion(outputs, target) + \
                        (entropy_reg_weight * entropy_loss)
            else:
                loss = criterion(outputs, target)

            softmax = torch.nn.Softmax(dim=1)
            labels = target.cpu().numpy()
            pred = softmax(outputs)

            if getPAttn == True:
                if storePAttn == True:
                    output_dir = out_dirc
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    with open(output_dir+'/PAttn_batch-'+str(batch_idx)+'.pckl', 'wb') as f:
                        pickle.dump(PAttn.cpu().detach().numpy(), f)

                    PAttn_all[batch_idx] = output_dir+'/PAttn_batch-' + \
                            str(batch_idx)+'.pckl'  # paths to the pickle PAttention
                else:
                    PAttn_all[batch_idx] = PAttn.cpu().detach().numpy()
                    ''' for the visualization of the attention weights'''
                    if i == 0 :
                        output_dir = out_dirc
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        with open(output_dir+'/PAttn-'+str(batch_idx)+'.pckl', 'wb') as f:
                                pickle.dump(PAttn_all[batch_idx], f)
                                
                        with open(output_dir+'/predictions-'+str(batch_idx)+'.pckl', 'wb') as f:
                                pickle.dump(pred.cpu().detach().numpy(), f)
                        with open(output_dir+'/labels-'+str(batch_idx)+'.pckl', 'wb') as f:
                                pickle.dump(labels, f)
                        with open(output_dir+'/headers-'+str(batch_idx)+'.pckl', 'wb') as f:
                                pickle.dump(headers, f)
                    i = i + 1
                    ''' for the visualization of the attention weights'''
                    
            pred = pred.cpu().detach().numpy()
            label_pred = np.column_stack((labels, pred[:,1]))#[:, 1]
            per_batch_labelPreds[batch_idx] = label_pred
            
            roc = np.row_stack((roc, label_pred))
            running_loss += loss.item()
            if getCNN == True:
                outputCNN = CNNlayer(data)
                #print(outputCNN[0,0,:], outputCNN.shape)
                #print(pred[0,:], labels[0])
                if storeCNNout == True:
                    output_dir = out_dirc
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    with open(output_dir+'/CNNout_batch-'+str(batch_idx)+'.pckl', 'wb') as f:
                        pickle.dump(outputCNN.cpu().detach().numpy(), f)
                    per_batch_CNNoutput[batch_idx] = output_dir + \
                            '/CNNout_batch-'+str(batch_idx)+'.pckl'
                else:
                    per_batch_CNNoutput[batch_idx] = outputCNN.cpu(
                        ).detach().numpy()
            if getSeqs == True:
                per_batch_testSeqs[batch_idx] = np.column_stack(
                        (headers, seqs))
    
    labels = roc[:, 0]
    preds = roc[:, 1]
    preds = np.nan_to_num(preds, nan=0.5)
    valid_auc = metrics.roc_auc_score(labels, preds)
    return running_loss/len(iterator), valid_auc, roc, PAttn_all, per_batch_labelPreds, per_batch_CNNoutput, per_batch_testSeqs