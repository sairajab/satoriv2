import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn import metrics
from satori.utils import compute_roc_auc

def trainRegularMC(model, device, iterator, optimizer, criterion, ent_loss=False, entropy_reg_weight=0.005):
    model.train()
    running_loss = 0.0
    train_auc = []
    all_labels = []
    all_preds = []
    count = 0
    for batch_idx, (headers, seqs, data, target) in enumerate(iterator):
        data, target = data.to(device, dtype=torch.float), target.to(
            device, dtype=torch.float)
        optimizer.zero_grad()
        outputs, PAttn = model(data)
        if ent_loss:
            epsilon = 1e-10
            PAttn = PAttn + epsilon
            
            attention_entropy = -torch.sum(PAttn * torch.log(PAttn), dim=-1)
            reshaped_tensor = attention_entropy.view(PAttn.shape[0], -1, PAttn.shape[1]).mean(dim=2)
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
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    for j in range(0, len(all_labels[0])):
        cls_labels = np.asarray(all_labels)[:, j]
        pred_probs = np.asarray(all_preds)[:, j]
        auc_score = metrics.roc_auc_score(cls_labels.astype(int), pred_probs)
        train_auc.append(auc_score)
    return running_loss/len(iterator), train_auc


def trainRegular(model, device, iterator, optimizer, criterion, ent_loss=False, entropy_reg_weight=0.005, reg=False):
    model.train()
    running_loss = 0.0
    pred_list = []
    labels_list = []
    train_auc = []
    for batch_idx, (headers, seqs, data, target) in enumerate(iterator):
        data, target = data.to(device, dtype=torch.float), target.to(
            device, dtype=torch.long)
        
        optimizer.zero_grad()
        outputs,PAttn = model(data)
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
            if reg:
                loss += model.regularization_loss()

        labels = target.cpu().numpy()
        softmax = torch.nn.Softmax(dim=1)
        pred = softmax(outputs).cpu().detach().numpy()

        labels_list.extend(labels)
        pred_list.extend(pred[:, 1])

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_auc = compute_roc_auc(np.asarray(labels_list), np.asarray(pred_list))
    
    return running_loss/len(iterator), train_auc


