import numpy as np
import os
import pickle
# import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, MultiStepLR, StepLR
from fastprogress import progress_bar
from sklearn import metrics
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
# local imports
from datasets import DatasetLoadAll, DatasetLazyLoad
from extract_motifs import get_motif
from satori.models import AttentionNet
from utils import get_shuffled_background, get_shuffled_background_presaved
from cnn_weights import *
# from cnn_motif_extraction import *
import torch
from torch.utils.tensorboard import SummaryWriter

###########################################################################################################################
# --------------------------------------------Train and Evaluate Functions-------------------------------------------------#
###########################################################################################################################


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
            entropy_loss = attention_entropy.mean()
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


def trainRegular(model, device, iterator, optimizer, criterion, ent_loss=False, entropy_reg_weight=0.005):
    model.train()
    running_loss = 0.0

    train_auc = []
    for batch_idx, (headers, seqs, data, target) in enumerate(iterator):
        data, target = data.to(device, dtype=torch.float), target.to(
            device, dtype=torch.long)
        optimizer.zero_grad()
        outputs,PAttn = model(data)
        # print(attn.type())
        # attn_weights = torch.split(attn, 32, dim = -1)
        # calculate entropy of attn
        if ent_loss:
            epsilon = 1e-10
            PAttn = PAttn + epsilon
            attention_entropy = -torch.sum(PAttn * torch.log(PAttn), dim=-1)
            entropy_loss = attention_entropy.mean()
            
            # attn_shape = PAttn.shape
            # n_heads = attn_shape[-1] // attn_shape[1] 
            # #print(n_heads)
            # attn_mat_split = PAttn.chunk(n_heads, dim=-1)
            # reshaped_PAattn = torch.stack(attn_mat_split, dim=1)
            # #print(reshaped_PAattn[0 , 0, :, 0].sum())

            #reshaped_PAattn = PAttn.view(attn_shape[0], n_heads, attn_shape[1], attn_shape[1])
            #temp = -torch.sum(reshaped_PAattn * torch.log(reshaped_PAattn), dim=-1)
            #print(temp.shape)
            #entropy_loss_overheads = temp.mean()
            
            #print(entropy_loss, PAttn[0,0,0,:].sum())

            
            # temp =torch.cat([temp] + entropy_loss2 , dim = 0)
            # print(temp)
            loss = criterion(outputs, target) + \
                (entropy_reg_weight * entropy_loss)
        else:
            loss = criterion(outputs, target)
        labels = target.cpu().numpy()
        softmax = torch.nn.Softmax(dim=1)
        pred = softmax(outputs).cpu().detach().numpy()
        try:
            train_auc.append(metrics.roc_auc_score(labels, pred[:, 1]))
        except:
            train_auc.append(0.0)
        loss.backward()

        optimizer.step()
        running_loss += loss.item()
    return running_loss/len(iterator), train_auc


def evaluateRegularMC(net, device, iterator, criterion, out_dirc,  ent_loss=False, entropy_reg_weight=0.005, getPAttn=False, storePAttn=False, getCNN=False, storeCNNout=False, getSeqs=False):
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
    with torch.no_grad():
        for batch_idx, (headers, seqs, data, target) in enumerate(iterator):
            data, target = data.to(device, dtype=torch.float), target.to(
                device, dtype=torch.float)
            outputs, PAttn = net(data)
            if ent_loss:
                epsilon = 1e-10
                PAttn = PAttn + epsilon
                attention_entropy = - \
                    torch.sum(PAttn * torch.log(PAttn), dim=-1)
                entropy_loss = attention_entropy.mean()
                loss = criterion(outputs, target) + \
                    (entropy_reg_weight * entropy_loss)
            else:
                loss = criterion(outputs, target)
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


def evaluateRegular(net, device, iterator, criterion, out_dirc, ent_loss=False, entropy_reg_weight=0.005, getPAttn=False, storePAttn=False, getCNN=False, storeCNNout=False, getSeqs=False):
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

    with torch.no_grad():
        for batch_idx, (headers, seqs, data, target) in enumerate(iterator):
            data, target = data.to(device, dtype=torch.float), target.to(
                device, dtype=torch.long)
            outputs, PAttn = net(data)  # dist
            #print("HERE", ent_loss)
            if ent_loss:
                epsilon = 1e-10
                PAttn = PAttn + epsilon
                attention_entropy = - \
                    torch.sum(PAttn * torch.log(PAttn), dim=-1)
                entropy_loss = attention_entropy.mean()
                # attn_shape = PAttn.shape
                # n_heads = attn_shape[-1] // attn_shape[1] 
  
                # attn_mat_split = PAttn.chunk(n_heads, dim=-1)
                # reshaped_PAattn = torch.stack(attn_mat_split, dim=0)

                # #reshaped_PAattn = PAttn.view(attn_shape[0], n_heads, attn_shape[1], attn_shape[1])
                # temp = -torch.sum(reshaped_PAattn * torch.log(reshaped_PAattn), dim=-1)
                # entropy_loss_overheads = temp.mean()
                # #print("Entropy loss",entropy_loss, PAttn.shape, attention_entropy.shape )
                # print(entropy_loss_overheads)
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
                    # with open(output_dir+'/dist_batch-'+str(batch_idx)+'.pckl','wb') as f:
                    #     pickle.dump(dist.cpu().detach().numpy(),f)
                    PAttn_all[batch_idx] = output_dir+'/PAttn_batch-' + \
                        str(batch_idx)+'.pckl'  # paths to the pickle PAttention
                else:
                    PAttn_all[batch_idx] = PAttn.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            label_pred = np.column_stack((labels, pred[:, 1]))
            per_batch_labelPreds[batch_idx] = label_pred
            roc = np.row_stack((roc, label_pred))
            try:
                valid_auc.append(metrics.roc_auc_score(labels, pred[:, 1]))
            except:
                valid_auc.append(0.0)
            running_loss += loss.item()
            outputCNN = CNNlayer(data).cpu().detach().numpy()
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
    labels = roc[:, 0]
    preds = roc[:, 1]
    valid_auc = metrics.roc_auc_score(labels, preds)
    return running_loss/len(iterator), valid_auc, roc, PAttn_all, per_batch_labelPreds, per_batch_CNNoutput, per_batch_testSeqs


###########################################################################################################################
# ---------------------------------------------------------End-------------------------------------------------------------#
###########################################################################################################################
def get_indices(dataset_size, test_split, output_dir, data, shuffle_data=True, seed_val=100, mode='train'):
    indices = list(range(dataset_size))
    split_val = int(np.floor(test_split*dataset_size))
    if shuffle_data:
        np.random.seed(seed_val)
        np.random.shuffle(indices)
    # --save indices for later use, when testing for example---#
    if mode == 'train':

        if data == "human_promoters":

            valid_indices = np.loadtxt(
                '/s/chromatin/p/nobackup/Saira/original/satori/data/Human_Promoters/valid_indices.txt', dtype=int)
            test_indices = np.loadtxt(
                '/s/chromatin/p/nobackup/Saira/original/satori/data/Human_Promoters/test_indices.txt', dtype=int)
            train_indices = np.loadtxt(
                '/s/chromatin/p/nobackup/Saira/original/satori/data/Human_Promoters/train_indices.txt', dtype=int)

        elif data == "arabidopsis":

            valid_indices = np.loadtxt(
                '/s/chromatin/p/nobackup/Saira/original/satori/data/Arabidopsis_ChromAccessibility/valid_indices.txt', dtype=int)
            test_indices = np.loadtxt(
                '/s/chromatin/p/nobackup/Saira/original/satori/data/Arabidopsis_ChromAccessibility/test_indices.txt', dtype=int)
            train_indices = np.loadtxt(
                '/s/chromatin/p/nobackup/Saira/original/satori/data/Arabidopsis_ChromAccessibility/train_indices.txt', dtype=int)

        else:
            train_indices, test_indices, valid_indices = np.array(
                indices[2*split_val:]), np.array(indices[:split_val]), np.array(indices[split_val:2*split_val])

        np.savetxt(output_dir+'/valid_indices.txt', valid_indices, fmt='%s')
        np.savetxt(output_dir+'/test_indices.txt', test_indices, fmt='%s')
        np.savetxt(output_dir+'/train_indices.txt', train_indices, fmt='%s')

    else:
        try:
            valid_indices = np.loadtxt(
                output_dir+'/valid_indices.txt', dtype=int)
            test_indices = np.loadtxt(
                output_dir+'/test_indices.txt', dtype=int)
            train_indices = np.loadtxt(
                output_dir+'/train_indices.txt', dtype=int)
        except:
            raise Exception(
                "Error! looks like you haven't trained the model yet. Rerun with --mode train.")
    return train_indices, test_indices, valid_indices


def load_datasets(arg_space, batchSize, dataset):
    """
    Loads and processes the data.
    """
    input_prefix = arg_space.inputprefix
    output_dir = arg_space.directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # save arguments to keep record
    with open(output_dir+'/arguments.txt', 'w') as f:
        f.writelines(str(arg_space))
    test_split = arg_space.splitperc/100

    if arg_space.verbose:
        print("test/validation split val: %.2f" % test_split)

    if arg_space.deskLoad:
        final_dataset = DatasetLazyLoad(
            input_prefix, num_labels=arg_space.numLabels)
    else:
        final_dataset = DatasetLoadAll(
            input_prefix, num_labels=arg_space.numLabels)
    seq_len = final_dataset.get_seq_len()
    train_indices, test_indices, valid_indices = get_indices(
        len(final_dataset), test_split, output_dir, dataset, mode=arg_space.mode)
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)
    train_loader = DataLoader(final_dataset, batch_size=batchSize,
                              sampler=train_sampler, num_workers=arg_space.numWorkers)
    test_loader = DataLoader(final_dataset, batch_size=batchSize,
                             sampler=test_sampler, num_workers=arg_space.numWorkers)
    valid_loader = DataLoader(final_dataset, batch_size=batchSize,
                              sampler=valid_sampler, num_workers=arg_space.numWorkers)
    return train_loader, train_indices, test_loader, test_indices, valid_loader, valid_indices, output_dir, seq_len


def run_experiment(device, arg_space, params):
    """
    Run the main experiment, that is, load the data and train-test the model and generate/store results.
    Args:
        device: (torch.device) Specifies the device (either gpu or cpu).
        arg_space: ArgParser object containing all the user-specified arguments.
        params: (dict) Dictionary of hyperparameters. 
    """
    num_labels = arg_space.numLabels
    load_cnn_ws = arg_space.load_motif_weights

    genPAttn = params['get_pattn']
    getCNNout = params['get_CNNout']
    getSequences = params['get_seqs']
    batch_size = params['batch_size']
    max_epochs = params['num_epochs']
    ent_loss = params['entropy_loss']
    ent_reg = float(params['entropy_reg_value'])
    writer = SummaryWriter()

    # Using generic, not sure if we need it as an argument or part of the params dict
    prefix = 'modelRes_Val'
    train_loader, train_indices, test_loader, test_indices, valid_loader, valid_indices, output_dir, seq_len = load_datasets(
        arg_space, batch_size, params['exp_name'])

    if load_cnn_ws:
        params['CNN_filtersize'] = [21]

    # print("dataaa", iter(train_loader).next())
    net = AttentionNet(arg_space, params, device=device,
                       seq_len=seq_len).to(device)
    if load_cnn_ws:
        net.layer1[0].weight.data.uniform_(-1.0, 0.0)
        weights_tensor = load_cnn_weights(
            output_dir, tfs_pairs_path=arg_space.pairs_file).to(device)
        kernels_to_freeze = 100  # Freeze half of the kernels

        print(net.layer1[0].weight.shape, weights_tensor.shape)
        net.layer1[0].weight = nn.parameter.Parameter(
            weights_tensor, requires_grad=True)
        # Loop through the kernels and freeze half of them
        # print(len(net.layer1[0].parameters()))
        # for idx in range(kernels_to_freeze):
        #     net.layer1[0].weight[idx].requires_grad = False

        #     # if idx < kernels_to_freeze:
        #     #     param.requires_grad = False  # Freeze the weight
        #     print(idx, "HERE")
        # else:
        #     print(idx,param.requires_grad)

    # dummy_input = torch.randn(10, 4, 200, device="cuda")
    # torch.onnx.export(net, dummy_input, output_dir + "/model.onnx", verbose=True)
    if num_labels == 2:

        criterion = nn.CrossEntropyLoss(reduction='mean')
        optimizer = optim.SGD(net.parameters(), lr=0.01)
        #optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)

    else:
        criterion = nn.BCEWithLogitsLoss()

        optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)
        #optimizer = optim.SGD(net.parameters(), lr=0.001)
        #optimizer = optim.AdamW(net.parameters(), lr=0.001,weight_decay=1e-5)

        #optimizer = optim.RMSprop(net.parameters(), lr=0.001) #optim.Adam(net.parameters(), lr=0.001)
        #optimizer = torch.optim.Adagrad(net.parameters(), lr=0.01,weight_decay=0.00001)

    # best_optimizer = torch.optim.Adagrad(net.parameters(), lr=0.01,weight_decay=0.00001)
    # optimizer = sgd #best_optimizer
    # Define ReduceLROnPlateau scheduler
    # scheduler = StepLR(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # scheduler1 = ExponentialLR(optimizer, gamma=0.5)
    # scheduler2 = MultiStepLR(optimizer, milestones=[5, 10, 20], gamma=0.1)

    if arg_space.finetune_model_path != None:
        try:
            checkpoint = torch.load(arg_space.finetune_model_path+'/model')
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print("Loaded model for finetunig...")
            #optimizer.param_groups[0]['lr'] = 0.0005

            # optimizer = optim.SGD(net.parameters(), lr=0.0025)
            #optimizer = optim.Adam(net.parameters(), lr=0.001)

            scheduler = ReduceLROnPlateau(
                optimizer, 'min', patience=5, factor=0.5)

        except:
            raise Exception(
                f"No pre-trained model found  for finetuning at {saved_model_dir}! Please run with --mode set to train.")

    logs = open(output_dir + "/training_logs.txt", "w")
    saved_model_dir = output_dir+'/Saved_Model'
    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir)
    print(net)

    ## -------Main train/test loop----------##
    if arg_space.mode == 'train':
        best_valid_loss = np.inf
        best_valid_auc = np.inf
        for epoch in progress_bar(range(1, max_epochs + 1)):
            if num_labels == 2:
                res_train = trainRegular(
                    net, device, train_loader, optimizer, criterion, ent_loss, ent_reg)
            else:
                res_train = trainRegularMC(
                    net, device, train_loader, optimizer, criterion, ent_loss, ent_reg)
            res_train_auc = np.asarray(res_train[1]).mean()
            res_train_loss = res_train[0]
            if arg_space.verbose:
                print("Train Results (Loss and AUC): ",
                      res_train_loss, res_train_auc)
            if num_labels == 2:
                res_valid = evaluateRegular(net, device, valid_loader, criterion, output_dir+"/Stored_Values", ent_loss, ent_reg, getPAttn=False,
                                            storePAttn=False, getCNN=False,
                                            storeCNNout=False, getSeqs=False)  # evaluateRegular(net,valid_loader,criterion)
                res_valid_loss = res_valid[0]
                res_valid_auc = res_valid[1]
            else:
                res_valid = evaluateRegularMC(net, device, valid_loader, criterion, output_dir+"/Stored_Values", getPAttn=False,
                                              storePAttn=False, getCNN=False,
                                              storeCNNout=False, getSeqs=False)  # evaluateRegular(net,valid_loader,criterion)
                res_valid_loss = res_valid[0]
                res_valid_auc = np.mean(res_valid[1])
            if res_valid_loss < best_valid_loss:
                best_valid_loss = res_valid_loss
                best_valid_auc = res_valid_auc
                if arg_space.verbose:
                    print("Best Validation Loss: %.3f and AUC: %.2f" %
                          (best_valid_loss, best_valid_auc), "\n")
                torch.save({'epoch': epoch,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': res_valid_loss
                            }, saved_model_dir+'/model')

                counter = 0
            else:

                counter = counter + 1

            scheduler.step(res_valid_loss)
            writer.add_scalar("Loss/train", res_train_loss, epoch)
            writer.add_scalar("Loss/val", res_valid_loss, epoch)

            # scheduler2.step(res_valid_loss)

            print('Epoch-{0} lr: {1} valid_loss: {2}' .format(epoch,
                  optimizer.param_groups[0]['lr'], res_valid_loss))
            logs.writelines('Epoch-{0} lr: {1} valid_loss: {2} \n' .format(
                epoch, optimizer.param_groups[0]['lr'], res_valid_loss))

            if counter >= 40:
                print("Early stopping at ",epoch )
                break
        if num_labels == 2:
            res_valid = evaluateRegular(net, device, valid_loader, criterion, output_dir+"/Stored_Values", ent_loss, ent_reg, getPAttn=False,
                                        storePAttn=False, getCNN=False,
                                        storeCNNout=False, getSeqs=False)
        else:
            res_valid = evaluateRegularMC(net, device, valid_loader, criterion, output_dir+"/Stored_Values", ent_loss, ent_reg, getPAttn=False,
                                          storePAttn=False, getCNN=False,
                                          storeCNNout=False, getSeqs=False)

        if num_labels == 2:
            # Save valid files
            valid_loss = res_valid[0]
            labels = res_valid[2][:, 0]
            preds = res_valid[2][:, 1]
            auc_valid = metrics.roc_auc_score(labels, preds)
            if arg_space.verbose:
                print("Valid Loss: %.3f and AUC: %.2f" %
                      (valid_loss, auc_valid), "\n")
            auprc_valid = metrics.average_precision_score(labels, preds)
            some_res = [['Valid_Loss', 'Valid_AUC', 'Valid_AUPRC']]
            some_res.append([valid_loss, auc_valid, auprc_valid])
            # ---Calculate roc and prc values---#
            fpr, tpr, thresholds = metrics.roc_curve(labels, preds)
            precision, recall, thresholdsPR = metrics.precision_recall_curve(
                labels, preds)
            roc_dict = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
            prc_dict = {'precision': precision,
                        'recall': recall, 'thresholds': thresholdsPR}
            # ---Store results----#
            with open(output_dir+'/'+prefix+'_roc.pckl', 'wb') as f:
                pickle.dump(roc_dict, f)
            with open(output_dir+'/'+prefix+'_prc.pckl', 'wb') as f:
                pickle.dump(prc_dict, f)
            np.savetxt(output_dir+'/'+prefix+'_results.txt',
                       some_res, fmt='%s', delimiter='\t')

    try:
        checkpoint = torch.load(saved_model_dir+'/model')
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    except:
        raise Exception(
            f"No pre-trained model found at {saved_model_dir}! Please run with --mode set to train.")

    if num_labels == 2:
        prefix = 'modelRes'
        arg_space.storeInterCNN = False
        ent_loss = True
        
        res_test = evaluateRegular(net, device, test_loader, criterion, output_dir+"/Stored_Values", ent_loss, ent_reg, getPAttn=genPAttn,
                                   storePAttn=arg_space.storeInterCNN, getCNN=getCNNout,
                                   storeCNNout=arg_space.storeInterCNN, getSeqs=getSequences)
        # uncomment for CNN based motif annotations
        # annotation_file = arg_space.inputprefix + "_info.txt"

        # ids_to_motifs = read_gt_motifs_location_ind(annotation_file, valid_indices)
        # print(ids_to_motifs)

        # _ = evaluate_cnn_motif(net, device, valid_loader, output_dir+"/Motif_Analysis/", ids_to_motifs, valid_indices)

        test_loss = res_test[0]
        labels = res_test[2][:, 0]
        preds = res_test[2][:, 1]
        auc_test = metrics.roc_auc_score(labels, preds)
        if arg_space.verbose:
            print("Test Loss: %.3f and AUC: %.2f" %
                  (test_loss, auc_test), "\n")
        auprc_test = metrics.average_precision_score(labels, preds)
        some_res = [['Test_Loss', 'Test_AUC', 'Test_AUPRC']]
        some_res.append([test_loss, auc_test, auprc_test])
        # ---Calculate roc and prc values---#
        fpr, tpr, thresholds = metrics.roc_curve(labels, preds)
        precision, recall, thresholdsPR = metrics.precision_recall_curve(
            labels, preds)
        roc_dict = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
        prc_dict = {'precision': precision,
                    'recall': recall, 'thresholds': thresholdsPR}
        # ---Store results----#
        with open(output_dir+'/'+prefix+'_roc.pckl', 'wb') as f:
            pickle.dump(roc_dict, f)
        with open(output_dir+'/'+prefix+'_prc.pckl', 'wb') as f:
            pickle.dump(prc_dict, f)
        np.savetxt(output_dir+'/'+prefix+'_results.txt',
                   some_res, fmt='%s', delimiter='\t')
    else:
        arg_space.storeInterCNN = False
        res_test = evaluateRegularMC(net, device, test_loader, criterion, output_dir+"/Stored_Values",
                                     ent_loss, ent_reg, getPAttn=genPAttn,
                                     storePAttn=arg_space.storeInterCNN, getCNN=getCNNout,
                                     storeCNNout=arg_space.storeInterCNN, getSeqs=getSequences)
        test_loss = res_test[0]
        test_auc = res_test[1]
        if arg_space.verbose:
            print("Test Loss and mean AUC: ", test_loss, np.mean(test_auc))
        np.savetxt(output_dir+'/per_class_AUC.txt',
                   test_auc, delimiter='\t', fmt='%s')
        np.savetxt(output_dir+'/median_mean_AUC.txt',
                   np.array([np.median(test_auc) , np.mean(test_auc)]), delimiter='\t', fmt='%s')

    CNNWeights = net.layer1[0].weight.cpu().detach().numpy()
    res_blob = {'res_test': res_test,
                'train_loader': train_loader,
                'train_indices': train_indices,
                'test_loader': test_loader,
                'test_indices': test_indices,
                'CNN_weights': CNNWeights,
                'criterion': criterion,
                'output_dir': output_dir,
                'net': net,
                'optimizer': optimizer,
                'saved_model_dir': saved_model_dir
                }
    writer.flush()

    return res_blob


def get_results_for_shuffled(argSpace, params, net, criterion, test_loader, device):
    genPAttn = params['get_pattn']
    getCNNout = params['get_CNNout']
    getSequences = params['get_seqs']
    batchSize = params['batch_size']
    num_labels = argSpace.numLabels
    output_dir = argSpace.directory
    bg_prefix = get_shuffled_background(test_loader, argSpace, pre_saved=True)
    if argSpace.deskLoad == True:
        data_bg = DatasetLazyLoad(bg_prefix, num_labels)
    else:
        data_bg = DatasetLoadAll(bg_prefix, num_labels)
    test_loader_bg = DataLoader(
        data_bg, batch_size=batchSize, num_workers=argSpace.numWorkers)
    if num_labels == 2:
        res_test_bg = evaluateRegular(net, device, test_loader_bg, criterion, out_dirc=output_dir+"/Temp_Data/Stored_Values", getPAttn=genPAttn,
                                      storePAttn=argSpace.storeInterCNN, getCNN=getCNNout,
                                      storeCNNout=argSpace.storeInterCNN, getSeqs=getSequences)
    else:
        res_test_bg = evaluateRegularMC(net, device, test_loader_bg, criterion, out_dirc=output_dir+"/Temp_Data/Stored_Values", getPAttn=genPAttn,
                                        storePAttn=argSpace.storeInterCNN, getCNN=getCNNout,
                                        storeCNNout=argSpace.storeInterCNN, getSeqs=getSequences)
    print(bg_prefix, res_test_bg[1])
    return res_test_bg, test_loader_bg


def motif_analysis(res_test, CNNWeights, argSpace, params, for_background=False):
    """
    Infer regulatory motifs by analyzing the first CNN layer filters.
    Args:
        res_test: (list) Returned by the experiment function after testing the model.
        CNNWeights: (numpy.ndarray) Weights of the first CNN layer.
        argSpace: The ArgParser object containing values of all the user-specificed arguments.
        for_background: (bool) Determines if the motif analysis is for the positive or the background set.
    """
    num_labels = argSpace.numLabels
    output_dir = argSpace.directory
    if not os.path.exists(output_dir):
        print("Error! output directory doesn't exist.")
        return
    NumExamples = 0
    pos_score_cutoff = argSpace.scoreCutoff
    k = 0  # batch number
    per_batch_labelPreds = res_test[4][k]
    # per_batch_Embdoutput = res_test[5][k]
    CNNoutput = res_test[5][k]
    if argSpace.storeInterCNN:
        with open(CNNoutput, 'rb') as f:
            CNNoutput = pickle.load(f)
    Seqs = np.asarray(res_test[6][k])
    print(pos_score_cutoff)
    if num_labels == 2:
        if for_background and argSpace.intBackground == 'negative':
            tp_indices = [i for i in range(0, per_batch_labelPreds.shape[0]) if (
                per_batch_labelPreds[i][0] == 0 and per_batch_labelPreds[i][1] < (1-pos_score_cutoff))]
            print(tp_indices)
        elif for_background and argSpace.intBackground == 'shuffle':
            tp_indices = [i for i in range(0, per_batch_labelPreds.shape[0])]
        else:
            tp_indices = [i for i in range(0, per_batch_labelPreds.shape[0]) if (
                per_batch_labelPreds[i][0] == 1 and per_batch_labelPreds[i][1] > pos_score_cutoff)]
    else:
        tp_indices = [i for i in range(
            0, per_batch_labelPreds['labels'].shape[0])]
    NumExamples += len(tp_indices)
    CNNoutput = CNNoutput[tp_indices]

    Seqs = Seqs[tp_indices]
    for k in range(1, len(res_test[3])):
        # if argSpace.verbose:
        # print("batch number: ",k)
        per_batch_labelPreds = res_test[4][k]
        # per_batch_Embdoutput = res_test[5][k]
        per_batch_CNNoutput = res_test[5][k]
        if argSpace.storeInterCNN:
            with open(per_batch_CNNoutput, 'rb') as f:
                per_batch_CNNoutput = pickle.load(f)
        per_batch_seqs = np.asarray(res_test[6][k])
        if num_labels == 2:
            if for_background and argSpace.intBackground == 'negative':
                tp_indices = [i for i in range(0, per_batch_labelPreds.shape[0]) if (
                    per_batch_labelPreds[i][0] == 0 and per_batch_labelPreds[i][1] < (1-pos_score_cutoff))]
            elif for_background and argSpace.intBackground == 'shuffle':
                tp_indices = [i for i in range(
                    0, per_batch_labelPreds.shape[0])]
            else:
                tp_indices = [i for i in range(0, per_batch_labelPreds.shape[0]) if (
                    per_batch_labelPreds[i][0] == 1 and per_batch_labelPreds[i][1] > pos_score_cutoff)]
        else:
            tp_indices = [i for i in range(
                0, per_batch_labelPreds['labels'].shape[0])]
        NumExamples += len(tp_indices)
        CNNoutput = np.concatenate(
            (CNNoutput, per_batch_CNNoutput[tp_indices]), axis=0)
        Seqs = np.concatenate((Seqs, per_batch_seqs[tp_indices]))
    if argSpace.tfDatabase == None:
        dbpath = '/s/jawar/h/nobackup/fahad/MEME_SUITE/motif_databases/CIS-BP/Homo_sapiens.meme'
    else:
        dbpath = argSpace.tfDatabase
    if argSpace.tomtomPath == None:
        tomtomPath = '/s/chromatin/h/nobackup/fahad/MEME_SUITE/meme-5.0.3/src/tomtom'
    else:
        tomtomPath = argSpace.tomtomPath
    if for_background and argSpace.intBackground != None:
        motif_dir = output_dir + '/Motif_Analysis_Negative'
    else:
        motif_dir = output_dir + '/Motif_Analysis'
    # print(CNNoutput)
    get_motif(CNNWeights, CNNoutput, Seqs, dbpath, dir1=motif_dir, embd=False,
              data='DNA', tomtom=tomtomPath, tomtompval=argSpace.tomtomPval, tomtomdist=argSpace.tomtomDist)
    return motif_dir, NumExamples


if __name__ == "__main__":

    data_prefix = "/s/chromatin/p/nobackup/Saira/original/satori/data/ToyData/NEWDATA/ctf_40pairs_eq"
    outdir = "../results/pretrain/data-40/"
    device = "cuda"
    num_labels = 2
    dataset = "simulated"
    run_pretraining(device, data_prefix, outdir, dataset, num_labels)
