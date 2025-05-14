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
from satori.datasets import DatasetLoadAll, DatasetLazyLoad, DatasetLazyLoadRC
from satori.extract_motifs import get_motif
from satori.models import AttentionNet
from satori.utils import get_shuffled_background, get_shuffled_background_presaved, get_indices
from satori.cnn_weights import *
# from cnn_motif_extraction import *
from torch.utils.tensorboard import SummaryWriter
from satori.train_model import *
from satori.evaluate_model import *
from satori.sei_model import *
from satori.tiana import TianaModel
import json
###########################################################################################################################
# --------------------------------------------Train and Evaluate Functions-------------------------------------------------#
###########################################################################################################################
## Moved to seperate files
###########################################################################################################################
# ---------------------------------------------------------End-------------------------------------------------------------#
###########################################################################################################################


def load_datasets(inputprefix, output_dir, batchSize, dataset_name, numLabels, deskLoad, mode, splitperc, rev_complement = False):
    """
    Loads and processes the data.
    """
    input_prefix = inputprefix
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    test_split = splitperc/100
    print("test/validation split val: %.2f" % test_split)

    if deskLoad:
        if rev_complement:
            final_dataset = DatasetLazyLoadRC(input_prefix,num_labels=numLabels, rev_complement=rev_complement)

        else:
            final_dataset = DatasetLazyLoad(
                input_prefix, num_labels=numLabels)
    else:
        final_dataset = DatasetLoadAll(
            input_prefix, num_labels=numLabels)

    seq_len = final_dataset.get_seq_len()
    train_indices, test_indices, valid_indices = get_indices(
        len(final_dataset), test_split, output_dir, dataset_name, mode=mode)
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)
    train_loader = DataLoader(final_dataset, batch_size=batchSize,
                              sampler=train_sampler, num_workers=4)
    test_loader = DataLoader(final_dataset, batch_size=batchSize,
                             sampler=test_sampler, num_workers=4)
    valid_loader = DataLoader(final_dataset, batch_size=batchSize,
                              sampler=valid_sampler, num_workers=4)
    return train_loader, train_indices, test_loader, test_indices, valid_loader, valid_indices, output_dir, seq_len


def run_experiment(device, arg_space, params, verbose = False):
    """
    Run the main experiment, that is, load the data and train-test the model and generate/store results.
    Args:
        device: (torch.device) Specifies the device (either gpu or cpu).
        arg_space: ArgParser object containing all the user-specified arguments.
        params: (dict) Dictionary of hyperparameters. 
    """
    num_labels = arg_space.numLabels
    load_cnn_ws = params['load_motif_weights']
    genPAttn = params['get_pattn']
    getCNNout = params['get_CNNout']
    getSequences = params['get_seqs']
    batch_size = params['batch_size']
    max_epochs = params['num_epochs']
    ent_loss = params['entropy_loss']
    ent_reg = float(params['entropy_reg_value'])
    if verbose:
        writer = SummaryWriter()

    # Using generic, not sure if we need it as an argument or part of the params dict
    prefix = 'modelRes_Val'
    
    train_loader, train_indices, test_loader, test_indices, valid_loader, valid_indices, output_dir, seq_len = load_datasets(arg_space.inputprefix, arg_space.directory,
                                                                                                                          batch_size, arg_space.dataset, num_labels, arg_space.deskLoad, arg_space.mode, arg_space.splitperc)
    
        # save arguments to keep record
    with open(output_dir + '/arguments.json', 'w') as f:
        json.dump({"arg_space": vars(arg_space), "params": params}, f, indent=4)
        

    getAttnAttr = False
    net = AttentionNet(arg_space.numLabels, params, device=device,
                       seq_len=seq_len, getAttngrad=getAttnAttr).to(device)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Number of parameters: {total_params}")
    
    if params['exp_name'] == 'arabidopsis':
        # load pssm
        pssm = '../TIANA_demo_upload/mycluster/arab/motif_pssm_arab.npy'
    
    elif params['exp_name'] == 'human_promoters':
        
        pssm = "../TIANA_demo_upload/mycluster/hp/motifs_pssm_hp.npy"

    else:
        pssm = '../TIANA_demo_upload/mycluster/padded_motifs_pssm_new.npy'
        # load pssm
    with open(pssm, 'rb') as f:
            motif_array = np.load(f)
            motif_array = motif_array[:, :, ::2]    
    
    # reg = False
    # tiana = False
    # if tiana:
    #     pssm = '../TIANA_demo_upload/mycluster/padded_motifs_pssm_new.npy' #motif_pssm.npy' #
    #     # load pssm
    #     with open(pssm, 'rb') as f:
    #         motif_array = np.load(f)
    #         # motif_array = motif_array[:,:, :360]
            
    #     #number of total tf (half of pssm)
    #     ntf = motif_array.shape[-1]//2
            
    #     # motif_size
    #     motif_size = motif_array.shape[0]
            
    #     # padding size
    #     if ntf%4 ==0:
    #         npad = 0
    #     elif ntf%4 !=0:
    #         npad = 4 - ntf%4
    #     # Example usage
    #     net = TianaModel(num_tf=ntf, pad_size=npad, max_len=motif_size, pssm_path=pssm, seq_size=seq_len, numClasses=num_labels).to(device)
    #     reg = False 
        
    if load_cnn_ws:
            print("Loading CNN weights from PSSM")
            net.layer1[0].weight.data = torch.from_numpy(np.transpose(motif_array, (2, 1, 0))).float().to(device)
            net.layer1[0].weight.requires_grad = False 

    if num_labels == 2:

        criterion = nn.CrossEntropyLoss(reduction='mean')
        if params["optimizer"] == "adam":
            optimizer = optim.Adam(net.parameters(), lr=float(params["lr"]), weight_decay=float(params["weight_decay"]))
        else:
            optimizer = optim.SGD(net.parameters(), lr=float(params["lr"]), momentum=float(params["momentum"]), weight_decay=float(params["weight_decay"]))

    else:
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)
    
    if params['schedular']:
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)


    if arg_space.finetune_model_path != None:
        try:
            checkpoint = torch.load(arg_space.finetune_model_path+'/model')
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print("Loaded model for finetunig...")
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
    best_auprc_valid = 0
    ## -------Main train/test loop----------##
    if arg_space.mode == 'train':
        best_valid_loss = np.inf
        best_valid_auc = np.inf
        with open(output_dir+'/entropy_argument.txt', 'w') as f:
            f.writelines(f"Ent_Regularization Value : {ent_loss} , {ent_reg}" )
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
                                            storeCNNout=False, getSeqs=False,motifweights=load_cnn_ws)  # evaluateRegular(net,valid_loader,criterion)
                res_valid_loss = res_valid[0]
                res_valid_auc = res_valid[1]
            else:
                res_valid = evaluateRegularMC(net, device, valid_loader, criterion, output_dir+"/Stored_Values", getPAttn=False,
                                              storePAttn=False, getCNN=False,
                                              storeCNNout=False, getSeqs=False,motifweights=load_cnn_ws)  # evaluateRegular(net,valid_loader,criterion)
                res_valid_loss = res_valid[0]
                res_valid_auc = np.mean(res_valid[1])
                
            if res_valid_loss < best_valid_loss:
                best_valid_loss = res_valid_loss
                best_valid_auc = res_valid_auc
                labels = res_valid[2][:, 0]
                preds = res_valid[2][:, 1]
                #best_auprc_valid = metrics.average_precision_score(labels, preds)

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
            if params['schedular']:
                scheduler.step(res_valid_loss)
            if verbose:
                writer.add_scalar("Loss/train", res_train_loss, epoch)
                writer.add_scalar("Loss/val", res_valid_loss, epoch)

            # scheduler2.step(res_valid_loss)

            print('Epoch-{0} lr: {1} valid_loss: {2} valid_auc : {3}'.format(epoch,
                  optimizer.param_groups[0]['lr'], res_valid_loss, res_valid_auc))
            logs.writelines('Epoch-{0} lr: {1} valid_loss: {2} \n' .format(
                epoch, optimizer.param_groups[0]['lr'], res_valid_loss))

            if counter >= 15:
                print("Early stopping at ",epoch )
                break
        if num_labels == 2:
            res_valid = evaluateRegular(net, device, valid_loader, criterion, output_dir+"/Stored_Values", ent_loss, ent_reg, getPAttn=False,
                                        storePAttn=False, getCNN=False,
                                        storeCNNout=False, getSeqs=False, motifweights=load_cnn_ws)
        else:
            res_valid = evaluateRegularMC(net, device, valid_loader, criterion, output_dir+"/Stored_Values", ent_loss, ent_reg, getPAttn=False,
                                          storePAttn=False, getCNN=False,
                                          storeCNNout=False, getSeqs=False,motifweights=load_cnn_ws)

        if num_labels == 2:
            # Save valid files
            valid_loss = best_valid_loss
            auprc_valid = best_auprc_valid
            auc_valid = best_valid_auc
            if arg_space.verbose:
                print("Valid Loss: %.3f and AUC: %.2f" %
                      (valid_loss, auc_valid), "\n")

            some_res = [['Valid_Loss', 'Valid_AUC', 'Valid_AUPRC']]
            some_res.append([valid_loss, auc_valid, auprc_valid])
            # # ---Calculate roc and prc values---#
            # fpr, tpr, thresholds = metrics.roc_curve(labels, preds)
            # precision, recall, thresholdsPR = metrics.precision_recall_curve(
            #     labels, preds)
            # roc_dict = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
            # prc_dict = {'precision': precision,
            #             'recall': recall, 'thresholds': thresholdsPR}
            # # ---Store results----#
            # with open(output_dir+'/'+prefix+'_roc.pckl', 'wb') as f:
            #     pickle.dump(roc_dict, f)
            # with open(output_dir+'/'+prefix+'_prc.pckl', 'wb') as f:
            #     pickle.dump(prc_dict, f)
            np.savetxt(output_dir+'/'+prefix+'_results.txt',
                       some_res, fmt='%s', delimiter='\t')

    try:
        checkpoint = torch.load(saved_model_dir+'/model')
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print("Model Loaded Successfully ...", loss)
    except:
        raise Exception(
            f"No pre-trained model found at {saved_model_dir}! Please run with --mode set to train.")

    if num_labels == 2:
        prefix = 'modelRes'
        arg_space.storeInterCNN = False
        ent_loss = params["entropy_loss"]

        
        res_test = evaluateRegular(net, device, test_loader, criterion, output_dir+"/Stored_Values", ent_loss, ent_reg, getPAttn=genPAttn,
                                   storePAttn=arg_space.storeInterCNN, getCNN=getCNNout,
                                   storeCNNout=arg_space.storeInterCNN, getSeqs=getSequences,motifweights=load_cnn_ws, getAttnAttr=getAttnAttr)
        # uncomment for CNN based motif annotations
        # annotation_file = arg_space.inputprefix + "_info.txt"

        # ids_to_motifs = read_gt_motifs_location_ind(annotation_file, valid_indices)
        # print(ids_to_motifs)

        # _ = evaluate_cnn_motif(net, device, valid_loader, output_dir+"/Motif_Analysis/", ids_to_motifs, valid_indices)

        test_loss = res_test[0]
        auc_test = res_test[1]
        labels = res_test[2][:, 0]
        preds = res_test[2][:, 1]
        # auc_test = metrics.roc_auc_score(labels, preds)
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
                                     storeCNNout=arg_space.storeInterCNN, getSeqs=getSequences, motifweights=load_cnn_ws)
        test_loss = res_test[0]
        test_auc = res_test[1]
        if arg_space.verbose:
            print("Test Loss and mean AUC: ", test_loss, np.mean(test_auc))
        np.savetxt(output_dir+'/per_class_AUC.txt',
                   test_auc, delimiter='\t', fmt='%s')
        np.savetxt(output_dir+'/median_mean_AUC.txt',
                   np.array([np.median(test_auc) , np.mean(test_auc)]), delimiter='\t', fmt='%s')
    # if tiana:
    #     CNNWeights = net.conv1.weight.cpu().detach().numpy()
    # else:
    CNNWeights = net.layer1[0].weight.cpu().detach().numpy()
    print(CNNWeights[0,:].max(), CNNWeights.shape)
    
    if arg_space.mode == 'test':
        res_valid = None
        
    res_blob = {'res_test': res_test,
                    'seq_len' : seq_len,
                    'train_loader': train_loader,
                    'train_indices': train_indices,
                    'test_loader': test_loader,
                    'test_indices': test_indices,
                    'CNN_weights': CNNWeights,
                    'criterion': criterion,
                    'output_dir': output_dir,
                    'net': net,
                    'optimizer': optimizer,
                    'saved_model_dir': saved_model_dir,
                    'res_valid' : res_valid
                }
    if verbose:
        writer.flush()

    return res_blob


def get_results_for_shuffled(argSpace, params, net, criterion, test_loader, device):
    genPAttn = params['get_pattn']
    getCNNout = params['get_CNNout']
    getSequences = params['get_seqs']
    batchSize = params['batch_size']
    motifweights = params['load_motif_weights']
    rev_complement = params['rev_complement']
    num_labels = argSpace.numLabels
    output_dir = argSpace.directory
    bg_prefix = get_shuffled_background(test_loader, argSpace, pre_saved=True)
    if rev_complement:
        print("Loading background data with reverse complement")
        data_bg = DatasetLazyLoadRC(bg_prefix, num_labels=num_labels, rev_complement=rev_complement)
    else:
        if argSpace.deskLoad == True:
            data_bg = DatasetLazyLoad(bg_prefix, num_labels)
        else:
            data_bg = DatasetLoadAll(bg_prefix, num_labels)
    test_loader_bg = DataLoader(
        data_bg, batch_size=batchSize, num_workers=argSpace.numWorkers)
    if num_labels == 2:
        res_test_bg = evaluateRegular(net, device, test_loader_bg, criterion, out_dirc=output_dir+"/Temp_Data/Stored_Values", getPAttn=genPAttn,
                                      storePAttn=argSpace.storeInterCNN, getCNN=getCNNout,
                                      storeCNNout=argSpace.storeInterCNN, getSeqs=getSequences,motifweights=motifweights)
    else:
        print("MOTIF WEIGHTS ", motifweights)
        res_test_bg = evaluateRegularMC(net, device, test_loader_bg, criterion, out_dirc=output_dir+"/Temp_Data/Stored_Values", getPAttn=genPAttn,
                                        storePAttn=argSpace.storeInterCNN, getCNN=getCNNout,
                                        storeCNNout=argSpace.storeInterCNN, getSeqs=getSequences, motifweights=motifweights)
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
    CNNoutput = res_test[5][k]
    if argSpace.storeInterCNN:
        with open(CNNoutput, 'rb') as f:
            CNNoutput = pickle.load(f)
    Seqs = np.asarray(res_test[6][k])
    if num_labels == 2:
        if for_background and argSpace.intBackground == 'negative':
            neg_score_cutoff = 1 - pos_score_cutoff #0.4
            tp_indices = [i for i in range(0, per_batch_labelPreds.shape[0]) if (
                per_batch_labelPreds[i][0] == 0 and per_batch_labelPreds[i][1] < (neg_score_cutoff))]
            print("Negative Background", tp_indices, 1 - pos_score_cutoff)
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
        per_batch_labelPreds = res_test[4][k]
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
    get_motif(CNNWeights, CNNoutput, Seqs, dbpath, dir1=motif_dir, embd=False,
              data='DNA', tomtom=tomtomPath, tomtompval=argSpace.tomtomPval, tomtomdist=argSpace.tomtomDist, motifweights = params['load_motif_weights'], dataset=params['exp_name'])
    return motif_dir, NumExamples

