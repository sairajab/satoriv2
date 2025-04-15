import numpy as np
import os
import pickle
# import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from fastprogress import progress_bar
from sklearn import metrics
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
# local imports
from datasets import DatasetLoadAll, DatasetLazyLoad
from extract_motifs import get_motif
from satori.modelsold import AttentionNet
from utils import get_shuffled_background
from cnn_weights import *
import re
from satori.experiment_2 import load_datasets


def read_gt_motifs_location(annotation_file, valid_data):
    f_lines = open(annotation_file, "r").readlines()
    intersection = []
    motifs_count = {}
    print(valid_data)
    pattern = r'(\w+)\[(\d+)\]'
    ids_to_motifs = {}

    for i in f_lines[1:]:
        intersected_list = []
        data = i.split('\t')
        #print(data[-2])
        idx = int(data[-2])
        if idx in valid_data:
            pos_to_motifs = {}

            #print("yes")
            matches = re.findall(pattern, data[-1])

            # line = data[-1].rstrip()[2:].split("[")

            # Iterate through the matches and print the results
            for match in matches:
                up_down_positions = range(int(match[1]) - 2, int(match[1])+3, 1)
                for p in up_down_positions:
                    pos_to_motifs[p] = match[0]
            ids_to_motifs[idx] = pos_to_motifs
    return ids_to_motifs


def read_gt_motifs_location_ind(annotation_file, valid_data):
    f_lines = open(annotation_file, "r").readlines()[1:]
    intersection = []
    motifs_count = {}
    pattern = r'(\w+)\[(\d+)\]'
    ids_to_motifs = {}


    #for batch_idx, (headers, seqs, data, target) in enumerate(iterator):
    #for item in range(len(headers)):
    for item in valid_data:
            #header = headers[item]
            #id = int(headers[item].split("-")[-1][:-3])
            line = f_lines[item]
            intersected_list = []
            data = line.split('\t')
            #print(data[-2])
            idx = int(data[-2])
            pos_to_motifs = {}

                #print("yes")
            matches = re.findall(pattern, data[-1])

                # line = data[-1].rstrip()[2:].split("[")

                # Iterate through the matches and print the results
            for match in matches:
                    up_down_positions = range(int(match[1]) - 2, int(match[1])+3, 1)
                    for p in up_down_positions:
                        pos_to_motifs[p] = match[0]
            ids_to_motifs[idx] = pos_to_motifs
        
    return ids_to_motifs


def evaluate_cnn_motif(net, device, iterator, out_dirc, ids_to_motifs, valid_indices):
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
    per_filter_motif_count = {}
    print("valid keys len", len(ids_to_motifs.keys()))
    c = 0
    for filter in range(200):
         per_filter_motif_count[filter] = {}
    with torch.no_grad():
        for batch_idx, (headers, seqs, data, target) in enumerate(iterator):
            data, target = data.to(device, dtype=torch.float), target.to(
                device, dtype=torch.long)
            outputCNN = CNNlayer(data).cpu().detach().numpy()
            #print(headers)
            for item in range(len(headers)):
                #print(headers[item].split("-")[-1],headers[item].split("-")[-1][:-3] )
                id = int(headers[item].split("-")[-1][:-3])
                if id in ids_to_motifs.keys():
                    c = c + 1

                    #print(id)
                    #print(ids_to_motifs[id])
                    #print("HELLO")
                    #print(outputCNN.shape)

                    for filter in range(outputCNN.shape[1]):
                       
                        max_ind = outputCNN[item, filter, :].argmax()
                        max_value = outputCNN[item, filter, max_ind]
                        threshold = 0.60 * max_value
                        indicies = np.where(outputCNN[item, filter, :] > threshold)
                        #print(indicies[0])
                        for ind in indicies[0]:
                            if ind in ids_to_motifs[id]:
                                #print(ids_to_motifs[id][max_ind])
                                if ids_to_motifs[id][ind] not in per_filter_motif_count[filter]:
                                    per_filter_motif_count[filter].update({ids_to_motifs[id][ind] : 1})
                                else:
                                    per_filter_motif_count[filter][ids_to_motifs[id][ind]] += 1
        print("Valid loader count ", c)
        motifs_count_total = {}
        for i in ids_to_motifs:
            for j in range(0, len(ids_to_motifs[i].keys()), 5):    
                p = list(ids_to_motifs[i].keys())[j]
                if ids_to_motifs[i][p] not in motifs_count_total:
                    motifs_count_total[ids_to_motifs[i][p]] = 1
                else:
                    motifs_count_total[ids_to_motifs[i][p]] += 1
        print("total motifs ", len(motifs_count_total))
        print(per_filter_motif_count)
        motifs_found = 0    
        filter_to_motif = {}
        for k in per_filter_motif_count:
            max_count_motif = 0
            max_count = 0
            for motif in per_filter_motif_count[k]:
                 if per_filter_motif_count[k][motif] > max_count:
                     max_count_motif = motif
                     max_count = per_filter_motif_count[k][motif]
                     
            if motifs_count_total[max_count_motif] * 0.70 <= max_count:
                    filter_to_motif[k] = max_count_motif
                    motifs_found += 1 
            else:
                    filter_to_motif[k] = '.'
        print(filter_to_motif, motifs_found)
        
    
    if not os.path.exists(out_dirc + "CNN_filters/"):
        os.mkdir(out_dirc + "CNN_filters/")
        
    file = open(out_dirc + "CNN_filters/" + "table_cnn.txt", "w")
    file_tsv = open(out_dirc + "CNN_filters/" + "cnnfilters.tsv", "w")
    
    file_tsv.writelines("Query_ID\tTarget_ID\tOptimal_offset\tp-value\tE-value\tq-value\tOverlap\tQuery_consensus\tTarget_consensus\tOrientation\n")
    file.writelines("filter          annotation\n")
    
    for filter in filter_to_motif:
        file.writelines(str(filter) + "          " + filter_to_motif[filter] +"\n")
        if filter_to_motif[filter] != ".":
            file_tsv.writelines("filter"+str(filter) + "\t" + filter_to_motif[filter] +
                                "\t0\t0.000\t0.0005\t0.0001\t13\tNNNNNN\tNNNNNN\t+"+"\n")

        
    
    file.close()
    file_tsv.close()
                
                
                
            


def motif_analysis_cnn(device, arg_space, params):

    num_labels = arg_space.numLabels
    load_cnn_ws = False#arg_space.load_motif_weights

    genPAttn = params['get_pattn']
    getCNNout = params['get_CNNout']
    getSequences = params['get_seqs']
    batch_size = params['batch_size']
    max_epochs = params['num_epochs']
    annotation_file = arg_space.inputprefix + "_info.txt"

    # Using generic, not sure if we need it as an argument or part of the params dict
    prefix = 'modelRes'
    train_loader, train_indices, test_loader, test_indices, valid_loader, valid_indices, output_dir = load_datasets(
        arg_space, batch_size)
    if load_cnn_ws:
        params['CNN_filtersize'] = [21]

    ids_to_motifs = read_gt_motifs_location_ind(annotation_file, valid_indices)

    net = AttentionNet(arg_space, params, device=device).to(device)
    if load_cnn_ws:
        weights_tensor = load_cnn_weights(
            output_dir, tfs_pairs_path="create_dataset/tf_pairs_60.txt").to(device)
        net.layer1[0].weight = nn.parameter.Parameter(
            weights_tensor, requires_grad=False)

    saved_model_dir = output_dir+'/Saved_Model'
    optimizer = optim.SGD(net.parameters(), lr=0.01)


    try:
        checkpoint = torch.load(saved_model_dir+'/model')
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    except:
        raise Exception(
            f"No pre-trained model found at {saved_model_dir}! Please run with --mode set to train.")

    evaluate_cnn_motif(net, device, valid_loader, output_dir+"/Motif_Analysis/", ids_to_motifs, valid_indices)
