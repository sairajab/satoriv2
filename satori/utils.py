import numpy as np
import os
import random
import torch

from Bio.motifs import minimal
from fastprogress import progress_bar
from sklearn import metrics
from collections import defaultdict
from random import shuffle
import json
##############################################
#Taken from: https://github.com/kundajelab/deeplift/blob/master/deeplift/dinuc_shuffle.py
#############################################


def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)  
    
#random.seed = 100
def load_config(config_path):
    """Load parameters from a JSON configuration file."""
    with open(config_path, 'r') as json_file:
        config = json.load(json_file)
    return config

#compile the dinucleotide edges
def prepare_edges(s):
    edges = defaultdict(list)
    for i in range(len(s)-1):
        edges[tuple(s[i])].append(s[i+1])
    return edges


def shuffle_edges(edges, rng=None):
    #for each character, remove the last edge, shuffle, add edge back
    for char in edges:
        last_edge = edges[char][-1]
        edges[char] = edges[char][:-1]
        the_list = edges[char]
        if (rng is None):
            shuffle(the_list)
        else:
            rng.shuffle(the_list)
        edges[char].append(last_edge)
    return edges


def traverse_edges(s, edges):
    generated = [s[0]]
    edges_queue_pointers = defaultdict(lambda: 0)
    for i in range(len(s)-1):
        last_char = generated[-1]
        generated.append(edges[tuple(last_char)][edges_queue_pointers[tuple(last_char)]])
        edges_queue_pointers[tuple(last_char)] += 1
    if isinstance(generated[0],str):
        return "".join(generated)
    else:
        import numpy as np
        return np.asarray(generated)


def dinuc_shuffle(s, rng=None):
    if isinstance(s, str):
        s=s.upper()
    return traverse_edges(s, shuffle_edges(prepare_edges(s), rng))



class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

    @classmethod
    def from_json(cls, file_path):
        with open(file_path, "r") as f:
            config_dict = json.load(f)
        return cls(config_dict)


def get_params_dict(params_path):
    param_data = np.loadtxt(params_path, dtype=str, delimiter='|')
    params = {}
    for entry in param_data:
        if entry[1] == 'False':
            params[entry[0]] = False
        elif entry[1] == 'True':
            params[entry[0]] = True
        else:
            try:
                params[entry[0]] = int(entry[1])
            except:
                if entry[1][0] == "[" and entry[1][-1] == "]":
                    list_elem = entry[1][1:-1].split(",")
                    elems = []
                    for el in list_elem:
                        elems.append(int(el))
                    params[entry[0]] = elems
                else:
                    params[entry[0]] = entry[1]
    return params


def calculate_padding(inputLength, filterSize):
    padding = inputLength - (inputLength - filterSize + 1)
    return int(padding/2)  # appended to both sides the half of what is needed


def annotate_motifs(annotate_arg, motif_dir, store=True):
    ### -----------------Adding TF details to TomTom results----------------###
    try:
        tomtom_res = np.loadtxt(
            motif_dir+'/tomtom/tomtom.tsv', dtype=str, delimiter='\t')
    except:  # shouldn't stop the flow of program if this fails
        print("Error! motif file not found. Make sure to do motif analysis first.")
        return
    # this for now makes sure that we don't annotate (will be added in future).
    if annotate_arg == 'default':
        database = np.loadtxt(
            '../../../Basset_Splicing_IR-iDiffIR/Analysis_For_none_network-typeB_lotus_posThresh-0.60/MEME_analysis/Homo_sapiens_2019_01_14_4_17_pm/TF_Information_all_motifs.txt', dtype=str, delimiter='\t')
    else:
        database = np.loadtxt(annotate_arg, dtype=str, delimiter='\t')
    final = []
    for entry in tomtom_res[1:]:
        motifID = entry[1]
        res = np.argwhere(database[:, 3] == motifID)
        TFs = ','.join(database[res.flatten(), 6])
        final.append(entry.tolist()+[TFs])
    if store:
        np.savetxt(motif_dir+'/tomtom/tomtom_annotated.tsv',
                   final, delimiter='\t', fmt='%s')
    else:
        return final

def get_indices(dataset_size, test_split, output_dir, data, shuffle_data=True, seed_val=100, mode='train', rev_complement=False, final_dataset=None):
    indices = list(range(dataset_size))
    split_val = int(np.floor(test_split*dataset_size))
    if shuffle_data:
        rng = np.random.RandomState(seed_val)
        rng.shuffle(indices)
    # --save indices for later use, when testing for example---#
    if mode == 'train':

        if data == "human_promoters":
            if rev_complement:
                valid_indices = np.loadtxt(
                    '/s/chromatin/p/nobackup/Saira/original/satori/data/Human_Promoters/valid_indices_rev.txt', dtype=int)
                test_indices = np.loadtxt(
                    '/s/chromatin/p/nobackup/Saira/original/satori/data/Human_Promoters/test_indices_rev.txt', dtype=int)
                train_indices = np.loadtxt(
                    '/s/chromatin/p/nobackup/Saira/original/satori/data/Human_Promoters/train_indices_rev.txt', dtype=int)
            
            else:
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
          try:
              valid_indices = np.loadtxt(
                  output_dir+'/valid_indices.txt', dtype=int)
              test_indices = np.loadtxt(
                  output_dir+'/test_indices.txt', dtype=int)
              train_indices = np.loadtxt(
                  output_dir+'/train_indices.txt', dtype=int)
          except:
                train_indices, test_indices, valid_indices = np.array(
                    indices[2*split_val:]), np.array(indices[:split_val]), np.array(indices[split_val:2*split_val])
    
                np.savetxt(output_dir+'/valid_indices.txt', valid_indices, fmt='%s')
                np.savetxt(output_dir+'/test_indices.txt', test_indices, fmt='%s')
                np.savetxt(output_dir+'/train_indices.txt', train_indices, fmt='%s')

    else:
        try:
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


def get_popsize_for_interactions(argSpace, per_batch_labelPreds, batchSize):
    pos_score_cutoff = argSpace.scoreCutoff
    num_labels = argSpace.numLabels
    numPosExamples = 0
    if argSpace.useAll == False and argSpace.numLabels != 2:
        bg_indices_multiLabel = []
    for j in range(0, len(per_batch_labelPreds)):
        if numPosExamples == argSpace.intSeqLimit:
            break
        batch_values = per_batch_labelPreds[j]
        if num_labels != 2:
            batch_preds = batch_values['preds']
            batch_values = batch_values['labels']
        for k in range(0, len(batch_values)):
            if num_labels == 2:
                if (batch_values[k][0] == 1 and batch_values[k][1] > pos_score_cutoff):
                    numPosExamples += 1
            else:
                if argSpace.useAll == True:
                    numPosExamples += 1
                else:
                    ex_labels = batch_values[k].astype(int)
                    ex_preds = batch_preds[k]
                    ex_preds = np.asarray(
                        [i >= 0.5 for i in ex_preds]).astype(int)
                    prec = metrics.precision_score(ex_labels, ex_preds)
                    if prec >= argSpace.precisionLimit:
                        numPosExamples += 1
                        bg_indices_multiLabel.append([j*batchSize + k, prec])
            if numPosExamples == argSpace.intSeqLimit:
                break
    if argSpace.useAll == False and argSpace.numLabels != 2:
        bg_indices_multiLabel = np.asarray(bg_indices_multiLabel)[
            :, 0].astype(int)

    numNegExamples = 0
    for j in range(0, len(per_batch_labelPreds)):
        if numNegExamples == argSpace.intSeqLimit:
            break
        batch_values = per_batch_labelPreds[j]
        if num_labels != 2:
            batch_values = batch_values['labels']
        for k in range(0, len(batch_values)):
            if num_labels == 2:
                if argSpace.intBackground == 'negative':
                    if (batch_values[k][0] == 0 and batch_values[k][1] < (1-pos_score_cutoff)):
                        numNegExamples += 1
                elif argSpace.intBackground == 'shuffle':
                    numNegExamples += 1
            else:
                numNegExamples += 1
            if numNegExamples == argSpace.intSeqLimit:
                break
    if argSpace.useAll == False and argSpace.numLabels != 2:
        numNegExamples = len(bg_indices_multiLabel)
    if argSpace.verbose:
        print('Positive and Negative Population: ',
              numPosExamples, numNegExamples)
    return numPosExamples, numNegExamples


def get_intr_filter_keys(num_filters=200):
    Filter_Intr_Keys = {}
    count_index = 0
    for i in range(0, num_filters):
        for j in range(0, num_filters):
            if i == j:
                continue
            intr = 'filter'+str(i)+'<-->'+'filter'+str(j)
            rev_intr = 'filter'+str(j)+'<-->'+'filter'+str(i)
            if intr not in Filter_Intr_Keys and rev_intr not in Filter_Intr_Keys:
                Filter_Intr_Keys[intr] = count_index
                count_index += 1
    return Filter_Intr_Keys


def get_random_seq(pwm, alphabet=['A', 'C', 'G', 'T']):
    seq = ''
    for k in range(0, pwm.shape[0]):
        nc = np.random.choice(alphabet, 1, p=pwm[k, :])
        seq += nc[0]
    return seq


def get_shuffled_background_presaved():
    out_directory = "/".join(argSpace.inputprefix.split("/")[:-1])
    print("Shuffled data path ", out_directory)
    return out_directory+'/'+'shuffled_background'


# this function is similar to the first one (get_shuffled_background()) however, instead of using consensus, it generates a random sequences (of same size as the PWM) based on the probability distributions in the matrix
def get_shuffled_background(tst_loader, argSpace, pre_saved=False):
    if pre_saved:
        out_directory = "/".join(argSpace.inputprefix.split("/")[:-1])
        print("Shuffled data path ", out_directory)

    else:
        out_directory = argSpace.directory+'/Temp_Data'
        if argSpace.mode == 'test':
            if pre_saved and os.path.exists(out_directory+'/'+'shuffled_background.txt') and os.path.exists(out_directory+'/'+'shuffled_background.fa'):
                return out_directory+'/'+'shuffled_background'  # name of the prefix to use
            else:
                print("Shuffled data missing! Regenerating...")
        labels_array = np.asarray([i for i in range(0, argSpace.numLabels)])
        final_fa = []
        final_bed = []
        for batch_idx, (headers, seqs, _, batch_targets) in enumerate(tst_loader):
            for i in range(0, len(headers)):
                header = headers[i]
                seq = seqs[i]
                targets = batch_targets[i]
                seq = dinuc_shuffle(seq)
                hdr = header.strip('>').split('(')[0]
                chrom = hdr.split(':')[0]
                start, end = hdr.split(':')[1].split('-')
                final_fa.append([header, seq])
                if type(targets) == torch.Tensor:
                    targets = targets.cpu().detach().numpy()
                target = targets.astype(int)
                labels = ','.join(
                    labels_array[np.where(target == 1)].astype(str))
                final_bed.append([chrom, start, end, '.', labels])
        final_fa_to_write = []
        # --load motifs
        try:
            with open(argSpace.directory+'/Motif_Analysis/filters_meme.txt') as f:
                filter_motifs = minimal.read(f)
        except:
            raise Exception(
                "motif file not found! Make sure to extract motifs first from the network output (hint: use --motifanalysis)")
        motif_len = filter_motifs[0].length
        seq_numbers = [i for i in range(0, len(final_bed))]
        # can't go beyond end of sequence so have to subtract motif_len
        seq_positions = [i for i in range(0, len(final_fa[0][1])-motif_len)]
        for i in progress_bar(range(0, len(filter_motifs))):
            motif = filter_motifs[i]
            pwm = np.column_stack(
                (motif.pwm['A'], motif.pwm['C'], motif.pwm['G'], motif.pwm['T']))
            # consensus = motif.consensus
            num_occ = motif.num_occurrences
            # randomly picking num_occ sequences (note that num_occ can be greater than population in this case since a single sequence can have multile occurence of a filter activation)
            random_seqs = random.choices(seq_numbers, k=num_occ)
            # print(num_occ, len(seq_positions))
            # randomly pick a position for a motif to occur
            random_positions = random.choices(seq_positions, k=num_occ)
            for seq_index, pos in zip(random_seqs, random_positions):
                # this will get us a random sequence generated based on the prob distribution of the PWM
                consensus = get_random_seq(pwm)
                seq = final_fa[seq_index][1]
                seq = seq[:pos]+str(consensus)+seq[pos+len(consensus):]
                final_fa[seq_index][1] = seq
        if not os.path.exists(out_directory):
            os.makedirs(out_directory)
        np.savetxt(out_directory+'/'+'shuffled_background.fa',
                   np.asarray(final_fa).flatten(), fmt='%s')
        np.savetxt(out_directory+'/'+'shuffled_background.txt',
                   np.asarray(final_bed), fmt='%s', delimiter='\t')
    return out_directory+'/'+'shuffled_background'  # name of the prefix to use
