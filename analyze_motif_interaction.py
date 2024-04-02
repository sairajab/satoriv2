import numpy as np
import os
from glob import glob
'''
Flatten 2D Input into 1D 
'''
def flatten(input):
    new_list = []
    for i in input:
        for j in i:
            new_list.append(j)
    return new_list

'''
Given a TF pairs file (.txt) it returns list of pairs and list of unique TFs
'''
def get_tf_pairs(tf_pairs_file):
    f = open(tf_pairs_file, "r")
    pairs = []
    for x in f:
        tf1, tf2 = x.split('\t')
        pairs.append([tf1, tf2.strip()])
    uniq_tfs = list(set(flatten(pairs)))
    return pairs, uniq_tfs
'''
Read motifs names correspoding to each filter from table.txt
'''

def read_motifs_table(p):
    f = open(p, "r")
    filters_tf = {}
    i = 0
    c_filter = []
    for x in f:
        if i > 0:
            data = x.split(' ')
            data = [j for j in data if j]
            filters_tf[data[0]] = data[2].strip()
            if data[1].strip() != ".":
                c_filter.append(data[2].strip())
        i = i + 1
    print(len(set(c_filter)))
    return filters_tf
'''
Read motifs names correspoding to each filter from table_multiple.txt, this has more than one motifs mappings for motifs
'''
def read_motifs_table_multiple(p):
    ## copies previous code
    f = open(p, "r")
    filters_tf = {}
    i = 0
    c_filter = []
    for x in f:
        if i > 0:
            data = x.split(' ')
            data = [j for j in data if j]
            filters_tf[data[0]] = data[1].strip()
            if data[1].strip() != ".":
                c_filter.append(data[1].strip())
        i = i + 1
    print(len(set(c_filter)))
    return filters_tf

'''
Translate motif_id to TF name using MEME file. Returns a dictionary
'''
def load_meme(meme_db_file):

    motif_protein = {}
    motif_to_tf = {}
    tfs = []
    for line in open(meme_db_file, "r"):
        a = line.split()
        if len(a) > 0 and a[0] == 'MOTIF':
            if a[2][0] == '(':
                motif_protein[a[1]] = a[2][1:a[2].find(')')]
                last_protien = motif_protein[a[1]]
                motif_to_tf[a[1]] = last_protien
                temp = []
            else:
                motif_protein[a[1]] = a[2]
                last_protien = motif_protein[a[1]]
                motif_to_tf[a[1]] = last_protien
                temp = []
            #tfs.append(last_protien)

    return motif_to_tf
'''
Reads tomtom generated .tsv file and maps each filter to the first motif (filter -> motif_id)
'''
def read_motif_meme(motif_dir_pos , meme_db_file, by_default_index = 0):

    filters_tf = {}

    motif_to_tf = load_meme(meme_db_file)

    tomtom_data = np.loadtxt(motif_dir_pos+'/tomtom/tomtom.tsv',dtype=str,delimiter='\t')

    for t in tomtom_data[1:]:

        if t[0][6:] not in filters_tf:

            filters_tf[t[0][6:]] = t[1]

    return motif_to_tf, filters_tf
'''
This function reads filters_to_motif.txt when 1st CNN layer is initialized by the motifs pwms.  
'''
def read_filters_to_motif(filter_to_motif_file , by_default_index = 0):

    # this function is for using pre loaded weights from meme db of CNN layer. 
    
    filters_tf = {}

    f = open(filter_to_motif_file, "r")
    filters_tf = {}

    for x in f:
        data = x.split('\t')
        data = [j for j in data if j]
        filters_tf[data[0]] = data[1].strip()

    #print(filters_tf)
    return filters_tf

    
'''

'''
def evaluate_interactions(interactions_file, thresholds, tfs_pairs, filters_to_motif, output_folder, motif_to_tf = None, head = ""):

    output_file = output_folder + "interaction_results_" + head +"_" + interactions_file.split("-")[-1]
    print(output_file)
    unknown_f = open(output_folder + "unknown_interaction_" + head +"_" +interactions_file.split("-")[-1] , "w")
    known_f = open(output_folder + "known_interaction_" + head +"_" + interactions_file.split("-")[-1] , "w")
    missing_f = open(output_folder + "missed_interaction_" + head +"_"  + interactions_file.split("-")[-1] , "w")

    res = open(output_file, "w")
    ps = []
    rs = []
    f1s = []
    
    for th in thresholds:

        interacting_tfs = []
        examples = []
        i = 0
        m = open(interactions_file, "r")
        for x in m:
            if i > 0:
                data = x.split('\t')
                if float(data[-1]) < th:
                    filters = data[0].split('<')
                    f1 = filters[0].split('r')[1]
                    f2 = filters[1].split('r')[1]
                    #print(f1,f2)
                    interacting_tfs.append((f1, f2))
            i = i + 1
        satori = []
        usatori = []
        count = 0
        for interacting_tf in interacting_tfs:
            if motif_to_tf: #when directly reading a tsv file, filter is mapped to motif_id, which needs to be converted to TF name. 
                tf1 = motif_to_tf[filters_to_motif[interacting_tf[0]]]
                tf2 = motif_to_tf[filters_to_motif[interacting_tf[1]]]
            else:
                tf1 = filters_to_motif[interacting_tf[0]]
                tf2 = filters_to_motif[interacting_tf[1]]
                
            if tf1 == tf2:
                continue
            else:
                for tfs_pair in tfs_pairs:
                    if (tf1 in tfs_pair) and (tf2 in tfs_pair):
                        satori.append([tf1, tf2])
                        already_known = True
                        break;
                    else:
                        already_known = False
                if not already_known:
                    usatori.append([tf1, tf2])
            count = count + 1
        no_dupes = [x for n, x in enumerate(satori) if x not in satori[:n] and [x[1], x[0]] not in satori[:n]]

        if th == 0.05:
            for tf in no_dupes:
                known_f.writelines(tf[0] + "\t"+ tf[1]  + "\n")

        tp = len(no_dupes)
        Missing = []
        count = 0
        for tf in tfs_pairs:
            if [tf[0], tf[1]] not in no_dupes and [tf[1], tf[0]] not in no_dupes:
                Missing.append(tf)
                if th == 0.05:
                    missing_f.writelines(tf[0] + "\t"+ tf[1] + "\n")
        fn = len(Missing)
        print("Count " , tp, fn)
        unknown = []
        uno_dupes = [x for n, x in enumerate(usatori) if x not in usatori[:n] and [x[1], x[0]] not in usatori[:n]]
        for tf in uno_dupes:
            print(tf , count)

            if tf not in tfs_pairs and [tf[1], tf[0]] not in tfs_pairs:
                unknown.append(tf)
                if th == 0.05:
                    unknown_f.writelines(tf[0] + "\t"+ tf[1]  + "\n")
        
        fp = len(unknown)
        recall = float(tp) / float(tp + fn)
        if tp + fp == 0:
            precision = 0
            f1_score = 0
        else:
            precision = float(tp) / float(tp + fp)
            if precision + recall == 0:
                f1_score = 0
            else:
                f1_score = 2 * ((precision * recall) / (precision + recall))


        ps.append(precision)
        rs.append(recall)
        f1s.append(f1_score)

        res.writelines(str(precision) + '\t' + str(recall) + '\t' +str(f1_score) + '\t' +str(th) + '\n')
    return ps, rs, f1s



def run_interaction_evaluation(exp_path, gt_tfs_path, method = "SATORI", motif_weights=False):

    output_folder =  exp_path
    '''
    Use following lines when you want to exclude sorting function and use first entry in tomtom.tsv
    '''
    motif_to_tf, filter_to_motif = read_motif_meme(output_folder + "/Motif_Analysis", "../../motif_databases/Jaspar.meme")

    # uncomment if using table.txt for motif analysis
    # filters_tf = read_motifs_table(output_folder + "/Motif_Analysis/table.txt")
    # ps, rs, f1s = evaluate_interactions(file, thresholds, tfs_pairs, filters_tf, output_folder)
    tfs_pairs, tfs = get_tf_pairs(gt_tfs_path)
    thresholds = np.arange(0, 1.0, 0.0005)

    # Uncomment if you want to use cnn based motif labels
    #filters_tf = read_motifs_table(output_folder + "/Motif_Analysis/CNN_filters/table_cnn.txt")
    if method == "SATORI":
        interaction_files = glob(output_folder + "/Interactions_SATORI/interactions_summary_attnLimit-*.txt")
    else:
        interaction_files = glob(output_folder + "/Interactions_FIS/interactions_summary_attnLimit-*.txt")

    for file in interaction_files:
        if motif_weights:
            print("Using filters_to_motif.txt......")
            filters_tf = read_filters_to_motif(output_folder + "filters_to_motif.txt")
            ps, rs, f1s = evaluate_interactions(file, thresholds, tfs_pairs, filters_tf, output_folder)    
        else:
            print("Using tomtom.txt......")
            print(filter_to_motif)
            print("TF pairs length " , len(tfs_pairs))
            ps, rs, f1s = evaluate_interactions(file, thresholds, tfs_pairs, filter_to_motif, output_folder , motif_to_tf)

    return ps, rs, f1s , thresholds   
    
if __name__ == "__main__" :

    exp_name = "results/newdata/ctf_60pairs_eq0/RNN/E_noseed/"

    pairs_file_gt = "create_dataset/tf_pairs_60.txt"
    
    run_interaction_evaluation(exp_name, pairs_file_gt) #, method = "FIS"

    # motif_to_tf, filter_to_motif = read_motif_meme(output_folder + "Motif_Analysis", "../../motif_databases/Jaspar.meme")

    # gt_tfs_path = "/s/jawar/i/nobackup/Saira/latest/satori_v2/data/ToyData/Ctfs/tf_pairs_data_1.txt"

    # tfs_pairs, tfs = get_tf_pairs(gt_tfs_path)
    
    
    # thresholds = np.arange(0, 1.0, 0.0005)

    # # _ , out = read_motif_meme(output_folder + "Motif_Analysis", "../../motif_databases/Jaspar.meme" )

    # #thresholds = [0.05]

    # # uncomment if using table.txt for motif analysis
    # filters_tf = read_motifs_table(output_folder + "Motif_Analysis/table.txt")
    # print(filters_tf)
    # # uncomment following line if using read_motifs_table
    # interaction_file = glob(output_folder + "/Interactions_SATORI/interactions_summary_attnLimit*.txt")
    # print(interaction_file[0])
    # ps, rs, f1s = evaluate_interactions( interaction_file[0], thresholds, tfs_pairs, filter_to_motif, output_folder , motif_to_tf)

    # for all heads

    os.mkdir(output_folder + "/Interaction_Results")
    output_dir = output_folder + "/Interaction_Results/"


    #for i in range(8):
    #    ps, rs = evaluate_interactions(output_folder + "/Interactions_SATORI/interactions_summary_" + str(i)+"_attnLimit-0.12.txt", thresholds, tfs_pairs, filter_to_motif, output_dir , motif_to_tf, head =str(i))



    # uncomment if using pre loaded motif weights
    #filters_tf = read_filters_to_motif(output_folder + "filters_to_motif.txt")
    #ps, rs = evaluate_interactions(output_folder + "Interactions_SATORI/interactions_summary_attnLimit-0.12.txt", thresholds, tfs_pairs, filters_tf, output_folder)

                
	

'''
    #Results CNN + CNN + Dot Product Attention
    ps_5, rs_5 = evaluate_interactions("cnn_cnn/2-original_satori/interactions_summary_attnLimit-0.12.txt")
#Results CNN + CNN + Relative Attention
filters_tf = read_motifs("cnn_cnn/sqrt_relative/table.txt")
ps_6, rs_6 = evaluate_interactions("cnn_cnn/sqrt_relative/interactions_summary_attnLimit-0.12.txt")
### Old Satori
filters_tf = read_motifs("Interactions_SATORI_0.87/table.txt")
ps_1, rs_1 = evaluate_interactions("Interactions_SATORI_0.87/interactions_summary_attnLimit-0.12.txt")
## CNN + Dot Product Attention
filters_tf = read_motifs("noRNN_satori/table.txt")
ps_2, rs_2 = evaluate_interactions("noRNN_satori/interactions_summary_attnLimit-0.12.txt")
## CNN + RNN +  Relative Attention (with sqrt)
filters_tf = read_motifs("Results_distance/sqrt/table.txt")
ps_3, rs_3 = evaluate_interactions("Results_distance/sqrt/interactions_summary_attnLimit-0.12.txt")
## CNN + RNN +  Relative Attention (with value)
filters_tf = read_motifs("Results_distance/with_val_latest/table.txt")
ps_4, rs_4 = evaluate_interactions("Results_distance/with_val_latest/interactions_summary_attnLimit-0.12.txt")
## CNN + Relative Attention + Dot Product Attention
filters_tf = read_motifs("cnn_attn/table.txt")
ps_7, rs_7 = evaluate_interactions("cnn_attn/interactions_summary_attnLimit-0.12.txt")

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

# plt.ion()
fig, ax = plt.subplots()
trans_offset = mtransforms.offset_copy(ax.transData, fig=fig,
                                       x=0.05, y=0.10, units='inches')

markers_on = [21, 101,201]
plt.plot(rs_1,ps_1 , linestyle='--', marker='o', markevery = markers_on ,label='1')
t =[]
s = []
ths = [0.01 ,  0.05, 0.1]
for m in markers_on:
    t.append(rs_1[m])
    s.append(ps_1[m])

i = 0
for a,b in zip(t, s):
    plt.text(a, b, str(ths[i]), transform=trans_offset, fontsize=8)
    i = i + 1

plt.plot(rs_2, ps_2, marker='v', markevery = markers_on ,label='2')
t =[]
s = []
for m in markers_on[:-1]:
    t.append(rs_2[m])
    s.append(ps_2[m])

i = 0
for a,b in zip(t, s):
    plt.text(a, b, str(ths[i]), transform=trans_offset, fontsize=8) #, color='C1'
    i = i + 1

# Plot for old  Satori

plt.plot(rs_3, ps_3, marker='v', markevery = markers_on ,label='3')
t =[]
s = []
for m in markers_on[:-1]:
    t.append(rs_3[m])
    s.append(ps_3[m])

i = 0
for a,b in zip(t, s):
    plt.text(a, b, str(ths[i]), transform=trans_offset, fontsize=8)
    i = i + 1

plt.plot(rs_4, ps_4, marker='v', markevery = markers_on ,label='4')
t =[]
s = []
for m in markers_on[:-1]:
    t.append(rs_4[m])
    s.append(ps_4[m])

i = 0
for a,b in zip(t, s):
    plt.text(a, b, str(ths[i]), transform=trans_offset, fontsize=8)
    i = i + 1

plt.plot(rs_5, ps_5, marker='v', markevery = markers_on ,label='5')
t =[]
s = []
for m in markers_on[:-1]:
    t.append(rs_5[m])
    s.append(ps_5[m])

i = 0
for a,b in zip(t, s):
    plt.text(a, b, str(ths[i]), transform=trans_offset, fontsize=8)
    i = i + 1


plt.plot(rs_6, ps_6, marker='v', markevery = markers_on ,label='6')
t =[]
s = []
for m in markers_on[:-1]:
    t.append(rs_6[m])
    s.append(ps_6[m])

i = 0
for a,b in zip(t, s):
    plt.text(a, b, str(ths[i]), transform=trans_offset, fontsize=8)
    i = i + 1


plt.plot(rs_7, ps_7, marker='v', markevery = markers_on ,label='7')
t =[]
s = []
for m in markers_on[:-1]:
    t.append(rs_7[m])
    s.append(ps_7[m])

i = 0
for a,b in zip(t, s):
    plt.text(a, b, str(ths[i]), transform=trans_offset, fontsize=8)
    i = i + 1
# axis labels

plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# for i in range(len(thresholds)):
#     if
# ax.annotate('T = 4K', xy=(2,4), xycoords='data',
#             xytext=(-100,60), textcoords='offset points',
#             arrowprops=dict(arrowstyle='fancy',fc='0.6',
#                             connectionstyle="angle3,angleA=0,angleB=-90"))

# show the plot
plt.show()


# plt.plot(rs, ps)
# plt.ylabel('precision')
# plt.xlabel('recall')
#
# plt.show()
'''









