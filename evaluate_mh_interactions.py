import numpy as np
from analyze_motif_interaction import get_tf_pairs
import os
tfs_pairs, tfs = get_tf_pairs('/s/jawar/i/nobackup/Saira/latest/satori_v2/data/ToyData/Ctfs/tf_pairs_data_1.txt')

def read_motifs(p):
    f = open(p, "r")
    filters_tf = {}
    i = 0
    for x in f:
        if i > 0:
            data = x.split(' ')
            data = [j for j in data if j]
            filters_tf[data[0]] = data[2].strip()
        i = i + 1
    print(filters_tf)
    return filters_tf

thresholds = np.arange(0.000000000000000001, 1.0, 0.0005)
dir_path = "results/CNN_S_REL_ATTN_e1/"
filters_tf = read_motifs(os.path.join(dir_path,"Motif_Analysis/table.txt"))#Results_distance/before_val/exp4/

    #[0.1,0.09,0.08,0.07,0.06,0.05, 0.04, 0.03,0.02,0.01]
def evaluate_union_interactions(heads_res):
    ps = []
    rs = []
    f1s = []
    for th in thresholds:
        union_tp = []
        union_fp = []
        for head in heads_res:
            tp, fp = head
            for motif1, motif2 in tp[th]:
                if [motif1, motif2] in union_tp or [motif2, motif1] in union_tp:
                    continue
                else:
                    union_tp.append([motif1, motif2])
            for motif1, motif2 in fp[th]:
                if [motif1, motif2] in union_fp or [motif2, motif1] in union_fp:
                    continue
                else:
                    union_fp.append([motif1, motif2])

        print("TP" + str(len(union_tp)))
        print("FP" + str(len(union_fp)))
        print("FN" + str(80 - len(union_tp)))
        recall = float(len(union_tp)) / float(len(union_tp) + (80 - len(union_tp)))
        if len(union_tp) + len(union_fp) == 0:
                precision = 0
                f1_score = 0
        else:
                precision = float(len(union_tp)) / float(len(union_tp) + len(union_fp))
                if float(precision + recall) == 0:
                    f1_score = 0
                else:
                    f1_score = 2 * (precision * recall) / float(precision + recall)


        ps.append(precision)
        rs.append(recall)
        f1s.append(f1_score)
    return ps, rs, f1s

def evaluate_interactions(file):
    print(file)
    res = open(file.split("Inter")[0] + "/Interaction_Results/" + file.split("/")[-1], "w")
    #open(os.path.join(os.path.join(dir_path, "res") , file.split('\\')[1]), "w")
    ps = []
    rs = []
    f1s = []
    hits_tp = {}
    hits_fp = {}
    for th in thresholds:
        interacting_tfs = []
        examples = []
        i = 0
        m = open(file, "r")
        for x in m:
            if i > 0:
                data = x.split('\t')
                if float(data[-1]) < th:
                    filters = data[0].split('<')
                    f1 = filters[0].split('r')[1]
                    f2 = filters[1].split('r')[1]
                    interacting_tfs.append((f1, f2))
            i = i + 1
        satori = []
        usatori = []
        count = 0
        for interacting_tf in interacting_tfs:
            tf1 = filters_tf[interacting_tf[0]]
            tf2 = filters_tf[interacting_tf[1]]
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
        tp = len(no_dupes)
        Missing = []
        for tf in tfs_pairs:
            if tf not in no_dupes and [tf[1], tf[0]] not in no_dupes:
                Missing.append(tf)
        fn = len(Missing)
        unknown = []
        uno_dupes = [x for n, x in enumerate(usatori) if x not in usatori[:n] and [x[1], x[0]] not in usatori[:n]]
        for tf in uno_dupes:
            if tf not in tfs_pairs and [tf[1], tf[0]] not in tfs_pairs:
                unknown.append(tf)
        fp = len(unknown)
        print(th)
        print("TP" + str(tp))
        print("FP" + str(fp))
        print("FN" + str(fn))
        recall = float(tp) / float(tp + fn)
        if tp + fp == 0:
            precision = 0
            f1_score = 0
        else:
            precision = float(tp) / float(tp + fp)
            if precision + recall == 0:
                f1_score = 0
            else:
                f1_score = 2 * (precision * recall) / float(precision + recall)

        ps.append(precision)
        rs.append(recall)
        f1s.append(f1_score)
        hits_fp[th] = unknown
        hits_tp[th] = no_dupes
        #if th not in hits_tp and th not in hits_fp:
        #    hits_fp[th] = []
        #res.writelines(str(precision) + '\t' + str(recall) + '\n')
        res.writelines(str(precision) + '\t' + str(recall) + '\t' +str(th) + '\t' +str(f1_score) + '\n')
    return ps, rs, f1s, hits_tp, hits_fp

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

# plt.ion()
fig, ax = plt.subplots()
trans_offset = mtransforms.offset_copy(ax.transData, fig=fig,
                                       x=0.05, y=0.10, units='inches')

markers_on = [21, 101,201]
markers = ['o', 'v']
heads_res = []
for j in range(8) : #number of multihead
    file_name = "Interactions_SATORI/interactions_summary_" + str(j) + "_attnLimit-0.12.txt"
    ps, rs, f1s, tp, fp  = evaluate_interactions(os.path.join(dir_path, file_name))
    # intrs = open(os.path.join(dir_path, "unique_interactions_" + str(j) + "0.05.txt"), "w")
    # for m in tp[0.05]:
    #     intrs.writelines(m[0] + '\t' + m[1] + '\n')
    # for m in fp[0.05]:
    #     intrs.writelines(m[0] + '\t' + m[1] + '\n')
    heads_res.append([tp,fp])
    plt.plot(rs, ps, linestyle='--', marker='o', markevery=markers_on, label=str(j))
    t = []
    s = []
    ths = [0.01, 0.05, 0.1]
    for m in markers_on:
        t.append(rs[m])
        s.append(ps[m])

    i = 0
    for a, b in zip(t, s):
        plt.text(a, b, str(ths[i]), transform=trans_offset, fontsize=8)
        i = i + 1
final_ps, final_rs , final_f1s = evaluate_union_interactions(heads_res)

combined_res = open(os.path.join(dir_path, "final_result_union_heads.txt"), "w")
for it in range(len(thresholds)):
    combined_res.writelines(str(final_ps[it]) + '\t' + str(final_rs[it]) + '\t' + str(thresholds[it]) + '\t' + str(final_f1s[it])+'\n')
plt.plot(final_rs, final_ps, marker='v', markevery = markers_on ,label='Union')
t =[]
s = []
for m in markers_on[:-1]:
    t.append(final_rs[m])
    s.append(final_ps[m])

i = 0
for a,b in zip(t, s):
    plt.text(a, b, str(ths[i]), transform=trans_offset, fontsize=8) #, color='C1'
    i = i + 1

# # axis labels
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

plt.savefig('Plots/CNN_RELATTN_MHUnion.png')


# plt.plot(rs, ps)
# plt.ylabel('precision')
# plt.xlabel('recall')
#
# plt.show()










