
import itertools

from gettfsfromfile import *


if __name__ == "__main__" :

    f = open("ctf_2_info.txt", "r")
    # pos_f = open("ctf_positive_gt.txt", "w")
    # neg_f = open("negative.txt", "w")

    tfs_pairs, _ = get_tf_pairs('tf_pairs_data_1.txt')

    pos = []
    neg = []

    gt = {}
    gt_neg = {}

    for i in f.readlines():

        c, _, _ , data = i.split("\t")

        motifs = data.strip().split(":")
        temp = []

        for m in motifs[1:]:



            tf = m.split("[")[0]
            if c == "Pos":
                pos.append(tf)
                temp.append(tf)
            else:
                neg.append(tf)
                temp.append(tf)

        if c == "Pos":

            inter_combs = itertools.combinations(temp, 2)

            for tf_pair in inter_combs:
                if [tf_pair[0],tf_pair[1]] in tfs_pairs or [tf_pair[1],tf_pair[0]] in tfs_pairs:
                    if ((tf_pair[0],tf_pair[1]) not in gt) and ((tf_pair[1],tf_pair[0]) not in gt):
                        gt[tf_pair] = 1
                    else:
                        if tf_pair in gt:
                            gt[tf_pair] += 1
                        else:
                            gt[(tf_pair[1],tf_pair[0])] += 1
        else:

            inter_combs = itertools.combinations(temp, 2)

            for tf_pair in inter_combs:
                if [tf_pair[0],tf_pair[1]] in tfs_pairs or [tf_pair[1],tf_pair[0]] in tfs_pairs:
                    if ((tf_pair[0],tf_pair[1]) not in gt_neg) and ((tf_pair[1],tf_pair[0]) not in gt_neg):
                        gt_neg[tf_pair] = 1
                    else:
                        if tf_pair in gt_neg:
                            gt_neg[tf_pair] += 1
                        else:
                            gt_neg[(tf_pair[1],tf_pair[0])] += 1


    print(len(gt))
    print(len(gt_neg))
    # for tf in gt:
    #     pos_f.writelines(tf[0] + "\t" + tf[1] + "\t" + str(gt[tf])  + "\n")
    
    # for tf in gt_neg:
    #     neg_f.writelines(tf[0] + "\t" + tf[1] + "\t" + str(gt_neg[tf])  +  "\n")


    one = set(pos)
    zero = set(neg)

    print(len(one))
    print(len(zero))
    print(one)
    print(zero)

    all = one | zero

    filters = open("fiters_to_motif_2.txt", "w")

    for i, a in enumerate(all):
            filters.writelines(str(i) +"\t"+ a + "\n" )




