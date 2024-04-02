import numpy


def flatten(input):
    new_list = []
    for i in input:
        for j in i:
            new_list.append(j)
    return new_list

def get_tf_pairs(tf_pairs_file):
    f = open(tf_pairs_file, "r")
    uniq_tfs = []
    pairs = []
    for x in f:
        tf1, tf2 = x.split('\t')
        pairs.append([tf1, tf2.strip()])
        if tf1 not in uniq_tfs:
            uniq_tfs.append(tf1)
        if tf2.strip() not in uniq_tfs:
            uniq_tfs.append(tf2.strip())
            
    #uniq_tfs = list(set(flatten(pairs)))
    return pairs, uniq_tfs



def get_proteins(meme_db_file):
    ''' Hash motif_id's to protein names using the MEME DB file '''
    motif_protein = {}
    tfs = []
    #temp = []
    for line in open(meme_db_file):
        a = line.split()
        if len(a) > 0 and a[0] == 'MOTIF':
            if a[2][0] == '(':
                motif_protein[a[1]] = a[2][1:a[2].find(')')]
                last_protien = motif_protein[a[1]]
                temp = []
            else:
                motif_protein[a[1]] = a[2]
                last_protien = motif_protein[a[1]]
                temp = []
            tfs.append(last_protien)
    return motif_protein, tfs
