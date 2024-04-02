import numpy
from random import choice, randint
from Bio import motifs
from Bio.Seq import Seq

DEFAULT_LETTER_TO_INDEX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
indexToLetter = dict((DEFAULT_LETTER_TO_INDEX[x], x) for x in DEFAULT_LETTER_TO_INDEX)
def flatten(input):
    new_list = []
    for i in input:
        for j in i:
            new_list.append(j)
    return new_list

#TFs in human promoters dataset

# req_tfs = ['EGR', 'LCOR', 'E2F1', 'DNMT1', 'SRF', 'SP2', 'E2F4','TLX2', 'HHEX']
# tfs_pairs = [['TLX2', 'SOX1'] , ['TLX2', 'HHEX'], ['TLX2' , 'DNMT1'],['TLX2','KDM2B'],['ZKSCAN1','TLX2'],
#             ['TLX2','EGR1'],['ZKSCAN1','SOX1'],['TLX2', 'MLL'],['TLX2','SP2'], ['KDM2B','SOX1'],['TLX2','ZNF263'],
#             ['SOX1','HHEX'],['DNMT1','SOX1'],['TLX2','LHX5'],['SOX1','SP2'],['E2F1', 'TLX2'],
#             ['TLX2', 'ZNF202'], ['SOX1', 'ZNF263'], ['EGR1', 'SOX1'],['ZNF202','SOX1']]
# tfs_pairs = [['TLX2', 'SOX1'] , ['TLX2', 'HHEX'], ['TLX2' , 'DNMT1'],['TLX2','KDM2B'],['ZKSCAN1','TLX2'],
#              ['TLX2','EGR1'],['ZKSCAN1','SOX1'],['TLX2', 'MLL'],['TLX2','SP2'], ['KDM2B','SOX1'],['TLX2','ZNF263'],
#              ['SOX1','HHEX'],['DNMT1','SOX1'],['TLX2','LHX5'],['SOX1','SP2'],['E2F1', 'TLX2'],
#              ['TLX2', 'ZNF202'], ['SOX1', 'ZNF263'], ['EGR1', 'SOX1'],['ZNF202','SOX1'], ['EGR1', 'LCOR'],
#              ['E2F1', 'LCOR'], ['DNMT1', 'LCOR'],['EGR1' ,'SRF'], ['SRF' ,'SP2'], ['E2F1','SRF'], ['EGR1','E2F4'],
#              ['E2F1' ,'DNMT1'], ['E2F1' ,'E2F4'], ['DNMT1' ,'E2F4'], ['E2F1' ,'EGR1'], ['DNMT1' ,'EGR1']]
# print(len(tfs_pairs))
# uniq_tfs = list(set(flatten(tfs_pairs)))
# print(len(uniq_tfs))

#***********uncomment for actual motifs**********
from gettfsfromfile import *

def load_data(pairs_file, tfs_count = -1, tf_database = '../../../motif_databases/Jaspar.meme'):
      
    _,motfs = get_proteins(tf_database)
    
    tfs_pairs, tfs = get_tf_pairs(pairs_file)
    if tfs_count == -1:
        total_unique = len(tfs) #30
    else:
         total_unique = tfs_count 
    uniq_tfs = []
    for u in tfs:
        uniq_tfs.append(u)
    for m in motfs:
        if m not in uniq_tfs: #tfs
            if len(uniq_tfs) < total_unique:
                uniq_tfs.append(m)
            else:
                break
    return tfs_pairs, uniq_tfs

def get_motif_proteins(meme_db_file):
    ''' Hash motif_id's to protein names using the MEME DB file '''
    motif_protein = {}
    pwm = {}
    temp = []
    for line in open(meme_db_file):
        a = line.split()
        if len(a) > 0 and a[0] == 'MOTIF':
            if a[2][0] == '(':
                if len(temp) > 0:
                    pwm[last_protien] = numpy.array(temp)
                motif_protein[a[1]] = a[2][1:a[2].find(')')]
                last_protien = motif_protein[a[1]]
                temp = []
            else:
                if len(temp) > 0:
                    pwm[last_protien] = numpy.array(temp)
                motif_protein[a[1]] = a[2]
                last_protien = motif_protein[a[1]]
                temp = []
        if len(a) ==4:
            temp.append([float(a[0]),float(a[1]),float(a[2]),float(a[3])])
    pwm[last_protien] = numpy.array(temp)
    return motif_protein, pwm

def get_motif_score(motif, tf):
        print(tf)
    
        freq = {"A" :  0.225, "C" : 0.278 , "G" : 0.271, "T" : 0.226}
        m = motifs.read(open("pfm/"+tf+".pfm"),"pfm")
        pwm = m.counts.normalize(pseudocounts=0.5)
        pssm = pwm.log_odds(background=freq)
        print("Max value" , pssm.max)
        thresh = pssm.max/2 
        passed = False
        for position, score in pssm.search(motif, threshold=thresh):
                #found_positions.append(position)
                if position >= 0:
                    #scores.append(score)
                    print("Position %d: score = %5.3f" % (position, score))
                    passed = True
        return passed
                    
    
def computeMotifGivenMatrix(matrix, tf):
    

        cols, rows = matrix.shape
        motif = ""

        while not get_motif_score(motif , tf):
            motif = ""


            for i in range(cols):
                probs = matrix[i]
                probs /= probs.sum()
                motif_letter = numpy.random.choice(4, p=probs)
                motif += indexToLetter[motif_letter]
                
        return motif

def get_motif_seq_non_interacting_negative(pwms, idx,tfs_pairs, uniq_tfs, last_batch = False, pair=False):
    tf_motif = {}
    req_tfs = []
    print(idx)
    if pair:
        for i in idx:
            req_tfs.append(tfs_pairs[i])
        ##req_tfs = list(flatten(req_tfs)) for close proximity
    else:
        if last_batch:
            for i in idx:
                pair = tfs_pairs[i]
                get_idx = numpy.random.choice(2)
                req_tfs.append(pair[get_idx])
            #req_tfs = list(set(flatten(tfs_pairs)))
        else:
            print(idx)
            c1 = uniq_tfs[idx[0]], uniq_tfs[idx[1]]
            c1_rev = uniq_tfs[idx[1]], uniq_tfs[idx[0]]
            c2 = uniq_tfs[idx[1]], uniq_tfs[idx[2]]
            c2_rev = uniq_tfs[idx[2]], uniq_tfs[idx[1]]
            c3 = uniq_tfs[idx[0]], uniq_tfs[idx[2]]
            c3_rev = uniq_tfs[idx[2]], uniq_tfs[idx[0]]
            c1_flag = False
            c2_flag = False
            c3_flag = False

            if c1 not in tfs_pairs and c1_rev not in tfs_pairs:
                c1_flag = True
            if c2 not in tfs_pairs and c2_rev not in tfs_pairs:
                c2_flag = True
            if c3 not in tfs_pairs and c3_rev not in tfs_pairs:
                c3_flag = True

            print(c1_flag, c2_flag, c3_flag)

            if c1_flag and c2_flag and c3_flag:

                print("all")

                req_tfs.append([c1[0]])
                req_tfs.append([c1[1]])
                req_tfs.append([c2[1]])
            
            if c1_flag and not c2_flag and not c3_flag:

                req_tfs.append([c1[0]])
                req_tfs.append([c1[1]])

            if not c1_flag and c2_flag and not c3_flag:

                req_tfs.append([c2[0]])
                req_tfs.append([c2[1]])

            if not c1_flag and not c2_flag and c3_flag:

                req_tfs.append([c3[0]])
                req_tfs.append([c3[1]])

            if not c1_flag and not c2_flag and not c3_flag:

                i = randint(0, 2)

                req_tfs.append([uniq_tfs[i]])
    
    k = 0
    for l in req_tfs:
        temp = {}
        for tf in l:
          if tf in pwms:
            motif_seq = computeMotifGivenMatrix(pwms[tf], tf)
            print("Motif seq:" + tf + " " + str(len(motif_seq)))
            #if not tf in tf_motif:
            #    tf_motif[tf] = []
            temp[tf] = motif_seq
        tf_motif[k] = temp #.append()
        k = k + 1
    return tf_motif



def get_motif_seq(pwms, idx, tfs_pairs, uniq_tfs, last_batch = False, pair=False):
    tf_motif = {}
    req_tfs = []
    if pair:
        for i in idx:
            req_tfs.append(tfs_pairs[i])
        ##req_tfs = list(flatten(req_tfs)) for close proximity
    else:
        if last_batch:
            for i in idx[:2]:
                pair = tfs_pairs[i]
                get_idx = numpy.random.choice(2)
                req_tfs.append([pair[get_idx]])
            for i in idx[2:]:
                req_tfs.append([uniq_tfs[i]])
            #req_tfs = list(set(flatten(tfs_pairs)))
        else:
            temp_tfs = []
            for i in idx:
                req_tfs.append([uniq_tfs[i]])
            # res = [[a[0], b[0]] for idx, a in enumerate(temp_tfs) for b in temp_tfs[idx + 1:]]
            # res_2 = res + [[b[0], a[0]] for idx, a in enumerate(temp_tfs) for b in temp_tfs[idx + 1:]]
            
            # for p in res_2:
            #     if res_2
            
    k = 0
    for l in req_tfs:
        temp = {}
        for tf in l:
          if tf in pwms:
            motif_seq = computeMotifGivenMatrix(pwms[tf], tf)
            print("Motif seq:" + tf + " " + str(len(motif_seq)))
            #if not tf in tf_motif:
            #    tf_motif[tf] = []
            temp[tf] = motif_seq
        tf_motif[k] = temp #.append()
        k = k + 1
    return tf_motif
