import torch
import numpy as np

###load_cnn_motifs
def flatten(input):
    new_list = []
    for i in input:
        for j in i:
            new_list.append(j)
    return new_list


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
                    pwm[last_protien] = np.array(temp)
                motif_protein[a[1]] = a[2][1:a[2].find(')')]
                last_protien = motif_protein[a[1]]
                temp = []
            else:
                if len(temp) > 0:
                    pwm[last_protien] = np.array(temp)
                motif_protein[a[1]] = a[2]
                last_protien = motif_protein[a[1]]
                temp = []
        if len(a) ==4:
            temp.append([float(a[0]),float(a[1]),float(a[2]),float(a[3])])

    pwm[last_protien] = np.array(temp)
    #pwm[last_protien] = log_odds(np.array(temp))

    return motif_protein, pwm

def get_all_motifs_from_data(tf_pairs_file, pwm):

    total_unique = 200 #for 20 filters

    f = open(tf_pairs_file, "r")
    pairs = []
    for x in f:
        tf1, tf2 = x.split('\t')
        pairs.append([tf1, tf2.strip()])
    uniq_tfs = list(set(flatten(pairs)))

    tfs = pwm.keys()

    for m in tfs:
        if m not in uniq_tfs: #tfs
            if len(uniq_tfs) < total_unique:
                uniq_tfs.append(m)
            else:
                break

    used_pwm = {}

    max_len = 0

    for u in uniq_tfs:

        used_pwm[u] = pwm[u]

        if max_len < pwm[u].shape[0]:

            #print(pwm[u].shape)

            max_len = pwm[u].shape[0]
        

    return used_pwm , max_len

def load_cnn_weights(output_dir ,tfs_pairs_path = "/s/jawar/i/nobackup/Saira/latest/satori_v2/data/ToyData/Ctfs/tf_pairs_data_1_shortest.txt", meme_path = "/s/jawar/i/nobackup/Saira/motif_databases/Jaspar.meme" ):
    
    _, pwm = get_motif_proteins(meme_path)

    motifs_weights, _ = get_all_motifs_from_data(tfs_pairs_path, pwm)

    outfile_path = output_dir + "/filters_to_motif.txt"
    outfile = open(outfile_path, "w")

    ws = []
    i = 0
    filter_size = 21
    


    for motif in motifs_weights:
        
        rand_p = np.random.randint(0, filter_size - len(motifs_weights[motif]) + 1)
        print(rand_p,len(motifs_weights[motif]) )


        outfile.write(str(i) + "\t" + motif + "\n")
        i = i + 1
        temp = np.zeros((4, filter_size))
        temp[:, rand_p:rand_p + len(motifs_weights[motif])] = motifs_weights[motif].T - 0.25
        ws.append(temp)

    ws_np = np.array(ws)

    return torch.Tensor(ws_np)