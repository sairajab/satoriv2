from readpwms import *
import numpy as np


if __name__ == "__main__":
    
    tfdatabase = '../../../motif_databases/Jaspar.meme'
    motfs , pwms = get_motif_proteins(tfdatabase)
    pairs_file = "/s/jawar/i/nobackup/Saira/latest/satori_v2/data/ToyData/Ctfs/tf_pairs_data_1.txt"
    tfs_pairs, tfs = get_tf_pairs("tf_pairs_60.txt")
    
    for tf in pwms:  

        pfm = pwms[tf] * 10000
        pfm = pfm.T

        f = open("pfm/" + tf +".pfm", "w")

        print(pfm.shape)
        for i in range(pfm.shape[0]):
            for j in range(pfm.shape[1]):
                f.writelines(str(pfm[i,j]) + "\t")
            f.writelines("\n")
