from gettfsfromfile import *
from readpwms import *
import random

def create_pairs_list(tf_database):
        _ , motfs = get_proteins(tf_database)
        #print(pwm)
        return motfs
        
def make_pairs(motifs, outfile="tf_pairs_80.txt"):
    
    res = [[a, b] for idx, a in enumerate(motifs) for b in motifs[idx + 1:]]
    #print(res)
    random.shuffle(res)
    pairs = res[:80]
    print(pairs)
    f = open(outfile, "w")

    uniq_tfs = []
    for x in pairs:
        tf1, tf2 = x[0] , x[1]
        if tf1 not in uniq_tfs:
            uniq_tfs.append(tf1)
        if tf2 not in uniq_tfs:
            uniq_tfs.append(tf2)
        f.writelines(str(tf1) + "\t" + str(tf2) + "\n")
            
    print(len(uniq_tfs))
    f.close()
    
    
    
    
if __name__ == "__main__":
    tfdatabase = '../../../motif_databases/Jaspar.meme'
    motfs_list = create_pairs_list(tfdatabase)
    #print(motfs_list)
    # random.shuffle(motfs_list)
    # random.shuffle(motfs_list)
    
    selected_tfs = motfs_list[:30]
    #print(selected_tfs)
    make_pairs(selected_tfs)
    
    
    

    
    