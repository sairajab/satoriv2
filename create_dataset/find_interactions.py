from gettfsfromfile import *
import numpy as np
from Bio import motifs


def read_info(file, tfs_pairs):
    f_lines = open(file, "r").readlines()
    intersection = []
    motifs_count = {}
    
    for i in f_lines[1:80000]:
        motifs = []
        intersected_list = []
        line = i.split('\t')[-1].rstrip()[2:].split("[")
        motifs.append(line[0])
        for g in line:
            
            if ":" in g:
                motifs.append(g.split(":")[-1])
                
        for m in motifs:
            if m in motifs_count:
                motifs_count[m] += 1
            else:
                motifs_count[m] = 1
            
        
        res = [[a, b] for idx, a in enumerate(motifs) for b in motifs[idx + 1:]]
        res_2 = [[b, a] for idx, a in enumerate(motifs) for b in motifs[idx + 1:]]
        
        pairs = res + res_2
        #print(pairs)
        
        for element in tfs_pairs:

            if element in pairs:

                intersected_list.append(element)

        intersection.append(len(intersected_list))
        
    return intersection, motifs_count
        

def get_sequences_with_motif(meme_path, out_path, sequences, headers):

    freq = {"A" :  0.25, "C" : 0.25 , "G" : 0.25, "T" : 0.25}


    w = open(out_path, "w")
    m = motifs.read(open(meme_path),"pfm")
    pwm = m.counts.normalize(pseudocounts=0.5)
    pssm = pwm.log_odds(background=freq)
    print(m.consensus)
    print(pssm.max)
    print(pssm)
    thresh = (pssm.max / 2) + 1
    hits = 0
    for i in range(len(sequences)):
        print(headers[i])
        found_positions = []
        scores = []
        if sequences[i]:
            if len(sequences[i]) >= len(m.consensus):
                for position, score in pssm.search(sequences[i], threshold=thresh):
                    found_positions.append(position)
                    scores.append(score)
                    print("Position %d: score = %5.3f" % (position, score))

                if len(scores) > 0:
                    w.write(headers[i] + "\t")
                for j in range(len(scores)):
                    w.write(str(found_positions[j]) + "\t" + str(scores[j]) + "\t")
                if len(scores) > 0:
                    w.write("\n")
                    hits += 1
            
    w.close()
    return hits   
        
def non_interacting_pairs(tfs_pairs, tfs):
	
	ntfs = []
	res = [[a, b] for idx, a in enumerate(tfs) for b in tfs[idx + 1:]]
	for p in res:
		if p not in tfs_pairs and [p[1], p[0]] not in tfs_pairs:
			ntfs.append(p)
	return ntfs
			
			
        	                
          
        


if __name__ == "__main__":
    tfs_pairs, tfs = get_tf_pairs("tf_pairs_60.txt")
    res = read_info('../data/ToyData/NEWDATA/ctf_60pairs_info.txt', tfs_pairs)
    print(np.sum(res[0]))
    print(res[1])
    print(len(tfs_pairs))
    print(len(tfs))
    # res = non_interacting_pairs(tfs_pairs, tfs)
    #print(res)
    # print(len(res))
    
    
