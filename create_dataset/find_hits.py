from Bio import motifs
from Bio.Seq import Seq
from fastaIO import *
import os
from scipy.stats import fisher_exact


def get_sequences_with_motif(meme_path, fasta_file):
    
    headers = []
    sequences = []
    
    fasta_iterator = fasta_itr(fasta_file)

    
    for fasta_record in fasta_iterator:
        if len(fasta_record.sequence) > 0:
            headers.append(fasta_record.header)
            sequences.append(fasta_record.sequence)

    #out_file = motif_file.split('/')[-1].split('.')[0] + "_allDIR.txt"
    out_file = meme_path.split('.')[0] + "_newdata.txt"

    freq = {"A" :  0.225, "C" : 0.278 , "G" : 0.271, "T" : 0.226}

    w = open(out_file, "w")
    m = motifs.read(open(meme_path),"pfm")
    pwm = m.counts.normalize(pseudocounts=0.5)
    pssm = pwm.log_odds(background=freq)
    print(m.consensus)
    print(pssm.max)
    print(pssm)
    thresh = pssm.max / 2 
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


if __name__ == "__main__":
    
    #fasta_file = "../data/ToyData/NEWDATA/ctf_60pairs.fa"
    fasta_file = "test.fa"
    meme_path = "pfm/ZBTB1.pfm"
    hits = get_sequences_with_motif(meme_path, fasta_file)
    
    

