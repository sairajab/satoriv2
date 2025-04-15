from gettfsfromfile import *
import numpy as np
from Bio import motifs

def get_motif_proteins(meme_db_file, tfs):
    ''' Hash motif_id's to protein names using the MEME DB file '''
    file = open("subset80.meme", "w")
    motif_protein = {}
    pwm = {}
    temp = []
    c = 0
    motif_started = False
    motif_lines = []
    selected_motifs = []
    for line in open(meme_db_file):
        a = line.split()
        #print(line)

        line = line.strip()
            
        # Detect the start of a motif block
        if line.startswith("MOTIF "):
                print(line)
                motif_started = True
                motif_lines = []
                motif_lines.append(line)
        elif motif_started:
            # Collect lines within the motif block
            motif_lines.append(line)
                
            # Detect the end of a motif block
            if line.startswith("URL "):
                #print("Motif ended")
                motif_started = False
                motif_data = "\n".join(motif_lines)
                #print(motif_data)
                    
                # Parse the motif data to extract the motif name
                motif_name = None
                for motif_line in motif_lines:
                        if motif_line.startswith("MOTIF "):
                            a = motif_line.split()
                            print(a)
                            if "(" in a[2]:
                                motif_name = a[2][1:a[2].find(')')]
                            else:
                                motif_name = a[2]
                            #[-1]
                            #print("motif name ",motif_name )
                            break
                    
                # Check if the motif name is in the desired list
                if motif_name in tfs:
                    #print(motif_data)
                    selected_motifs.append(motif_data)
        
        # if len(a) > 0 and a[0] == 'MOTIF' and c < len(tfs):
        #     if a[2][1:a[2].find(')')] in tfs:
        #         file.writelines(line)
        #         c = c + 1     
        # else:
        #     #if c == 20: 
        #     file.writelines(line)
            
        # if len(a) ==4:
        #     temp.append([float(a[0]),float(a[1]),float(a[2]),float(a[3])])
    #pwm[last_protien] = numpy.array(temp)
    #return #motif_protein, pwm
    for ln in selected_motifs:
        file.writelines(ln)
        file.writelines("\n\n")

from Bio import motifs

def create_subset_meme(meme_file_path, desired_motif_names):

    # Parse the MEME database
    with open(meme_file_path) as handle:
        
        meme_db = motifs.parse(handle, "MEME")

        # Create an empty list to store the selected motifs
        selected_motifs = []

        # Iterate through motifs in the database and check if their name is in the desired list
        for motif in meme_db:
            if motif.name in desired_motif_names:
                selected_motifs.append(motif)

        # Create a new MEME database with the selected motifs
        subset_meme_db = motifs.MotifDB(selected_motifs)

        # Save the subset MEME database to a new file (optional)
        subset_meme_file_path = "clustered_subset80.meme"
        with open(subset_meme_file_path, "w") as handle:
            subset_meme_db.write(handle)

if __name__ == "__main__":

    tfdatabase = '/s/chromatin/p/nobackup/Saira/motif_databases/Jaspar.meme' #JASPAR2024_CORE_non-redundant_pfms
    tfs_pairs, tfs = get_tf_pairs("/s/chromatin/p/nobackup/Saira/original/satori/create_dataset/tf_pairs_80.txt")
    print(tfs, len(tfs))
    get_motif_proteins(tfdatabase, tfs)