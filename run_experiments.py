import os
from evaluate_motif_interaction import *
import numpy as np
import subprocess
import pandas
import glob as glob
import os
import shutil

def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))
    
import os

def get_valid_auc_from_file(file_path):
    """Extract the Valid_AUC from the file if it exists and return it as a float."""
    try:
        with open(file_path, 'r') as f:
            # Skip the header
            f.readline()
            # Read the content (Valid_Loss, Valid_AUC, Valid_AUPRC)
            content = f.readline().strip().split()
            if len(content) >= 2:  # Valid_AUC is the second column
                return float(content[1])  # Return the Valid_AUC
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return None

def find_highest_auc_folder(root_folder):
    """Recursively go through folders, find files, and get the folder with the highest AUC."""
    highest_auc = -1
    best_folder = None
    
    # Walk through the directory tree
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for file in filenames:
            if file.endswith('.txt'):  # Assuming the files are .txt files
                file_path = os.path.join(dirpath, file)
                valid_auc = get_valid_auc_from_file(file_path)
                if valid_auc is not None and valid_auc > highest_auc:
                    highest_auc = valid_auc
                    best_folder = dirpath  # Store the folder path
    
    return best_folder, highest_auc
                
def run(mode):
    
    rep = 3
    print("Running SATORI with mode: ", mode)

    dataset_path = "/s/chromatin/p/nobackup/Saira/original/satori/data/ToyData/NEWDATA/clustered_tfs/" #"data/LongSimulated_Data/Data-80/" #
    d_files =  ["ctf_80pairs_2"]#"ctf_80pairs_0", ,"ctf_80pairs_2" #"seqs_80_2" run deep E3 and deep entropy ##["seqs_80_1" , "seqs_80_2", "seqs_80_3"] 
    pairs_n_meme = {"seqs_1500bp_3" : ["tf_pairs_40.txt", "subset40.meme"] , 
                    "seqs_1500bp_2" : ["tf_pairs_40.txt", "subset40.meme"] , 
                    "seqs_1500bp_1" : ["tf_pairs_40.txt", "subset40.meme"] , 
                    "seqs_80_1" : ["tf_pairs_80.txt", "subset80.meme"] ,
                    "seqs_80_2" :  ["tf_pairs_80.txt", "subset80.meme"],
                    "seqs_80_3" :  ["tf_pairs_80.txt", "subset80.meme"],
                    "ctf_80pairs_0": ["clustered_tf_pairs_80.txt", "subset80clustered.meme"],
                    "ctf_80pairs_1" : ["clustered_tf_pairs_80.txt", "subset80clustered.meme"],
                    "ctf_80pairs_2" : ["clustered_tf_pairs_80.txt", "subset80clustered.meme"]}
    hyperparams_dirs = ["modelsparam/all_exps/simulated/basic/baseline_tiana_pe.txt"]
    loadmotfs = False
    best_val_auc = -1
    best_exp = ""
    best_seed = -1
    find_SATORI_intrs = False
    for d in d_files:
        outdir = "results/Data-80/Clustered/"  + d + "/"  #/"LongSeqs/selected_model/
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        for hyperparams_dir in hyperparams_dirs:
            modelparams = glob.glob(hyperparams_dir + "*")

            for param in modelparams:
                best_val_auc = -1
                best_exp = ""
                best_seed = -1
                if "01" in param:
                    continue

                exp = param.split("/")[-1].split(".t")[0]
                in_dir = outdir + exp + "/"
                
                if not os.path.exists(in_dir):
                    os.mkdir(in_dir)

                for i in range(rep):
                    
                    exp_name = in_dir + "E" + str(i+1)
                    
                    if mode == "train":

        
                        try:
                            # Run Satori
                            cmd = "python satori.py " + os.path.join(dataset_path, d) +  " " + param + " -w 8" + \
                            " --outDir "  + exp_name + " --mode train -v -s --background negative --intseqlimit 5000" + \
                            " --numlabels 2 --gt_pairs " + os.path.join("create_dataset",pairs_n_meme[d][0]) + \
                            " --method SATORI --set_seed --attrbatchsize 32 --deskload --seed " + str(i) + \
                            " --tomtompath /s/chromatin/p/nobackup/Saira/meme/src/tomtom --database " + os.path.join("create_dataset",pairs_n_meme[d][1])
                            print(cmd)
                            subprocess.call(cmd, shell=True)

                        except subprocess.CalledProcessError as e:
                            # Handle any errors that occur during command execution
                            print(f"Error executing command: {cmd}")
                            print(f"Return code: {e.returncode}")
                            print(f"Standard Error of '{cmd}':\n{e.stderr}")
                    
                            
                    elif mode == "satori":

                        val_auc = get_valid_auc_from_file(exp_name + "/modelRes_Val_results.txt")
                        if best_val_auc < val_auc:
                            best_val_auc = val_auc
                            best_exp = exp_name
                            best_seed = i
                        find_SATORI_intrs = True

                    elif mode == "fis":
                        
                        fis_path = exp_name+"/Interactions_FIS/"
                        print(fis_path)
                        
                        if os.path.exists(fis_path):
                            print("*******Removed directory...")
                            remove(fis_path)

                        # Run FIS
                        cmd = "python satori.py " + os.path.join(dataset_path, d) +  " " + param + " -w 8" + \
                            " --outDir "  + exp_name + " --mode test -v -s --background negative --intseqlimit 5000" + \
                            " --numlabels 2 --interactions --interactionanalysis --gt_pairs " + os.path.join("create_dataset",pairs_n_meme[d][0]) + \
                            " --method FIS --attrbatchsize 32 --deskload --set_seed --seed " + str(i) + \
                            " --tomtompath /s/jawar/i/nobackup/Saira/meme/src/tomtom --database " + os.path.join("create_dataset",pairs_n_meme[d][1])
                        print(cmd)
                        subprocess.call(cmd, shell=True)
                    else:
                        
                        cmd = "python satori.py " + os.path.join(dataset_path, d) +  " " + param + " -w 8" + \
                            " --outDir "  + exp_name + " --mode test -v --deskload --seed " + str(i) 
                        print(cmd)
                        subprocess.call(cmd, shell=True)

                if find_SATORI_intrs:
                                    
                        satori_path = best_exp+"/Interactions_SATORI/"
                        
                        if os.path.exists(satori_path):
                            print("*******Removed directory...", satori_path)
                            remove(satori_path)
                            
                        motif_path = best_exp+"/Motif_Analysis/"
                        if os.path.exists(motif_path):
                            remove(motif_path)
                            print("*******Removed Motifs directory...")
                            
                        motif_path_neg = best_exp+"/Motif_Analysis_Negative/"
                        if os.path.exists(motif_path_neg):
                            remove(motif_path_neg)
                            print("*******Removed Motifs Neg directory...")
                        # Run Satori
                        cmd = "python satori.py " + os.path.join(dataset_path, d) +  " " + param + " -w 8" + \
                            " --outDir "  + best_exp + " --mode test -v -s --background negative --intseqlimit 5000" + \
                            " --numlabels 2 --motifanalysis --interactions --interactionanalysis --gt_pairs " + os.path.join("create_dataset",pairs_n_meme[d][0]) + \
                            " --method SATORI --attrbatchsize 32 --deskload --set_seed --seed " + str(best_seed) + \
                            " --tomtompath /s/chromatin/p/nobackup/Saira/meme/src/tomtom --database " + os.path.join("create_dataset",pairs_n_meme[d][1])
                        print(cmd)
                        subprocess.call(cmd, shell=True)
                        attnattr_path = best_exp + "/Interactions_ATTNATTR/"
                        
                        if os.path.exists(attnattr_path):
                            print("*******Removed directory...", attnattr_path )
                            remove(attnattr_path)
                                                # Run ATTNATTR
                        # cmd = "python satori.py " + os.path.join(dataset_path, d) +  " " + param + " -w 8" + \
                        #     " --outDir "  + best_exp + " --mode test -v -s --background negative --intseqlimit 5000" + \
                        #     " --numlabels 2 --interactions --interactionanalysis --gt_pairs " + os.path.join("create_dataset",pairs_n_meme[d][0]) + \
                        #     " --method ATTNATTR --attrbatchsize 32 --deskload --set_seed --seed " + str(best_seed) + \
                        #     " --database " + os.path.join("create_dataset",pairs_n_meme[d][1])
                        # print(cmd)
                        # subprocess.call(cmd, shell=True)
                        
                        #print("Write CODE to do something else..............")
                

if __name__ == "__main__":

    run("train")

    #run("satori")