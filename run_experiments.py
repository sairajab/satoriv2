import os
from analyze_motif_interaction import *
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
    
   
                
def run(mode):
    
    rep = 3

    dataset_path = "data/Simulated_Data/Data-40/"
    d_files = ["ctf_40pairs_eq0", "ctf_40pairs_eq1", "ctf_40pairs_eq2"] 
    pairs_n_meme = {"ctf_40pairs_eq2" : ["tf_pairs_40.txt", "subset40.meme"] , 
                    "ctf_40pairs_eq0" : ["tf_pairs_40.txt", "subset40.meme"] , 
                    "ctf_40pairs_eq1" : ["tf_pairs_40.txt", "subset40.meme"] , 
                    "ctf_60pairs_eq" : ["tf_pairs_60.txt", "subset60.meme"] ,
                    "ctf_80pairs_eq" :  ["tf_pairs_80.txt", "subset80.meme"]}
    hyperparams_dirs = ["modelsparam/all_exps/simulated/basic/"]
    for d in d_files:
        outdir = "results/Data-40/" + d + "/"
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        for hyperparams_dir in hyperparams_dirs:
            modelparams = glob.glob(hyperparams_dir + "*")
            for param in modelparams:
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
                
                        satori_path = exp_name+"/Interactions_SATORI/"
                        print(satori_path)
                        
                        if os.path.exists(satori_path):
                            print("*******Removed directory...")
                            remove(satori_path)
                            
                        motif_path = exp_name+"/Motif_Analysis/"
                        if os.path.exists(motif_path):
                            remove(motif_path)
                            print("*******Removed Motifs directory...")
                        # Run Satori
                        cmd = "python satori.py " + os.path.join(dataset_path, d) +  " " + param + " -w 8" + \
                            " --outDir "  + exp_name + " --mode test -v -s --background negative --intseqlimit 5000" + \
                            " --numlabels 2 --motifanalysis --interactions --interactionanalysis --gt_pairs " + os.path.join("create_dataset",pairs_n_meme[d][0]) + \
                            " --method SATORI --attrbatchsize 32 --deskload --set_seed --seed " + str(i) + \
                            " --tomtompath /s/jawar/i/nobackup/Saira/meme/src/tomtom --database " + os.path.join("create_dataset",pairs_n_meme[d][1])
                        print(cmd)
                        subprocess.call(cmd, shell=True)
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
                        
                        #print("Write CODE to do something else..............")
                

if __name__ == "__main__":

    run("train")

    #run("satori")



