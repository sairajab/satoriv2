import os
from satori.evaluate_motif_interaction import *
import numpy as np
import subprocess
import pandas
import glob as glob
import os
from satori.experiment import run_experiment
import shutil
from satori.utils import Config , load_config, get_params_dict, setup_seed
import torch
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
                
def run():
    
    rep = 3
    data_id = "40"
    dataset_path = f"data/Simulated_Data/Data-{data_id}/" 
    d_files =  [f"ctf_{data_id}pairs_eq0", f"ctf_{data_id}pairs_eq1", f"ctf_{data_id}pairs_eq2"]
    pairs_n_meme = {"ctf_40pairs_eq0" : ["tf_pairs_40.txt", "subset40.meme"] , 
                    "ctf_40pairs_eq1" : ["tf_pairs_40.txt", "subset40.meme"] , 
                    "ctf_40pairs_eq2" : ["tf_pairs_40.txt", "subset40.meme"] , 
                    "ctf_60pairs_eq0" : ["tf_pairs_60.txt", "subset60.meme"] ,
                    "ctf_60pairs_eq1" :  ["tf_pairs_60.txt", "subset60.meme"],
                    "ctf_60pairs_eq1" :  ["tf_pairs_60.txt", "subset60.meme"],
                    "ctf_80pairs_eq0": ["tf_pairs_80.txt", "subset80.meme"],
                    "ctf_80pairs_eq1" : ["tf_pairs_80.txt", "subset80.meme"],
                    "ctf_80pairs_eq2" : ["tf_pairs_80.txt", "subset80.meme"]}
    
    
    
    
    arg_space = Config.from_json("modelsparam/train_config.json")

   
    hyperparams_dirs = ["modelsparam/all_exps/simulated/basic/*.txt"]
    
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    for d in d_files:
        outdir = f"results/Data-{data_id}/"  + d + "/"  
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        for hyperparams_dir in hyperparams_dirs:
            modelparams = glob.glob(hyperparams_dir + "*")

            for param in modelparams:
                arg_space.hparamfile = param
                params_dict = get_params_dict(arg_space.hparamfile)
                best_val_auc = -1
                best_exp = ""
                best_seed = -1
                for seed in range(rep):
                    
                    arg_space.inputprefix = os.path.join(dataset_path, d)
                    arg_space.directory = os.path.join(outdir, d) + "/run_" + str(seed)
                    arg_space.pairs_file = os.path.join(dataset_path, pairs_n_meme[d][0])
                    arg_space.tfDatabase = os.path.join(dataset_path, pairs_n_meme[d][1])
                    # Set the random seed for reproducibility
                    setup_seed(seed)
                    # Run the SATORI model with the current parameters
                    experiment_blob = run_experiment(device, arg_space, params_dict)
                
                    
                
                
                    
                
if __name__ == "__main__":

    run()

    #run("satori")