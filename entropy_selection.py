
from create_dataset.generate_data import SimulatedDataGenerator
from randomized_model_selection import Calibration, setup_seed
from satori.experiment_2 import run_experiment
import torch
import os
from satori.utils import Config , load_config, get_params_dict
import numpy as np
from toydata_inference_pipeline import run_interactions_inference, get_valid_auc_from_file
def run_training_pipeline():
    
    
    data_dir = "data/LongSimulated_Data/Data-80/LongSeqs/" 
    result_dir = "results/Data-80/LongSeqs/Clustered/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    ## Run Model Selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = os.path.join(result_dir, "entropy_selection")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
          
    ## Loop through three instances of baseline model training using best model
    arg_space = Config.from_json("results/Data-80/LongSeqs/Clustered/model_selection/best_overall_params.json")
    params_dict = get_params_dict(arg_space.hparamfile)
    #params_dict = load_config(config_path)
    print(params_dict)
    runs = 3
    seeds = [0, 1, 2] 
    best_val_auc = 0
    data_instances = 3
    entropy_vals = np.logspace(-2, -1, 2)
    ## Run for PSSM 
    for ent in entropy_vals:
        dataname = f"data_80_0"
        arg_space.inputprefix = os.path.join(data_dir,dataname)
        
        for j in range(runs):
            print("Running Entropy Loss for seed", seeds[j])
            arg_space.mode = "test"
            
            if arg_space.mode == 'test':
                
                arg_space.directory = out_dir + "/baseline_ent_" + str(ent) + "/" + dataname + "/" + "run_" + str(j+5)
                val_auc = get_valid_auc_from_file(arg_space.directory + "/modelRes_Val_results.txt")
                if best_val_auc < val_auc:
                    best_val_auc = val_auc
                    best_exp = arg_space.directory
                    best_seed = j
                
            else:
                setup_seed(seeds[j])
                arg_space.seed = seeds[j]
                params_dict['dropout_rate'] = 0.1
                params_dict['CNN_poolsize'] = 8
                #params_dict["optimizer"] = "sgd"
                params_dict["entropy_loss"] = True
                params_dict["entropy_reg_value"] = ent
                #params_dict["sei"] = True
                #params_dict["multiple_linear"]= False
                arg_space.directory = out_dir + "/baseline_ent_" + str(ent) + "/" + dataname + "/" + "run_" + str(j)
                experiment_blob = run_experiment(device, arg_space, params_dict)
                
        if best_val_auc > 0:
            print("Best run ", best_exp , best_val_auc)
            setup_seed(best_seed)
            arg_space.seed = best_seed
            arg_space.mode = "test"
            params_dict['dropout_rate'] = 0.1
            params_dict['CNN_poolsize'] = 8
            #params_dict["optimizer"] = "sgd"
            params_dict["entropy_loss"] = True
            params_dict["entropy_reg_value"] = ent
            #params_dict["sei"] = True
            #params_dict["multiple_linear"]= False
            arg_space.motifAnalysis = False
            arg_space.featInteractions = True
            arg_space.interactionsAnalysis = True
            arg_space.methodType = 'SATORI'
            arg_space.intBackground = 'negative'
            arg_space.directory = best_exp
            experiment_blob = run_experiment(device, arg_space, params_dict)
            run_interactions_inference(experiment_blob, arg_space,params_dict,device)
            
        best_val_auc = 0
    
    print("DONE!!!")
        
        

            

if __name__ == "__main__":
    run_training_pipeline()