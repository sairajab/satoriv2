
from create_dataset.generate_data import SimulatedDataGenerator
from randomized_model_selection import Calibration
from satori.experiment import run_experiment
import torch
import os
from satori.utils import Config , load_config, get_params_dict, setup_seed
from satori.interaction_analysis import run_interactions_inference

def run_training_pipeline():
    
    ## Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dirs = {"arabidopsis" : "data/Arabidopsis_ChromAccessibility/atAll_m200_s600" , "human_promoters" : "data/Human_Promoters/encode_roadmap_inPromoter" }
    config_files = {"arabidopsis" : "modelsparam/train_arab_config.json" , "human_promoters" : "modelsparam/train_hp_config.json" }
    result_dirs = {"arabidopsis" : "results/Arabidopsis/" , "human_promoters" : "results/HP/" }
    runs = 3
    seeds = [42, 1337, 2024] 
    exps = ["pssm_deep_ent"] #"pssm_pe" run pssm for human promoters and attnattr for arabidopsis

    for exp in exps:
        for data_name in list(data_dirs.keys())[1:]: 

            data_dir = data_dirs[data_name]
            result_dir = result_dirs[data_name]

            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            arg_space = Config.from_json(config_files[data_name])
            arg_space.mode = "train"
            arg_space.inputprefix = data_dir

            params_dict = get_params_dict(arg_space.hparamfile)
            print(params_dict)
            best_run = None
            best_loss = float("inf")

            ### Run with PSSM
            for i in range(runs):
                ## Set Seed
                setup_seed(seeds[i])
                arg_space.seed = seeds[i]
                params_dict["load_motif_weights"]= True
                params_dict["use_posEnc"] = False
                print("Running with seed ", seeds[i], exp)
                if "pssm_deep_ent" == exp:
                    
                    params_dict["entropy_reg_value"] = 0.001
                    params_dict["entropy_loss"] = True
                    params_dict["sei"]= True
                    params_dict["multiple_linear"] = False
                
                if exp == "pssm_pe":
                    params_dict["use_posEnc"] = True
                    
                arg_space.directory = os.path.join(result_dir, f"deep_{data_name}", exp, f"run_{i}")
                experiment_blob = run_experiment(device, arg_space, params_dict)
                loss = experiment_blob["res_valid"][0]
                print("Loss: ", loss)
                if loss < best_loss:
                    best_loss = loss
                    best_run = experiment_blob
                    best_run["seed"] = seeds[i]
                    best_run["directory"] = arg_space.directory
                    best_run["params_dict"] = params_dict

            arg_space.mode = "test"
            arg_space.directory = best_run["directory"]
            arg_space.motifAnalysis = True
            arg_space.featInteractions = True
            arg_space.methodType = 'RAWnATTR'
            arg_space.intBackground = 'shuffle'
            params_dict = best_run["params_dict"]

            setup_seed(best_run["seed"])

            run_interactions_inference(best_run, arg_space,params_dict,device)

            

if __name__ == "__main__":
    run_training_pipeline()