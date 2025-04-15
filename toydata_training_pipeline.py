
from create_dataset.generate_data import SimulatedDataGenerator
from randomized_model_selection import Calibration
from satori.experiment import run_experiment
import torch
import os
from satori.utils import Config , load_config, get_params_dict, setup_seed

def run_training_pipeline():
    
    ## Generate Data
    path_to_meme = '/s/chromatin/p/nobackup/Saira/motif_databases/JASPAR2024_CORE_non-redundant_pfms.meme'
    pairs_file = "create_dataset/clustered_tf_pairs_80.txt"
    num_of_seq = 30000
    seq_len = 1500
    data_dir = "data/LongSimulated_Data/Data-80/LongSeqs/" 
    result_dir = "results/Data-80/LongSeqs/Clustered/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data_name = os.path.join(data_dir,"data_80_0")
    #embedder = SimulatedDataGenerator(seq_len=seq_len, path_to_meme=path_to_meme, pairs_file=pairs_file, num_of_seq=num_of_seq, output_name=data_name)
    #embedder.generate_data()

    ## Run Model Selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    param_dist = {
    "lr": [1e-4, 1e-3, 1e-2],
    "batch_size": [16, 32, 64],
    "linear_layer_size": [256, 512, 1024],
    "dropout_rate": [0.1, 0.2, 0.3],
    "multiple_linear" : [True, False],
    "weight_decay": [0, 1e-4, 1e-3],
    "num_multiheads": [2, 4, 8],
    "singlehead_size": [32,64, 128],
    "CNN_filters": [[256], [380], [512]],  # Number of Conv1D filters
    "CNN_filtersize": [[13], [16], [24]],  # Kernel size for Conv1D
    "momentum": [0.8, 0.9, 0.99],  # For SGD
    "optimizer": ["adam", "sgd"]}
    out_dir = os.path.join(result_dir, "model_selection")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    #config_path = Calibration(dataset_path=data_name, numLabels=2, seq_length=seq_len, device=device, search_space_dict=param_dist, out_dir=out_dir,  n_iter = 30)
    
    #print("Model Selection Complete ", config_path)
    
    ## Loop through three instance of data generation
    data_instances = 3
    for i in range(data_instances):
        ## Set Seed
        #setup_seed(42 + i)
        ## Generate Data
        data_name = os.path.join(data_dir,f"data_80_{i+1}")
        #embedder = SimulatedDataGenerator(seq_len=seq_len, path_to_meme=path_to_meme, pairs_file=pairs_file, num_of_seq=num_of_seq, output_name=data_name)
        #embedder.generate_data()
        
    print("Data Generation Complete")
    ## Loop through three instances of baseline model training using best model
    arg_space = Config.from_json("results/Data-80/LongSeqs/Clustered/model_selection/best_overall_params.json")
    params_dict = get_params_dict(arg_space.hparamfile)
    #params_dict = load_config(config_path)
    print(params_dict)
    runs = 1
    seeds = [42, 1337, 2024] 
    
    # for i in range( data_instances):
    #     dataname = f"data_80_{i+1}"
    #     arg_space.inputprefix = os.path.join(data_dir,dataname)
        
    #     for j in range(runs):
    #         setup_seed(seeds[j])
    #         arg_space.directory = result_dir + "/baseline/" + dataname + "/" + "run_" + str(j)
    #         experiment_blob = run_experiment(device, arg_space, params_dict)
            
    ## Loop through three instances of baseline model training with Entropy Loss
    for i in range(data_instances):
        dataname = f"data_80_{i+1}"
        arg_space.inputprefix = os.path.join(data_dir,dataname)
        
        for j in range(runs):
            print("Running Entropy Loss for seed", seeds[j])
            setup_seed(seeds[j])
            arg_space.seed = seeds[j]
            params_dict["entropy_loss"] = True
            #params_dict["optimizer"]   = "sgd"
            params_dict["entropy_reg_value"] = 0.001
            arg_space.directory = result_dir + "/baseline_" + str(params_dict["entropy_reg_value"]) +  "/" + dataname + "/" + "run_" + str(j)
            experiment_blob = run_experiment(device, arg_space, params_dict)

    
    ## Loop through three instances of baseline model training with PE
    # for i in range(data_instances):
    #     dataname = f"data_80_{i+1}"
    #     arg_space.inputprefix = os.path.join(data_dir,dataname)
        
    #     for j in range(runs):
    #         setup_seed(seeds[j])
    #         params_dict["use_posEnc"]= True
    #         arg_space.directory = result_dir + "/baseline_pe/" + dataname + "/" + "run_" + str(j)
    #         experiment_blob = run_experiment(device, arg_space, params_dict)
    
        
    ## Run for PSSM 
    for i in range(data_instances):
        dataname = f"data_80_{i+1}"
        arg_space.inputprefix = os.path.join(data_dir,dataname)
        
        #for j in range(runs):
            # setup_seed(seeds[j])
            # print("Running Entropy Loss for seed", seeds[j])
            # params_dict["load_motif_weights"]= True
            # params_dict["CNN_filtersize"] = [24]
            # params_dict["optimizer"] = "sgd"
            # arg_space.directory = result_dir + "/baseline_pssm/" + dataname + "/" + "run_" + str(j)
            # experiment_blob = run_experiment(device, arg_space, params_dict)
            
    ## Run for PSSM and PE
    for i in range(data_instances):
        dataname = f"data_80_{i+1}"
        arg_space.inputprefix = os.path.join(data_dir,dataname)
        
        # for j in range(runs):
        #     setup_seed(seeds[j])
        #     print("Running for seed", seeds[j])
        #     params_dict["load_motif_weights"]= True
        #     params_dict["CNN_filtersize"] = [24]
        #     params_dict["optimizer"] = "sgd"
        #     params_dict["use_posEnc"]= True
        #     params_dict["multiple_linear"]= False
        #     arg_space.directory = result_dir + "/baseline_pssm_pe/" + dataname + "/" + "run_" + str(j)
        #     experiment_blob = run_experiment(device, arg_space, params_dict)

    ## Loop through three instances of baseline model training with PE and Entropy Loss
    for i in range(data_instances):
        dataname = f"data_80_{i+1}"
        arg_space.inputprefix = os.path.join(data_dir,dataname)
        
        # for j in range(runs):
        #     setup_seed(seeds[j])
        #     params_dict["use_posEnc"]= True
        #     params_dict["entropy_loss"] = True
        #     params_dict["optimizer"]   = "sgd"
        #     params_dict["entropy_reg_value"] = 0.01
        #     arg_space.directory = result_dir + "/baseline_pe_entropy/" + dataname + "/" + "run_" + str(j)
        #     experiment_blob = run_experiment(device, arg_space, params_dict)
    
    ## Run for PSSM + pe + entropy
    for i in range(data_instances):
        dataname = f"data_80_{i+1}"
        arg_space.inputprefix = os.path.join(data_dir,dataname)
        
        # for j in range(runs):
        #     setup_seed(seeds[j])
        #     params_dict["use_posEnc"]= True
        #     params_dict["entropy_loss"] = True
        #     params_dict["optimizer"]   = "sgd"
        #     params_dict["entropy_reg_value"] = 0.01
        #     params_dict["load_motif_weights"]= True
        #     params_dict["CNN_filtersize"] = [24]
        #     params_dict["multiple_linear"]= False
        #     arg_space.directory = result_dir + "/baseline_pssm_pe_entropy/" + dataname + "/" + "run_" + str(j)
        #     experiment_blob = run_experiment(device, arg_space, params_dict)

    
    ## Run for deeper models 
    for i in range(data_instances):
        dataname = f"data_80_{i+1}"
        arg_space.inputprefix = os.path.join(data_dir,dataname)
        
        # for j in range(runs):
        #     setup_seed(seeds[j])
        #     params_dict["sei"]= True
        #     params_dict["multiple_linear"]= False
        #     arg_space.directory = result_dir + "/deep/" + dataname + "/" + "run_" + str(j)
        #     experiment_blob = run_experiment(device, arg_space, params_dict)
            
    ## Run for deeper models with Entropy Loss
    for i in range(data_instances):
        dataname = f"data_80_{i+1}"
        arg_space.inputprefix = os.path.join(data_dir,dataname)
        
        # for j in range(runs):
        #     setup_seed(seeds[j])
        #     params_dict["sei"]= True
        #     params_dict["multiple_linear"]= False
        #     params_dict["entropy_loss"] = True
        #     params_dict["optimizer"]   = "sgd"
        #     params_dict["entropy_reg_value"] = 0.01
        #     arg_space.directory = result_dir + "/deep_ent/" + dataname + "/" + "run_" + str(j)
        #     experiment_blob = run_experiment(device, arg_space, params_dict)
            
    ## Run for deeper models with PE
    for i in range(data_instances):
        dataname = f"data_80_{i+1}"
        arg_space.inputprefix = os.path.join(data_dir,dataname)
   
        # for j in range(runs):
        #     setup_seed(seeds[j])
        #     print("Running for seed", seeds[j])
        #     params_dict["sei"]= True
        #     params_dict["use_posEnc"]= True
        #     params_dict["multiple_linear"]= False
        #     arg_space.directory = result_dir + "/deep_pe/" + dataname + "/" + "run_" + str(j)
        #     experiment_blob = run_experiment(device, arg_space, params_dict)


  ## Run for deeper models with PSSM
    for i in range(data_instances):
        dataname = f"data_80_{i+1}"
        arg_space.inputprefix = os.path.join(data_dir,dataname)
   
        # for j in range(runs):
        #     setup_seed(seeds[j])
        #     print("Running for seed", seeds[j])
        #     params_dict["sei"]= True
        #     params_dict["load_motif_weights"]= True
        #     params_dict["CNN_filtersize"] = [24]
        #     params_dict["optimizer"] = "sgd"
        #     params_dict["use_posEnc"]= True
        #     params_dict["multiple_linear"]= False
        #     arg_space.directory = result_dir + "/deep_pssm_pe/" + dataname + "/" + "run_" + str(j)
        #     experiment_blob = run_experiment(device, arg_space, params_dict)


    ## Run for deeper models with PSSM + PE and Entropy Loss
    
    # for i in range(data_instances):
    #     dataname = f"data_80_{i+1}"
    #     arg_space.inputprefix = os.path.join(data_dir,dataname)
   
    #     for j in range(runs):
    #         setup_seed(seeds[j])
    #         print("Running for seed", seeds[j])
    #         #params_dict["sei"]= True
    #         params_dict["load_motif_weights"]= True
    #         params_dict["CNN_filtersize"] = [24]
    #         #params_dict["use_posEnc"]= True
    #         #params_dict["multiple_linear"]= False
    #         params_dict["entropy_loss"] = True
    #         params_dict["optimizer"]   = "sgd"
    #         params_dict["entropy_reg_value"] = 0.0025
    #         arg_space.directory = result_dir + "/baseline_pssm_entropy/" + dataname + "/" + "run_" + str(j)
    #         experiment_blob = run_experiment(device, arg_space, params_dict)


if __name__ == "__main__":
    run_training_pipeline()