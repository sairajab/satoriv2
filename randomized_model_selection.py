
from satori.train_model import *
from satori.evaluate_model import *
from satori.utils import get_params_dict
from sklearn.model_selection import ParameterSampler
from satori.experiment import load_datasets
from satori.modelsold import AttentionNet
import torch
import json


import json
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

def setup_seed(seed):
    random.seed(seed)                           # Python random seed
    np.random.seed(seed)                        # NumPy random seed
    torch.manual_seed(seed)                     # CPU random seed
    torch.cuda.manual_seed(seed)                # Single GPU random seed
    torch.cuda.manual_seed_all(seed)            # All GPU random seeds
    torch.backends.cudnn.deterministic = True   # Ensure deterministic results for CUDA operations


def Calibration(dataset_path, numLabels, seq_length, device, search_space_dict, out_dir,  n_iter = 30):
    """
    Perform calibration of the model using random search.
    """
    
    # Get the base parameters
    params_dict = get_params_dict("modelsparam/all_exps/simulated/basic/baseline.txt")
    param_list = list(ParameterSampler(search_space_dict, n_iter=n_iter, random_state=42))
    dataset_name = "Simulated"
    deskLoad = False
    mode = "train"
    splitperc = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_epochs = 50
    # Define number of runs per experiment
    num_runs = 3  
    seeds = [42, 1337, 2024]  # Different seeds for each run

    # Initialize tracking variables
    best_avg_auc = 0
    best_avg_params = None
    all_results = []

    for i, params in enumerate(param_list):
        print(f"\n### Running Experiment {i+1}/{len(param_list)} ###\n")
        auc_scores = []
        for run_idx, seed in enumerate(seeds):
            print(f"   Run {run_idx+1}/{num_runs} (Seed: {seed})")
            
            # Set random seed
            setup_seed(seed)
            
            # Load dataset
            train_loader, train_indices, _, _, valid_loader, valid_indices, _, seq_len = load_datasets(
                dataset_path, out_dir, params['batch_size'], dataset_name, numLabels, deskLoad, mode, splitperc
            )
            
            # Update parameters
            params_dict.update(params)
            model = AttentionNet(numLabels, params_dict, device=device, seq_len=seq_len, getAttngrad=False).to(device)
            criterion = nn.CrossEntropyLoss(reduction='mean')
            
            # Select optimizer
            if params["optimizer"] == "adam":
                optimizer = optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
            else:
                optimizer = optim.SGD(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"], momentum=params["momentum"])
            
            # Training Loop
            best_valid_loss = float("inf")
            best_valid_auc = 0
            
            for epoch in range(1, max_epochs + 1):
                res_train = trainRegular(model, device, train_loader, optimizer, criterion, ent_loss=False, entropy_reg_weight=0.005, reg=False)
                res_train_loss, res_train_auc = res_train[0], np.mean(res_train[1])
                
                res_valid = evaluateRegular(model, device, valid_loader, criterion, out_dir+"/Stored_Values")
                res_valid_loss, res_valid_auc = res_valid[0], res_valid[1]
                
                print(f"      Epoch {epoch}: Train Loss={res_train_loss:.4f}, Train AUC={res_train_auc:.4f}, Valid Loss={res_valid_loss:.4f}, Valid AUC={res_valid_auc:.4f}")
                
                # Track the best model for this run
                if res_valid_auc > best_valid_auc:
                    best_valid_auc = res_valid_auc
                    best_valid_loss = res_valid_loss

            print(f"Best Validation AUC for Run {run_idx+1}: {best_valid_auc:.4f}\n")
            # Store best AUC from this run
            auc_scores.append(best_valid_auc)
            
            # Save results for this run
            all_results.append({
                "experiment": i+1,
                "run": run_idx+1,
                "seed": seed,
                "params": params,
                "valid_loss": best_valid_loss,
                "valid_auc": best_valid_auc
            })
            
        # Compute average AUC across the 3 runs
        avg_auc = np.mean(auc_scores)
        print(f"### Average AUC for Experiment {i+1}: {avg_auc:.4f} ###\n")
        # Update overall best model across all runs
        if avg_auc > best_avg_auc:
            best_avg_auc = avg_auc
            best_avg_params = params

    # Save results to JSON
    with open(out_dir+"/all_experiment_results.json", "w") as f:
        json.dump(all_results, f, indent=4)

    with open(out_dir+"/best_overall_params.json", "w") as f:
        json.dump(best_avg_params, f, indent=4)

    print(f"\n### Best Overall Params Selected with AUC: {best_avg_auc:.4f} ###")
    print(best_avg_params)
    
    return out_dir+"/best_overall_params.json"
    
    
    

if __name__ == "__main__":
    
    param_dist = {
    "lr": [1e-4, 1e-3, 1e-2],
    "batch_size": [16, 32, 64],
    "linear_layer_size": [256, 512, 1024],
    "dropout_rate": [0.1, 0.2, 0.3],
    "weight_decay": [0, 1e-4, 1e-3],
    "num_multiheads": [2, 4, 8],
    "singlehead_size": [32,64, 128],
    "CNN_filters": [[256], [380], [512]],  # Number of Conv1D filters
    "CNN_filtersize": [[13], [16], [24]],  # Kernel size for Conv1D
    "momentum": [0.8, 0.9, 0.99],  # For SGD
    "optimizer": ["adam", "sgd"]
}
