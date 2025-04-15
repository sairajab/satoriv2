
from create_dataset.generate_data import SimulatedDataGenerator
from randomized_model_selection import Calibration
from satori.experiment import run_experiment
import torch
import os
from satori.utils import Config , load_config, get_params_dict
from satori.experiment import run_experiment, motif_analysis, get_results_for_shuffled
from satori.process_attention_mh import infer_intr_attention as mh_infer_intr_attention
from satori.process_attention import infer_intr_attention
from satori.process_fis import infer_intr_FIS
from satori.process_attnattr import infer_intr_ATTNATTR
from satori.utils import get_params_dict, annotate_motifs, setup_seed
from evaluate_motif_interaction import *


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

def run_interactions_inference(experiment_blob, arg_space,params_dict, device):
    head = "max"
    output_dir = experiment_blob['output_dir']
    test_resBlob = experiment_blob['res_test']
    CNNWeights = experiment_blob['CNN_weights']

    if arg_space.motifAnalysis:
        motif_dir_pos, _ = motif_analysis(test_resBlob, CNNWeights, arg_space, params_dict)
        if arg_space.intBackground == 'negative':
            motif_dir_neg, _ = motif_analysis(test_resBlob, CNNWeights,  arg_space, params_dict, for_background=True)
        if arg_space.intBackground == 'shuffle':
            test_resBlob_bg = get_results_for_shuffled(arg_space, params_dict, experiment_blob['net'], experiment_blob['criterion'], experiment_blob['test_loader'], device)
            experiment_blob['res_test_bg'] = test_resBlob_bg[0]
            experiment_blob['test_loader_bg'] = test_resBlob_bg[1]
            motif_dir_neg, _ = motif_analysis(test_resBlob_bg[0], CNNWeights, arg_space, params_dict, for_background=True)
    else:
        motif_dir_pos = output_dir + '/Motif_Analysis'
        motif_dir_neg = output_dir + '/Motif_Analysis_Negative'
    
    if not os.path.exists(motif_dir_pos) or not os.path.exists(motif_dir_neg):
        raise Exception("Please extract motifs from the network outputs first (hint: use --motifanalysis)")
    experiment_blob['motif_dir_pos'] = motif_dir_pos
    experiment_blob['motif_dir_neg'] = motif_dir_neg

    if arg_space.annotateTomTom != None:
        if arg_space.verbose:
            print("Annotating motifs...")
        annotate_motifs(arg_space.annotateTomTom, motif_dir_pos)
        annotate_motifs(arg_space.annotateTomTom, motif_dir_neg)

    if arg_space.featInteractions:
        if arg_space.intBackground == 'shuffle' and not arg_space.motifAnalysis:
            test_resBlob_bg = get_results_for_shuffled(arg_space, params_dict, experiment_blob['net'], experiment_blob['criterion'], experiment_blob['test_loader'], device, motifweights=params_dict['load_motif_weights'])
            experiment_blob['res_test_bg'] = test_resBlob_bg[0]
            experiment_blob['test_loader_bg'] = test_resBlob_bg[1]
        if arg_space.methodType in ['SATORI','ALL']:
            if head == "multi":
                mh_infer_intr_attention(experiment_blob, params_dict, arg_space)
            else:
                infer_intr_attention(experiment_blob, params_dict, arg_space)
        
        if arg_space.methodType in ['FIS','ALL']:
            infer_intr_FIS(experiment_blob, params_dict, arg_space, device)
        
        if arg_space.methodType in ['ATTNATTR', 'ALL']:
            experiment_blob['res_test_bg'] = []
            experiment_blob['train_loader'] = []
            experiment_blob['CNN_weights'] = []
            experiment_blob['net'] = []
            infer_intr_ATTNATTR(experiment_blob, params_dict, arg_space, device = device)

            
    if arg_space.interactionsAnalysis:
        
        motif_weights = params_dict['load_motif_weights']
        pairs_file = arg_space.pairs_file
                   
        if arg_space.methodType in ['SATORI','ALL']:            
            _,_,_,_ = run_interaction_evaluation(output_dir+"/",pairs_file,arg_space.tfDatabase, motif_weights=motif_weights)
                
        if arg_space.methodType in ['FIS','ALL']:
            _,_,_,_ = run_interaction_evaluation(output_dir + "/",pairs_file,arg_space.tfDatabase, method="FIS", motif_weights=motif_weights)
            
        if arg_space.methodType in ['ATTNATTR', 'ALL']:
            _,_,_,_ = run_interaction_evaluation(output_dir + "/",pairs_file,arg_space.tfDatabase, method="ATTNATTR", motif_weights=motif_weights)
        
        
        print("DONE!!!")
    

def run_inference_pipeline():
    
    ## Generate Data
    path_to_meme = '/s/chromatin/p/nobackup/Saira/motif_databases/JASPAR2024_CORE_non-redundant_pfms.meme'
    pairs_file = "create_dataset/clustered_tf_pairs_80.txt"
    data_dir = "data/LongSimulated_Data/Data-80/LongSeqs/" 
    result_dir = "results/Data-80/LongSeqs/Clustered/"
    

    ## Loop through three instances of baseline model training using best model
    arg_space = Config.from_json("results/Data-80/LongSeqs/Clustered/model_selection/best_overall_params.json")
    params_dict = get_params_dict(arg_space.hparamfile)
    #params_dict = load_config(config_path)
    print(params_dict)
    runs = 1
    seeds = [42, 1337, 2024] 
    data_instances = 1 
    device = "cuda"                       
                        
    for i in range(data_instances):
        dataname = f"data_80_{i+1}"
        arg_space.inputprefix = os.path.join(data_dir,dataname)
        best_seed = 0
        best_val_auc = 0
        for j in range(runs):
            
            arg_space.directory = result_dir + "/baseline_pe/" + dataname + "/" + "run_" + str(j)
            val_auc = get_valid_auc_from_file(arg_space.directory + "/modelRes_Val_results.txt")
            if best_val_auc < val_auc:
                best_val_auc = val_auc
                best_exp = arg_space.directory
                best_seed = j
                
        arg_space.mode = "test"
        arg_space.motifAnalysis = True
        arg_space.featInteractions = True
        arg_space.interactionsAnalysis = True
        arg_space.directory = best_exp
        arg_space.methodType = 'SATORI'
        arg_space.intBackground = 'negative'
        params_dict["use_posEnc"] = True
        # params_dict["load_motif_weights"]= True
        # params_dict["CNN_filtersize"] = [24]
        params_dict["optimizer"]   = "sgd"
        # params_dict["entropy_reg_value"] = 0.07
        # params_dict["entropy_loss"] = True
        # params_dict["sei"]= True
        # params_dict["multiple_linear"] = False
        setup_seed(best_seed)
        print("Best run ", best_exp , best_val_auc)
        experiment_blob = run_experiment(device, arg_space, params_dict)
        print(experiment_blob.keys())
        run_interactions_inference(experiment_blob, arg_space,params_dict,device)
            



if __name__ == "__main__":
    run_inference_pipeline()