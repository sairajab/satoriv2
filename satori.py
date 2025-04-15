#!/usr/bin/env python
import os
import sys
import torch
import random
sys.path.insert(0,'satori')

from argparse import ArgumentParser
from torch.backends import cudnn
import numpy

#local imports
from satori.experiment import run_experiment, motif_analysis, get_results_for_shuffled
from satori.process_attention_mh import infer_intr_attention as mh_infer_intr_attention
from satori.process_attention import infer_intr_attention
from satori.process_fis import infer_intr_FIS
from satori.process_attnattr import infer_intr_ATTNATTR
from satori.utils import get_params_dict, annotate_motifs, Config, setup_seed
from evaluate_motif_interaction import *
from model_selection_base_model import Calibration
import json
####################################################################################################################
##################################--------------Argument Parsing--------------######################################
def parseArgs():
    """Parse command line arguments
    
    Returns
    -------
    a : argparse.ArgumentParser
    
    """
    parser = ArgumentParser(description='Main SATORI script.')
    parser.add_argument('-config', '--configfile', dest='config', type=str, required=True,
                    help="Provide the JSON file with all the arguments.")
    args = parser.parse_args()

    return args
####################################################################################################################


# def setup_seed(seed):
#     random.seed(seed)                          
#     numpy.random.seed(seed)                       
#     torch.manual_seed(seed)                    
#     torch.cuda.manual_seed(seed)               
#     torch.cuda.manual_seed_all(seed)           
#     torch.backends.cudnn.deterministic = True 



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
            test_resBlob_bg = get_results_for_shuffled(arg_space, params_dict, experiment_blob['net'], experiment_blob['criterion'], experiment_blob['test_loader'], device)
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
    
def main():
    #CUDA for pytorch
    head = "max"
    use_cuda = torch.cuda.is_available()
    device = torch.device(torch.cuda.current_device() if use_cuda else "cpu")
    #parse arguments
    args = parseArgs()
    arg_space = Config.from_json(args.config) 
    if arg_space.set_seed:
        print("Calling ...", arg_space.seed)
        setup_seed(arg_space.seed)
    #create params dictionary
    params_dict = get_params_dict(arg_space.hparamfile)
    print("Params dict: ", params_dict)
    experiment_blob = run_experiment(device, arg_space, params_dict)
    
    run_interactions_inference(experiment_blob, arg_space,params_dict, device)
    


    
if __name__ == "__main__": 
    
    main()
