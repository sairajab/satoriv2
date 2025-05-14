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
from satori.evaluate_motif_interaction import *
from model_selection_base_model import Calibration
import json
from satori.interaction_analysis import run_interactions_inference
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
