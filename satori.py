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
from satori.utils import get_params_dict, annotate_motifs
from model_selection import Calibration
from analyze_motif_interaction import *
####################################################################################################################
##################################--------------Argument Parsing--------------######################################
def parseArgs():
    """Parse command line arguments
    
    Returns
    -------
    a : argparse.ArgumentParser
    
    """
    parser = ArgumentParser(description='Main SATORI script.')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', 
                        default=False, help="verbose output [default is quiet running]")
    parser.add_argument('-o','--outDir', dest='directory', type=str,
                        action='store', help="output directory", default='')
    parser.add_argument('-m','--mode', dest='mode', type=str,
                        action='store', help="Mode of operation: train or test.", default='train')     
    parser.add_argument('--deskload', dest='deskLoad',
                        action='store_true', default=False,
                        help="Load dataset from desk. If false, the data is converted into tensors and kept in main memory (not recommended for large datasets).")  
    parser.add_argument('-w','--numworkers', dest='numWorkers', type=int,
                        action='store', help="Number of workers used in data loader. For loading from the desk, use more than 1 for faster fetching.", default=1)        
    parser.add_argument('--splitperc', dest='splitperc', type=float, action='store',
                        help="Pecentages of test, and validation data splits, eg. 10 for 10 percent data used for testing and validation.", default=10)
    parser.add_argument('--motifanalysis', dest='motifAnalysis',
                        action='store_true', default=False,
                        help="Analyze CNN filters for motifs and search them against known TF database.")
    parser.add_argument('--filtersanalysis', dest='filterAnalysis',
                        action='store_true', default=False,
                        help="Analyze CNN filters for motifs based on annotation file.")
    parser.add_argument('--scorecutoff', dest='scoreCutoff', type=float,
                        action='store', default=0.65,
                        help="In case of binary labels, the positive probability cutoff to use.")
    parser.add_argument('--tomtompath', dest='tomtomPath',
                        type=str, action='store', default=None,
                        help="Provide path to where TomTom (from MEME suite) is located.") 
    parser.add_argument('--database', dest='tfDatabase', type=str, action='store',
                        help="Search CNN motifs against known TF database. Default is Human CISBP TFs.", default=None)
    parser.add_argument('--annotate',dest='annotateTomTom',type=str,action='store',
                        default=None, help="Annotate tomtom motifs. The value of this variable should be path to the database file used for annotation. Default is None.")                   
    parser.add_argument('-i','--interactions', dest='featInteractions',
                        action='store_true', default=False,
                        help="Self attention based feature(TF) interactions analysis.")
    parser.add_argument('--interactionanalysis', dest='interactionsAnalysis',
                        action='store_true', default=False,
                        help="interactions analysis with ground truth values")
    
    parser.add_argument('-b','--background', dest='intBackground', type=str,
                        action='store', default=None,
                        help="Background used in interaction analysis: shuffle (for di-nucleotide shuffled sequences with embedded motifs.), negative (for negative test set). Default is not to use background (and significance test).")
    parser.add_argument('--attncutoff', dest='attnCutoff', type=float,
                        action='store', default=0.04,
                        help="Attention cutoff value. For a given interaction, it should have an attention value at least as high as this value across all examples.") #In human promoter DHSs data analysis, lowering the cutoff leads to more TF interactions. 
    parser.add_argument('--fiscutoff', dest='fisCutoff', type=float,
                        action='store', default=0,
                        help="FIS score cutoff value. For a given interaction, it should have an FIS score at least as high as this value across all examples.") 
    parser.add_argument('--intseqlimit', dest='intSeqLimit', type=int,
                        action='store', default = -1,
                        help="A limit on number of input sequences to test. Default is -1 (use all input sequences that qualify).")
    parser.add_argument('-s','--store', dest='storeInterCNN',
                        action='store_true', default=False,
                        help="Store per batch attention and CNN outpout matrices. If false, the are kept in the main memory.")
    parser.add_argument('--numlabels', dest='numLabels', type=int,
                        action='store', default = 2,
                        help="Number of labels. 2 for binary (default). For multi-class, multi label problem, can be more than 2. ")
    parser.add_argument('--tomtomdist', dest='tomtomDist', type=str,
                        action='store', default = 'ed',
                        help="TomTom distance parameter (pearson, kullback, ed etc). Default is euclidean (ed). See TomTom help from MEME suite.")
    parser.add_argument('--tomtompval', dest='tomtomPval', type=float,
                        action='store', default = 0.05,
                        help="Adjusted p-value cutoff from TomTom. Default is 0.05.")
    parser.add_argument('--testall', dest='testAll',
                        action='store_true', default=False,
                        help="Test on the entire dataset (default False). Useful for interaction/motif analysis.")
    parser.add_argument('--useall', dest='useAll',
                        action='store_true', default=False,
                        help="Use all examples in multi-label problem instead of using precision based example selection.  Default is False.")
    parser.add_argument('--precisionlimit', dest='precisionLimit', type=float,
                        action='store', default = 0.50,
                        help="Precision limit to use for selecting examples in case of multi-label problem.")
    parser.add_argument('--attrbatchsize', dest='attrBatchSize', type=int,
                        action='store', default = 12,
                        help="Batch size used while calculating attributes for FIS scoring. Default is 12.")
    parser.add_argument('--method', dest='methodType', type=str,
                        action='store', default='SATORI',
                        help="Interaction scoring method to use; options are: SATORI, FIS, or BOTH. Default is SATORI.")
    parser.add_argument('inputprefix', type=str,
                        help="Input file prefix for the bed/text file and the corresponding fasta file (sequences).")
    parser.add_argument('hparamfile',type=str,
                        help='Name of the hyperparameters file to be used.')
    parser.add_argument('--gt_pairs', dest='pairs_file',
                        action='store', default='',
                        help="Path to groud truth pairs file")
    parser.add_argument('--finetune_model', dest='finetune_model_path',
                        action='store', default=None,
                        help="Path to the pre-trained model ")
    parser.add_argument('--set_seed', dest='set_seed',
                        action='store_true', default=False,
                        help="Set seed or not")  

    parser.add_argument('--seed', dest='seed',
                        action='store', type=int, default=0,
                        help="Seed to intialize model")   
    parser.add_argument('--motifweights', dest='load_motif_weights',
                        action='store_true', default=False,
                        help="Load weights of first CNN from motifs PWM and freeze them, by default false hence weights are randomly intialized and trained.).")  

    args = parser.parse_args()

    return args
####################################################################################################################


def setup_seed(seed):
    random.seed(seed)                          
    numpy.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True 
    
    
def main():
    #CUDA for pytorch
    head = "max"
    use_cuda = torch.cuda.is_available()
    device = torch.device(torch.cuda.current_device() if use_cuda else "cpu")
    cudnn.benchmark = True
    arg_space = parseArgs()
    if arg_space.set_seed:
        print("Calling ...", arg_space.seed)
        setup_seed(arg_space.seed)
    #create params dictionary
    params_dict = get_params_dict(arg_space.hparamfile)
    experiment_blob = run_experiment(device, arg_space, params_dict)
    output_dir = experiment_blob['output_dir']
    test_resBlob = experiment_blob['res_test']
    CNNWeights = experiment_blob['CNN_weights']

    if arg_space.motifAnalysis:
        #print(test_resBlob)
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
        if arg_space.methodType in ['SATORI','BOTH']:
            if head == "multi":
                mh_infer_intr_attention(experiment_blob, params_dict, arg_space)
            else:
                infer_intr_attention(experiment_blob, params_dict, arg_space)
        
        if arg_space.methodType in ['FIS','BOTH']:
            infer_intr_FIS(experiment_blob, params_dict, arg_space, device)
            
    if arg_space.interactionsAnalysis:
        
        print(arg_space.load_motif_weights)
        motif_weights = arg_space.load_motif_weights
        pairs_file = arg_space.pairs_file
                   
        if arg_space.methodType in ['SATORI','BOTH']:            
            _,_,_,_ = run_interaction_evaluation(output_dir+"/",pairs_file, motif_weights=motif_weights)
                
        if arg_space.methodType in ['FIS','BOTH']:
            _,_,_,_ = run_interaction_evaluation(output_dir + "/",pairs_file, method="FIS", motif_weights=motif_weights)

        print("DONE!!!")

def hyperparameter_selection():

    arg_space = parseArgs()
    

    best_Res = Calibration(arg_space)

    print(best_Res)


if __name__ == "__main__":                 
    main()
