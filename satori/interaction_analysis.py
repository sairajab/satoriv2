from satori.experiment import run_experiment, motif_analysis, get_results_for_shuffled
from satori.process_attention_mh import infer_intr_attention as mh_infer_intr_attention
from satori.process_attention import infer_intr_attention
from satori.process_fis import infer_intr_FIS
from satori.process_attnattr import infer_intr_ATTNATTR
from satori.utils import get_params_dict, annotate_motifs, Config, setup_seed
from satori.evaluate_motif_interaction import *


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
        print("Running interaction analysis...", arg_space.methodType)
        if arg_space.methodType in ['RAWATTN','RAWnATTR','ALL']:
            if head == "multi":
                mh_infer_intr_attention(experiment_blob, params_dict, arg_space)
            else:
                infer_intr_attention(experiment_blob, params_dict, arg_space)
        
        if arg_space.methodType in ['FIS','ALL']:
            infer_intr_FIS(experiment_blob, params_dict, arg_space, device)
        
        if arg_space.methodType in ['ATTNATTR', 'RAWnATTR' , 'ALL']:
            experiment_blob['res_test_bg'] = []
            experiment_blob['train_loader'] = []
            experiment_blob['CNN_weights'] = []
            experiment_blob['net'] = []
            infer_intr_ATTNATTR(experiment_blob, params_dict, arg_space, device = device)

            
    if arg_space.interactionsAnalysis:
        
        motif_weights = params_dict['load_motif_weights']
        pairs_file = arg_space.pairs_file
                   
        if arg_space.methodType in ['RAWATTN','RAWnATTR','ALL']:            
            _,_,_,_ = run_interaction_evaluation(output_dir+"/",pairs_file,arg_space.tfDatabase, method = 'RAWATTN' , motif_weights=motif_weights)
                
        if arg_space.methodType in ['FIS','ALL']:
            _,_,_,_ = run_interaction_evaluation(output_dir + "/",pairs_file,arg_space.tfDatabase, method='FIS', motif_weights=motif_weights)
            
        if arg_space.methodType in ['ATTNATTR', 'RAWnATTR' , 'ALL']:
            _,_,_,_ = run_interaction_evaluation(output_dir + "/",pairs_file,arg_space.tfDatabase, method='ATTNATTR', motif_weights=motif_weights)
        
        
        print("DONE!!!")
    