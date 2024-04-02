import os
from analyze_motif_interaction import *
import numpy as np
import subprocess
import pandas


def satori_inter():
    dataset_path = "data/ToyData/10_datasets"
    hyperparams = "modelsparam/best_hyperparams_relative_ctf1.txt"
    exp = "final/CNN_NS_REL_ATTN/"
    outputDir = "results/final/CNN_NS_REL_ATTN/"
    dataset = "ctf_0"
    cmd = "python satori.py " + os.path.join(dataset_path, dataset) +  " " + hyperparams + " -w 8" + \
                " --outDir "  + outputDir + dataset + "/E" + str(3) + " --mode test -v -s --background negative --intseqlimit 5000" + \
                    " --numlabels 2 --interactions --method SATORI --attrbatchsize 32 --deskload" + \
                " --tomtompath /s/jawar/i/nobackup/Saira/meme/src/tomtom --database /s/jawar/i/nobackup/Saira/motif_databases/Jaspar.meme"
    print(cmd)
    os.system(cmd)
    
    
def write_res(info_dict, auc, output_folder, th, method = "satori", summarize ="mean"):
    #print(info_dict[0])
    if summarize == "mean":
        avg_prec = np.mean(np.array([info_dict[0][0],info_dict[1][0], info_dict[2][0]]), axis = 0)
        avg_recall = np.mean(np.array([info_dict[0][1], info_dict[1][1], info_dict[2][1]]), axis = 0)
        avg_f1s = np.mean(np.array([info_dict[0][2], info_dict[1][2], info_dict[2][2]]), axis = 0)
        
    if summarize == "max":
        
        avg_prec = np.max(np.array([info_dict[0][0],info_dict[1][0], info_dict[2][0]]), axis = 0)
        avg_recall = np.max(np.array([info_dict[0][1], info_dict[1][1], info_dict[2][1]]), axis = 0)
        avg_f1s = np.max(np.array([info_dict[0][2], info_dict[1][2], info_dict[2][2]]), axis = 0)

    
    with open(output_folder + "/avg_auc.txt", "w" ) as fw:
            fw.writelines(str(np.mean(np.array([auc[0][0],auc[1][0],auc[2][0]])) ) + "\t" + 
                          str(np.mean(np.array([auc[0][1],auc[1][1],auc[2][1]])) ) +"\t" + 
                         str(np.mean(np.array([auc[0][2],auc[1][2],auc[2][2]])) ) +"\t" + "\n")
            
    with open(output_folder + "/avg_interaction_results_"+ method +".txt", "w" ) as f:
        for i in range(len(th)):
            f.writelines(str(avg_prec[i]) + "\t" + str(avg_recall[i]) +"\t" + 
                         str(avg_f1s[i]) +"\t" + str(th[i]) + "\n")
        
        
                                
def run(mode):

    dataset_path = "data/ToyData/10_datasets"
    hyperparams = "modelsparam/best_hyperparams_relative_ctf1_ns.txt"
    exp = "final/CNN_NS_REL_ATTN/"
    outputDir = "results/final/CNN_NS_REL_ATTN/"

    datasets = ["ctf_0", "ctf_1", "ctf_2", "ctf_3", "ctf_4","ctf_5","ctf_6",
                "ctf_7", "ctf_8","ctf_9"] #["ctf_2","ctf_3"]
    
    final_res = open(outputDir + "/res_satori.csv", "w" )

    rep = 3
    results = {}
    results_fis = {}
    auc = {}

    for dataset in datasets:
        results[dataset] = []
        results_fis[dataset] = []
        auc[dataset] = []
        
        for i in range(rep):
            if not os.path.exists(outputDir + dataset):
                os.mkdir(outputDir + dataset)
                
            exp_name = exp + dataset + "/E" + str(i+1) + "/"
            
            print(mode == "train")
            if mode == "train":

                # Run Satori
                cmd = "python satori.py " + os.path.join(dataset_path, dataset) +  " " + hyperparams + " -w 8" + \
                " --outDir "  + outputDir + dataset + "/E" + str(i+1) + " --mode train -v -s --background negative --intseqlimit 5000" + \
                    " --numlabels 2 --motifanalysis --interactions --method BOTH --attrbatchsize 32 --deskload" + \
                " --tomtompath /s/jawar/i/nobackup/Saira/meme/src/tomtom --database /s/jawar/i/nobackup/Saira/motif_databases/Jaspar.meme"
                print(cmd)
                os.system(cmd)
            elif mode == "mh_intercations":
                cmd = "python satori.py " + os.path.join(dataset_path, dataset) +  " " + hyperparams + " -w 8" + \
                " --outDir "  + outputDir + dataset + "/E" + str(i+1) + " --mode test -v -s --background negative --intseqlimit 5000" + \
                    " --numlabels 2 --interactions --method SATORI --attrbatchsize 32 --deskload" + \
                " --tomtompath /s/jawar/i/nobackup/Saira/meme/src/tomtom --database /s/jawar/i/nobackup/Saira/motif_databases/Jaspar.meme"
                print(cmd)
                os.system(cmd)
                # ps, rs , f1s, th = run_interaction_evaluation(exp_name)
                # results[dataset].append((ps,rs, f1s))
            elif mode == "test":
                    #do nothing
                print(dataset)
            else:
                print(dataset)
                
                ps, rs , f1s, th = run_interaction_evaluation(exp_name)
                results[dataset].append((ps,rs, f1s))
                
                ps, rs , f1s, th = run_interaction_evaluation_fis(exp_name)
                results_fis[dataset].append((ps,rs, f1s))
                
                scores = open("results/"+ exp_name + "modelRes_results.txt", "r").readlines()[-1][:-1]
                auc[dataset].append(tuple(map(float,scores.split("\t"))))
                
        if mode == "combine":
            write_res(results[dataset],auc[dataset], "results/" + exp + dataset, th)
            write_res(results_fis[dataset],auc[dataset], "results/"+ exp + dataset, th, "fis")

            
        
        if mode == "test":
            AUC = ""
            path = outputDir + dataset
            with open(path + "/avg_auc.txt", "r" ) as fr:
                AUC = fr.readlines()[0].split('\t')[1]
            
            df = pandas.read_csv(path + "/avg_interaction_results_satori.txt", sep='\t', lineterminator='\n',header = None)
            print(df.loc[100].values)
            y = df.loc[100].values
            
                
            
            
            final_res.writelines(dataset +","+ str(y[0]) +","+ str(y[1])+","+str(y[2]) + "," + AUC + "\n")
                

                
            
            
    


if __name__ == "__main__":
    #run("train")
    #run("combine")
    run("test")
    #satori_inter()



