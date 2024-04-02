import os
from analyze_motif_interaction import *
import numpy as np
import subprocess
import pandas
import glob as glob
import os
import shutil

def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))
    
    
def write_res(info_dict, auc, output_folder, th, method = "satori", summarize ="mean"):
    #print(info_dict[0])
    print(output_folder)
    
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
            
    with open(output_folder + "/avg_interaction_results_" +method+ ".txt", "w" ) as f:
        for i in range(len(th)):
            f.writelines(str(avg_prec[i]) + "\t" + str(avg_recall[i]) +"\t" + 
                         str(avg_f1s[i]) +"\t" + str(th[i]) + "\n")
        
        
def satori_inter():
    dataset_path = "data/ToyData/10_datasets"
    hyperparams = "modelsparam/best_hyperparams_ctf1.txt"
    exp = "final/CNN_ATTN/" 
    outputDir = "results/final/CNN_ATTN/"
    dataset = "ctf_2"
    cmd = "python satori.py " + os.path.join(dataset_path, dataset) +  " " + hyperparams + " -w 8" + \
                " --outDir "  + outputDir + dataset + "/E" + str(3) + " --mode test -v -s --background negative --intseqlimit 5000" + \
                    " --numlabels 2 --interactions --method SATORI --attrbatchsize 32 --deskload" + \
                " --tomtompath /s/jawar/i/nobackup/Saira/meme/src/tomtom --database /s/jawar/i/nobackup/Saira/motif_databases/Jaspar.meme"
    print(cmd)
    os.system(cmd)
                                    

def run(mode):
    
    rep = 3

    dataset_path = "data/ToyData/NEWDATA"
    d_files = [ "ctf_80pairs_eq2"] #, "ctf_60pairs", "ctf_80pairs"]
    pairs_n_meme = {"ctf_40pairs_eq" : ["tf_pairs_40.txt", "subset40.meme"] , 
                    "ctf_60pairs" : ["tf_pairs_60.txt", "subset60.meme"] ,
                    "ctf_80pairs_eq0" :  ["tf_pairs_80.txt", "subset80.meme"],
                    "ctf_80pairs_eq2" :  ["tf_pairs_80.txt", "subset80.meme"],
                    "ctf_80pairs_eq1" :  ["tf_pairs_80.txt", "subset80.meme"]}
    hyperparams_dirs = ["modelsparam/all_exps/baseline/baseline_entropy_0.01.tx"]
    for d in d_files:
        outdir = "results/newdata/" + d + "/"
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        for hyperparams_dir in hyperparams_dirs:
            modelparams = glob.glob(hyperparams_dir + "*")
            for param in modelparams:
                exp = param.split("/")[-1].split(".t")[0]
                in_dir = outdir + exp + "/"
                
                if not os.path.exists(in_dir):
                    os.mkdir(in_dir)
                    
                # if exp in ["baseline_relative_ns"]:
                #      continue

                    
                for i in range(rep):
                    
                    # if exp == "baseline_relative_ns" and i == 0:
                        
                    #     i += 1
                    exp_name = in_dir + "E" + str(i+1)

                                        
                    if mode == "train":
        
                        try:
                            # Run Satori
                            cmd = "python satori.py " + os.path.join(dataset_path, d) +  " " + param + " -w 8" + \
                            " --outDir "  + exp_name + " --mode train -v -s --background negative --intseqlimit 5000" + \
                            " --numlabels 2 --gt_pairs " + os.path.join("create_dataset",pairs_n_meme[d][0]) + \
                            " --method SATORI --set_seed --attrbatchsize 32 --deskload --seed " + str(i) + \
                            " --tomtompath /s/chromatin/p/nobackup/Saira/meme/src/tomtom --database " + os.path.join("create_dataset",pairs_n_meme[d][1])
                            print(cmd)
                            subprocess.call(cmd, shell=True)
                            #result = subprocess.call(cmd, shell=True, check=True,  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                            #os.system(cmd)
                            #if result.stdout:
                            #    print(f"Standard Output of '{cmd}':\n{result.stdout}")
                        except subprocess.CalledProcessError as e:
                            # Handle any errors that occur during command execution
                            print(f"Error executing command: {cmd}")
                            print(f"Return code: {e.returncode}")
                            print(f"Standard Error of '{cmd}':\n{e.stderr}")
                    elif mode == "interactions":
                        
                        satori_path = exp_name+"/Interactions_SATORI/"
                        print(satori_path)
                        
                        if os.path.exists(satori_path):
                            print("*******Removed directory...")
                            remove(satori_path)
                            
                        motif_path = exp_name+"/Motif_Analysis/"
                        if os.path.exists(motif_path):
                            remove(motif_path)
                            print("*******Removed Motifs directory...")
                        # Run Satori
                        cmd = "python satori.py " + os.path.join(dataset_path, d) +  " " + param + " -w 8" + \
                            " --outDir "  + exp_name + " --mode test -v -s --background negative --intseqlimit 5000" + \
                            " --numlabels 2 --motifanalysis --interactions --interactionanalysis --gt_pairs " + os.path.join("create_dataset",pairs_n_meme[d][0]) + \
                            " --method SATORI --attrbatchsize 32 --deskload --seed " + str(i) + \
                            " --tomtompath /s/jawar/i/nobackup/Saira/meme/src/tomtom --database " + os.path.join("create_dataset",pairs_n_meme[d][1])
                        print(cmd)
                        subprocess.call(cmd, shell=True)
                        
                    else:
                        
                        print("Write CODE to do something else..............")
                        
                        
                
                
                    
                            
            
            
    
    # hyperparams_file = "modelsparam/best_hyperparams_ctf1.txt"
    # exp = "final/CNN_ATTN/" 
    # outputDir = "results/final/CNN_ATTN/"
    # #

    # datasets = ["ctf_0", "ctf_1", "ctf_2", "ctf_3", "ctf_4","ctf_5","ctf_6",
    #            "ctf_7", "ctf_8","ctf_9"] #["ctf_2","ctf_3"]
    
    # final_res = open(outputDir + "/res_best_satori.csv", "w" )
    # rep = 3
    # results = {}
    # results_fis = {}
    # auc = {}

    # for dataset in datasets:
    #     results[dataset] = []
    #     results_fis[dataset] = []
    #     auc[dataset] = []
        
    #     for i in range(rep):
    #         if not os.path.exists(outputDir + dataset):
    #             os.mkdir(outputDir + dataset)
                
    #         exp_name = exp + dataset + "/E" + str(i+1) + "/"
            
    #         print(mode == "train")
    #         if mode == "train":

    #             # Run Satori
    #             cmd = "python satori.py " + os.path.join(dataset_path, dataset) +  " " + hyperparams + " -w 8" + \
    #             " --outDir "  + outputDir + dataset + "/E" + str(i+1) + " --mode train -v -s --background negative --intseqlimit 5000" + \
    #                 " --numlabels 2 --motifanalysis --interactions --method BOTH --attrbatchsize 32 --deskload" + \
    #             " --tomtompath /s/jawar/i/nobackup/Saira/meme/src/tomtom --database /s/jawar/i/nobackup/Saira/motif_databases/Jaspar.meme"
    #             print(cmd)
    #             os.system(cmd)
    #             # ps, rs , f1s, th = run_interaction_evaluation(exp_name)
    #             # results[dataset].append((ps,rs, f1s))
    #         if mode == "test":
    #             #do nothing
    #             print(dataset)
                
    #         else:
    #             print(dataset)
                
    #             ps, rs , f1s, th = run_interaction_evaluation(exp_name)
    #             results[dataset].append((ps,rs, f1s))
                
    #             ps, rs , f1s, th = run_interaction_evaluation_fis(exp_name)
    #             results_fis[dataset].append((ps,rs, f1s))
                
                
    #             scores = open("results/" + exp_name + "modelRes_results.txt", "r").readlines()[-1][:-1]
    #             auc[dataset].append(tuple(map(float,scores.split("\t"))))
                
                
    #     if mode != "train" and mode != "test":
    #         print("writing resultssss")
    #         write_res(results[dataset],auc[dataset],"results/" +  exp + dataset, th)
    #         write_res(results_fis[dataset],auc[dataset],"results/" +  exp + dataset, th, method = "fis")
            
    #     if mode == "test":
    #         AUC = ""
    #         path = "results/final/CNN_ATTN/" + dataset
    #         with open(path + "/avg_auc.txt", "r" ) as fr:
    #             print("here")
    #             AUC = fr.readlines()[0].split('\t')[1]
            
    #         df = pandas.read_csv(path + "/avg_interaction_results_satori.txt", sep='\t', lineterminator='\n',header = None)
    #         print(df.loc[100].values)
    #         y = df.loc[100].values
            
                
            
            
    #         final_res.writelines(dataset +","+ str(y[0]) +","+ str(y[1])+","+str(y[2]) + "," + AUC + "\n")

            
                

                
            
            
    


if __name__ == "__main__":
    #satori_inter()
    run("train")
    #run("combine")
    #run("interactions")



