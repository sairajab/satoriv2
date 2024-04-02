import pandas as pd
import numpy as np
from glob import glob
import os
def write_res(info_dict, auc, output_folder, th, method = "satori", summarize ="mean"):
    #print(info_dict[0])
    print(output_folder)
    
    if summarize == "mean":
        avg_prec = np.mean(np.array([info_dict[0][0],info_dict[1][0], info_dict[2][0]]), axis = 0)
        avg_recall = np.mean(np.array([info_dict[0][1], info_dict[1][1], info_dict[2][1]]), axis = 0)
        avg_f1s = np.mean(np.array([info_dict[0][2], info_dict[1][2], info_dict[2][2]]), axis = 0)
        avg_stds = np.std(np.array([info_dict[0][2], info_dict[1][2], info_dict[2][2]]), axis = 0)
        print(avg_stds)

        
        
        with open(output_folder + "/avg_auc.txt", "w" ) as fw:
            fw.writelines(str(np.mean(np.array([auc[0][0],auc[1][0],auc[2][0]])) ) + "\t" + 
                          str(np.mean(np.array([auc[0][1],auc[1][1],auc[2][1]])) ) +"\t" + 
                         str(np.mean(np.array([auc[0][2],auc[1][2],auc[2][2]])) ) +"\t" + "\n")
            
        with open(output_folder + "/avg_interaction_results_" +method+ ".txt", "w" ) as f:
            for i in range(len(th)):
                f.writelines(str(avg_prec[i]) + "\t" + str(avg_recall[i]) +"\t" + 
                            str(avg_f1s[i]) +"\t" + str(th[i]) + "\n")
        
    if summarize == "max":
        
        avg_prec = np.max(np.array([info_dict[0][0],info_dict[1][0], info_dict[2][0]]), axis = 0)
        avg_recall = np.max(np.array([info_dict[0][1], info_dict[1][1], info_dict[2][1]]), axis = 0)
        avg_f1s = np.max(np.array([info_dict[0][2], info_dict[1][2], info_dict[2][2]]), axis = 0)

    
        with open(output_folder + "/best_auc.txt", "w" ) as fw:
                fw.writelines(str(np.mean(np.array([auc[0][0],auc[1][0],auc[2][0]])) ) + "\t" + 
                            str(np.mean(np.array([auc[0][1],auc[1][1],auc[2][1]])) ) +"\t" + 
                            str(np.mean(np.array([auc[0][2],auc[1][2],auc[2][2]])) ) +"\t" + "\n")
                
        with open(output_folder + "/best_interaction_results_" +method+ ".txt", "w" ) as f:
            for i in range(len(th)):
                f.writelines(str(avg_prec[i]) + "\t" + str(avg_recall[i]) +"\t" + 
                            str(avg_f1s[i]) +"\t" + str(th[i]) + "\n")
            


def read_results(dirr):
    
    rep = 3
    auc = []
    inter_res = []
    
    for i in range(rep):
        
        outdir = dirr + "/E" + str(i+1)
        
        

        
        scores = open(outdir + "/modelRes_results.txt", "r").readlines()[-1][:-1]
        auc.append(tuple(map(float,scores.split("\t"))))
            
        txt = glob(outdir + "/interaction_results__0.*.txt")
        print(txt)
            
        
        data = pd.read_csv(txt[0], delimiter="\t", header=None)
        #print(data)
        ps, rs, f1s, th = data[0], data[1], data[2], data[3]
        inter_res.append((ps,rs, f1s))

        
    write_res(inter_res , auc, dirr, th)


def read_results_best(dirr, method = "satori"):
    
    rep = 3
    auc = []
    inter_res = []
    best_auc = 0
    best_E = dirr + "/E1"
    for i in range(rep):
        
        outdir = dirr + "/E" + str(i+1)
        
        if os.path.exists(outdir):
            scores = open(outdir + "/modelRes_Val_results.txt", "r").readlines()[-1][:-1]
            res = tuple(map(float,scores.split("\t")))
            if best_auc < res[1]:
                best_auc = res[1]
                best_E = outdir
            
    
    print(best_E)

    threshold_val = glob(best_E + "/Interactions_SATORI/interactions_summary_attnLimit-*.txt")[0].split("-")[-1]
    print(best_E, threshold_val, best_E + "/interaction_results__" + threshold_val + ".txt")
    txt = glob(best_E + "/interaction_results__" + threshold_val)
    print(txt)
    
    import shutil
    shutil.copy2(txt[0], dirr + "/best_interaction_results_" +method+ ".txt") 
    shutil.copy2(best_E + "/modelRes_Val_results.txt", dirr + "/best_val_auc.txt") 
    shutil.copy2(best_E + "/modelRes_results.txt", dirr + "/best_auc.txt")         
        
    
    
def combine_res(data_folder, s = "best"):
    
    exps = glob(data_folder + "*")
    final_res = open(data_folder + "/results.csv", "w" ) 
    final_res.writelines( "Experiment,Precision,Recall,F1-Score,Val AUC,Test AUC\n")

    for exp in exps:
        print(exp)
        folder_name = exp.split("/")[-1]
        try:
            fr = open(exp + "/"+ s+"_auc.txt", "r" ) 
            AUC_test = fr.readlines()[1].split('\t')[1]
            fr = open(exp + "/"+ s+"_val_auc.txt", "r" ) 
            AUC_val = fr.readlines()[1].split('\t')[1]
            print(AUC_val, AUC_test)
            df = pd.read_csv(exp + "/"+ s +"_interaction_results_satori.txt", sep='\t', lineterminator='\n',header = None)
            print(df.loc[100].values)
            y = df.loc[100].values           
            final_res.writelines(folder_name +","+ str(y[0]) +","+ str(y[1])+","+str(y[2]) + "," + AUC_val + "," + AUC_test + "\n")
        
        except:
            pass

    

if __name__ == "__main__":
    
    print("Hello")
    
    f = "results/newdata/ctf_80pairs_eq1/"
    exclude = ["entropy_selection", "other_exps"]

    exps = glob(f + "*")
    print(exps)
    
    for exp in exps:
        if os.path.isdir(exp):
            if exclude[0] not in exp and exclude[1] not in exp:
                read_results_best(exp)
        
    combine_res(f)
      
        
        
        
