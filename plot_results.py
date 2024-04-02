import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import os


def plot_multiple():
        
    # plt.ion()
    fig, ax = plt.subplots()
    trans_offset = mtransforms.offset_copy(ax.transData, fig=fig,
                                        x=0.05, y=0.10, units='inches')

    dpath = "results/"
    out_folders = ["best_hyperparameters_relative/ctf_1/", "best_hyperparameters_relative/ctf_2/", "best_hyperparameters_relative/ctf_3/"] #Interaction_Results/"] #"CNN_No_dropout_e6", "CNN_No_dropout_e7"]
    out_folders.append("best_hyperparameters/CTF_1")
    out_folders.append("best_hyperparameters/CTF_2")
    out_folders.append("best_hyperparameters/CTF_3")

    markers_on = [21, 101, 201]

    linestyles = ['--' , '']

    count = 1
    nh = 0

    for out_f in out_folders:

        print(out_f)

        if nh > 0 :

            for j in range(nh): 

                rs = []
                ps = []

                #with open(os.path.join(dpath, os.path.join(out_f, "interaction_results_" +str(j) +"_0.12.txt")), "r") as f:
                with open(os.path.join(dpath, os.path.join(out_f, "avg_interaction_results.txt")), "r") as f:


                    x = f.readlines()
                    for data in x:
                        print(data)

                        p , r, f,t = data.split("\t")
                        t = float(t.strip())

                        rs.append(float(r))
                        ps.append(float(p))

                plt.plot(rs ,ps , linestyle='--', marker='o', markevery = markers_on ,label="MH" + str(j))
                t =[]
                s = []
                ths = [0.01 ,  0.05, 0.1]
                for m in markers_on:
                        t.append(rs[m])
                        s.append(ps[m])
                i = 0
                for a,b in zip(t, s):
                        plt.text(a, b, str(ths[i]), transform=trans_offset, fontsize=8)
                        i = i + 1
                count = count + 1
        else:
             
                rs = []
                ps = []

                with open(os.path.join(dpath, os.path.join(out_f, "avg_interaction_results.txt")), "r") as f:


                    x = f.readlines()
                    for data in x:
                        print(data)

                        p , r,f, t = data.split("\t")
                        t = float(t.strip())

                        rs.append(float(r))
                        ps.append(float(p))
                if "relative" in out_f:

                        plt.plot(rs ,ps , linestyle='--', marker='o', markevery = markers_on ,label=out_f[-6:-1]+"_relAttn")
                
                else:
                        plt.plot(rs ,ps , linestyle='solid', marker='o', markevery = markers_on ,label=out_f[-5:]+"_Attn")

                t =[]
                s = []
                ths = [0.01 ,  0.05, 0.1]
                for m in markers_on:
                        t.append(rs[m])
                        s.append(ps[m])
                i = 0
                for a,b in zip(t, s):
                        plt.text(a, b, str(ths[i]), transform=trans_offset, fontsize=8)
                        i = i + 1
                count = count + 1
        
            # axis labels

    print("*******************Plotting****************")

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('best hyperparameters')
    # show the legend
    plt.legend()
    # for i in range(len(thresholds)):
    #     if
    # ax.annotate('T = 4K', xy=(2,4), xycoords='data',
    #             xytext=(-100,60), textcoords='offset points',
    #             arrowprops=dict(arrowstyle='fancy',fc='0.6',
    #                             connectionstyle="angle3,angleA=0,angleB=-90"))

    # show the plot
    #plt.show()

    plt.savefig('Plots/best_hyperparameters_comparison.png')


    # plt.plot(rs, ps)
    # plt.ylabel('precision')
    # plt.xlabel('recall')
    #
    # plt.show()
    

def plot(out_f, fs):
        
        # plt.ion()
     fig, ax = plt.subplots()
     trans_offset = mtransforms.offset_copy(ax.transData, fig=fig,
                                        x=0.05, y=0.10, units='inches')
     markers_on = [21, 101, 201] #, 401]

     linestyles = ['--' , '']
        
     for fi in fs:
        rs = []
        ps = []
        

        with open(os.path.join(out_f, fi), "r") as f:


                x = f.readlines()
                for data in x:
                        print(data)

                        p , r,f, t = data.split("\t")
                        t = float(t.strip())

                        rs.append(float(r))
                        ps.append(float(p))

                plt.plot(rs ,ps , linestyle='solid', marker='o', markevery = markers_on ,label=fi)

                t =[]
                s = []
                ths = [0.01 ,  0.05, 0.1] #, 0.2
                for m in markers_on:
                        t.append(rs[m])
                        s.append(ps[m])
                i = 0
                for a,b in zip(t, s):
                        #plt.text(a, b, str(ths[i]), transform=trans_offset, fontsize=8)
                        i = i + 1
                #count = count + 1
        
            # axis labels

     print("*******************Plotting****************")

     plt.xlabel('Recall')
     plt.ylabel('Precision')
     #plt.title('Relative NS')
        # show the legend
     plt.legend()
    # for i in range(len(thresholds)):
    #     if
    # ax.annotate('T = 4K', xy=(2,4), xycoords='data',
    #             xytext=(-100,60), textcoords='offset points',
    #             arrowprops=dict(arrowstyle='fancy',fc='0.6',
    #                             connectionstyle="angle3,angleA=0,angleB=-90"))

    # show the plot
    #plt.show()

     plt.savefig('Plots/data_40.png')
        
                
                
                

    
if __name__ == "__main__" :
        
        #plot("results/newdata/ctf_40pairs_eq/baseline_entropy_0.01/",[ "adam/avg_interaction_results_satori.txt", "rmsprop/avg_interaction_results_satori.txt", "avg_interaction_results_satori.txt"])
        plot("results/newdata/ctf_40pairs_eq1/" , ["baseline/E1/interaction_results__0.txt", "baseline_entropy_0.005/E3interaction_results__0.2715124008113738.txt"])