'''
Read meta data file as dict
Read interaction file
Check if there's downloaded file other wise check in mete_data dict and download
Run LOLA analysis
Combine all all_Enrichment files 
'''

import pandas as pd
import subprocess
from ftplib import FTP
from bs4 import BeautifulSoup
import ftputil
from urllib.parse import urlparse
import requests
import re
import os
from glob import glob

def get_data_file_names(ftp_link):
    
    def get_html_content(url):
        try:
            # Make an HTTP GET request to the URL
            response = requests.get(url)

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                # Return the HTML content
                return response.text
            else:
                # Print an error message if the request was not successful
                print("Error: Unable to fetch HTML content. Status Code: {}".format(response.status_code))
                return None

        except requests.RequestException as e:
            # Handle any request exceptions (e.g., connection error)
            print("Error: {}".format(str(e)))
            return None

    # Call the function to get HTML content
    html_content = get_html_content(ftp_link)

    # Print the HTML content if available
    if html_content:

        # Parse the HTML content
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find all 'a' tags within the 'pre' tag
        links = soup.find('pre').find_all('a')
        
        all_files = []

        # Extract and print href values
        for link in links:
            href_value = link.get('href')
            all_files.append(href_value)  
        pattern = r'.*target\.all.*'
        matching_files = [file for file in all_files if re.match(pattern, file)]
    
    return matching_files
 
        
def read_metadata(file_path = "/s/chromatin/a/nobackup/Saira/Downloads/Experiment_level_ChipHub_metadata.xlsx"):
    metadata = {}
    df = pd.read_excel(file_path)
    #df = df.set_index(drop = True)
    print(df.columns)
    
    for ind in df.index:
        if df['Factor'][ind] not in metadata:
            metadata[df['Factor'][ind]] = df['BioProject ID'][ind]
    
    return metadata

def read_interactions(file):
    df = pd.read_csv(file)
    inter_list = []
    for ind in df.index:
        inter_list.append(df.iloc[:,0][ind].split('$\longleftrightarrow$'))
    return inter_list

def get_bed_files(interactions, metadata, output_directory ):
    
    input_dir = "/s/chromatin/o/nobackup/Saira/Arabidopsis/Bed_files/"
    db_path = "/s/chopin/k/grad/saira55/R/x86_64-redhat-linux-gnu-library/4.3/LOLA/extdata/tair10/interacting_tfs/regions/"
    
    universe = "/s/chromatin/o/nobackup/Saira/Arabidopsis/Bed_files/universe/arabidopsis_thaliana.final.annotatePeak"
 
    for i in interactions:
        args = []
        files_in_region = glob(db_path + "*.*")
        for f in files_in_region:
            print("Removing  " , f)
            os.remove(f)
        if i[0] in metadata and i[1] in metadata:
            motif1 = i[0]
            motif2 = i[1]
            url = "https://biobigdata.nju.edu.cn/ChIPHub_download/arabidopsis_thaliana/" + metadata[i[0]] + "/hammock/"
            bed_files = get_data_file_names(url)
            if metadata[i[0]] == metadata[i[1]]:
                for bed_file in bed_files:
                    print(motif1, bed_file)
                    if motif1 in bed_file:
                        if not os.path.exists(input_dir + bed_file[:-3]):
                            subprocess.run(["wget", url+bed_file, "-P", input_dir], check=True)
                            subprocess.run(["gzip" , "-d" , input_dir + bed_file ])
                            print("Download successful!")
                        args.append(input_dir + bed_file)
                            
                    if motif2 in bed_file:
                        if not os.path.exists(db_path + bed_file[:-3]):
                            subprocess.run(["wget", url+bed_file, "-P", db_path], check=True)
                            subprocess.run(["gzip" , "-d" , db_path + bed_file ])
                            print("Download successful!")
            else:
                for bed_file in bed_files:
                    print(motif1, bed_file)
                    if motif1 in bed_file:
                        if not os.path.exists(input_dir + bed_file[:-3]):
                            subprocess.run(["wget", url+bed_file, "-P", input_dir], check=True)
                            subprocess.run(["gzip" , "-d" , input_dir + bed_file ])
                            print("Download successful!")
                        args.append(input_dir + bed_file)
                url = "https://biobigdata.nju.edu.cn/ChIPHub_download/arabidopsis_thaliana/" + metadata[i[1]] + "/hammock/"
                bed_files = get_data_file_names(url)
                for bed_file in bed_files:
                    if motif2 in bed_file:
                        if not os.path.exists(db_path + bed_file[:-3]):
                            subprocess.run(["wget", url+bed_file, "-P", db_path], check=True)
                            subprocess.run(["gzip" , "-d" , db_path + bed_file ])
                            print("Download successful!")
            temp_int = 0
            for arg in args:
                # if "colamp" in arg:
                #     motif1 = motif1 + "_colamp"
                peaks_count = 0
                with open(arg[:-3], "r") as f:
            
                    peaks_count = len(f.readlines())
                result_dir = output_directory+motif1 + "_" + motif2
                
                print(result_dir)
                
                if os.path.exists(result_dir):
                    result_dir  = result_dir + "_" + str(temp_int)
                    temp_int += 1

                subprocess.run(["Rscript", "../LOLA_analysis.R", arg[:-3], universe, result_dir])
                args_file =open(result_dir+"/args.csv", "w")
                args_file.writelines(arg[:-3] + "\t" + str(peaks_count) + "\n")
                args_file.close()
                
                
                
                
        
                    
def combine_sort_results(outdir, exp_dir):
    
    folders = os.listdir(outdir)
    df1 = pd.read_csv(outdir + folders[0] + "/allEnrichments.tsv", delimiter = "\t")
    df1.insert(2, "TF Interaction", [folders[0]] * len(df1.index), True)
    args = pd.read_csv(outdir + folders[0] + "/args.csv", delimiter = "\t")
    for idx in df1.index:
        df1["userSet"][idx] = args.iloc[0,0] 
    df1.insert(1, "input_size", [args.iloc[0,1]] * len(df1.index), True)

    for dirr in folders[1:]:
        df = pd.read_csv(outdir + dirr + "/allEnrichments.tsv", delimiter = "\t")
        args = pd.read_csv(outdir + dirr + "/args.csv", delimiter = "\t")
        
        for idx in df.index:
            df["userSet"][idx] = args.iloc[0,0] 
            
        df.insert(1, "input_size", [args.iloc[0,1]] * len(df1.index), True)
        df.insert(2, "TF Interaction", [dirr] * len(df.index), True)
        
        df1 = pd.concat([df1, df], ignore_index=True)
        
    print(df1.index)    
    df_sorted = df1.sort_values(by='pValueLog', ascending=False)
    
    df_sorted.to_csv(exp_dir + "all_interactions_FIS.csv" )


                
                
                

            #print(result.stdout)
# )
#             wget
#         if i[1] in metadata[i[1]]:
            
            
if __name__ == "__main__":
    metadata = read_metadata()
    interaction_file = "../results/arabidopsis/E1/unique_interactions_FIS.csv"
    inters = read_interactions(interaction_file)
    # #print(metadata)
    outdir = "../data/overlap_analysis_dir/FIS/"
    get_bed_files(inters, metadata , outdir)
    combine_sort_results(outdir, "../results/arabidopsis/E1/")
    
    
    # Replace the link with your actual link
    #ftp_link = "https://biobigdata.nju.edu.cn/ChIPHub_download/arabidopsis_thaliana/SRP045296/hammock/"
    



        

    
    
