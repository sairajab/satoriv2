#!/bin/bash

#SBATCH --job-name="Arabadopsis"	 # job name
#SBATCH --partition=kestrel-gpu 		 # partition to which job should be submitted
#SBATCH --qos=gpu_medium			  		 # qos type
#SBATCH --nodes=1                 		 # node count
#SBATCH --ntasks=1                		 # total number of tasks across all nodes
#SBATCH --cpus-per-task=7         		 # cpu-cores per task
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:3090:1 # Request 1 GPU (A100 40GB)
#SBATCH --time=69:00:00 				 #  wall time
#SBATCH --error=temp.err  		# error file
#SBATCH --output=temp.out 		# output log file

source /s/chromatin/a/nobackup/Saira/anaconda3/etc/profile.d/conda.sh
conda activate venv3
python satori.py data/Arabidopsis_ChromAccessibility/atAll_m200_s600 modelsparam/all_exps/arabidopsis/arabidopsis.txt -w 8 --outDir results/arabidopsis/sei/final2/ --mode test -v -s --numlabels 36 --method SATORI --attrbatchsize 32 --deskload --tomtompath /s/chromatin/p/nobackup/Saira/meme/src/tomtom --database /s/chromatin/p/nobackup/Saira/motif_databases/ArabidopsisDAPv1.meme --interactions --background shuffle