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
#SBATCH --error=my-job.err  		# error file
#SBATCH --output=my-job.out 		# output log file

source /s/chromatin/a/nobackup/Saira/anaconda3/etc/profile.d/conda.sh
conda activate venv3
python satori.py data/ToyData/NEWDATA/ctf_40pairs_eq modelsparam/all_exps/baseline/cbaseline_entropy_0.005.txt -w 8 --outDir results/newdata/ctf_40pairs_eq/cbaseline_entropy_0.005/E2 --mode test -v -s --numlabels 2 --attrbatchsize 32 --deskload --intseqlimit 5000 --seed 1 --background negative --interactions --interactionanalysis --method FIS --tomtompath /s/chromatin/p/nobackup/Saira/meme/src/tomtom --database create_dataset/subset40.meme --gt_pairs create_dataset/tf_pairs_40.txt