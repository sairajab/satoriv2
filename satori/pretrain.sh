#!/bin/bash

#SBATCH --job-name="Arabadopsis"	 # job name
#SBATCH --partition=kestrel-gpu 		 # partition to which job should be submitted
#SBATCH --qos=gpu_medium			  		 # qos type
#SBATCH --nodes=1                 		 # node count
#SBATCH --ntasks=1                		 # total number of tasks across all nodes
#SBATCH --cpus-per-task=7         		 # cpu-cores per task
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:3090:1 # Request 1 GPU (A100 40GB)
#SBATCH --time=69:00:00 
#SBATCH --error=my-job.err  		# error file
#SBATCH --output=my-job.out 		# output log file
source /s/chromatin/a/nobackup/Saira/anaconda3/etc/profile.d/conda.sh
conda activate venv3
python build_pretrain.py
