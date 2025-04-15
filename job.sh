#!/bin/bash
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --job-name="SATORI"	 # job name
#SBATCH --partition=peregrine-gpu 		 # partition to which job should be submitted
#SBATCH --qos=gpu_medium			  		 # qos type
#SBATCH --nodes=1                 		 # node count
#SBATCH --ntasks=1                		 # total number of tasks across all nodes
#SBATCH --cpus-per-task=7         		 # cpu-cores per task
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:nvidia_a100_3g.39gb:1# Request 1 GPU (A100 40GB)
#SBATCH --time=69:00:00 				 #  wall time

# Add your commands here
source /s/chromatin/a/nobackup/Saira/anaconda3/etc/profile.d/conda.sh
conda activate venv3
python run_experiments_ent.py
# End of script

