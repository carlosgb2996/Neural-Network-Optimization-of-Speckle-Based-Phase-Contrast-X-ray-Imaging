#!/bin/bash

###### Resource Request Zone for SLURM ##########################################
#
#SBATCH --job-name=computeMetrics         # Job name
#SBATCH -p gpu                            # Partition to use, Default=short (Check partitions and limits in /hpcfs/shared/README/partitions.txt)
#SBATCH -N 1                              # Number of nodes, Default=1
#SBATCH -n 1                              # Number of tasks, recommended for MPI, Default=1
#SBATCH --cpus-per-task=4                 # CPUs per task, recommended for multi-thread, Default=1
#SBATCH --mem=32G                          # Memory in GB per CPU, Default=16G
#SBATCH --gres=gpu:1                      # Request 1 GPU
#SBATCH --time=02:00:00                   # Maximum runtime, Default=10 days
#SBATCH --mail-user=s.naranjob@uniandes.edu.co
#SBATCH --mail-type=ALL                   
#SBATCH -o computeMetrics.o%j             # Output file name
#SBATCH -e computeMetrics_error.e%j       # Error file name
#
########################################################################################

# ################## Module Loading Zone ##############################################
module load anaconda
source activate StrainNet
pip install --upgrade pip
pip install --quiet tqdm scipy matplotlib pandas GPUtil
########################################################################################


###### Execution Zone: Code and Commands to Execute Sequentially ####################
host=/bin/hostname
date=/bin/date

echo "Host: $(hostname)" 
echo "Started at: $(date)"

# Define directory paths
pred_dir="/hpcfs/home/fisica/s.naranjob/StrainNet/StrainNet/output_inference/"
gt_dir="/hpcfs/home/fisica/s.naranjob/StrainNet/Dataset/Test_Data/"
output_dir="/hpcfs/home/fisica/s.naranjob/StrainNet/StrainNet/output_metrics/"

# Create the output directory if it does not exist
mkdir -p "$output_dir"

# Execute the Python script to calculate metrics
python compute_metrics.py --pred_dir "$pred_dir" \
                          --gt_dir "$gt_dir" \
                          --output_csv "$output_dir/metrics_results.csv"

echo -e "Finished executing the metrics calculation script.\n"
########################################################################################