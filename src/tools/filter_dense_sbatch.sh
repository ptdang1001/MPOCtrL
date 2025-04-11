#!/usr/bin/bash

#SBATCH --job-name=flter_dense            # Job name
#SBATCH --partition=batch               # Use the GPU partition (modify as needed)
#SBATCH --ntasks=1 # Number of tasks
#SBATCH --cpus-per-task=8 # Number of CPU cores per task (adjust as needed)
#SBATCH --mem=32G                     # Request xGB of memory
#SBATCH --time=2:00:00               # Set a x-hour time limit
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xxxxxxx@xxxx.com # your email address
#SBATCH -o ./%j_stdOutput.txt # Standard output
#SBATCH -e ./%j_stdError.txt # Standard error

# Read the input arguments
DATA_DIR="/path/to/data/"       # Replace with your data directory
DATA_NAME="dataset_name.csv.gz" # Replace with your dataset name
PERCENTAGES="5,10,20,30"        # default percentages using comma as separator, later will be converted to a list [5,10,20,30,40,50,60]

python filter_dense.py \
    --data_dir ${DATA_DIR} \
    --data_name ${DATA_NAME} \
    --percentages ${PERCENTAGES}
