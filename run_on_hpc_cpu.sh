#!/usr/bin/bash

#SBATCH --job-name=MPOCtrL            # Job name
#SBATCH --partition=batch               # Use the GPU partition (modify as needed)
#SBATCH --ntasks=1 # Number of tasks
#SBATCH --cpus-per-task=8 # Number of CPU cores per task (adjust as needed)
#SBATCH --mem=128G                     # Request xGB of memory
#SBATCH --time=02:00:00               # Set a x-hour time limit


# Read the input arguments
input_dir_path=$1
network_dir_path=$2
output_dir_path=$3
gene_expression_file_name=$4
compounds_reactions_file_name=$5
reactions_genes_file_name=$6

python3 src/main.py \
    --input_dir_path ${input_dir_path} \
    --network_dir_path ${network_dir_path} \
    --output_dir_path ${output_dir_path} \
    --gene_expression_file_name ${gene_expression_file_name} \
    --compounds_reactions_file_name ${compounds_reactions_file_name} \
    --reactions_genes_file_name ${reactions_genes_file_name}
