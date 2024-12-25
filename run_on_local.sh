#!/bin/bash

python3 src/main.py \
    --input_dir_path /your_data_path_to_data \ # eg. /xxxx/xxxx/data
    --network_dir_path /your_path_to_compounds_reaction_network \ # eg. /xxxx/xxxx/network
    --output_dir_path /your_path_to_save_results \ # eg. /xxxx/xxxx/results
    --gene_expression_file_name your_data_file_name \ # eg. xxx.csv.gz, rows=genes, columns=samples
    --compounds_reactions_file_name your_reaction_data \ # eg.GGSL_V3_cmMat.csv, an adjacency matrix, rows=compounds, columns=reactions
    --reactions_genes_file_name your_reactions_genes_data \ # eg. GGSL_V3_modules_genes.json, a json file, keys=reactions, values=genes
    --n_epoch 200 # number of epochs
