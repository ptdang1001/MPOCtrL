#!/bin/bash
uv run src/main.py \
    --input_dir_path /your_data_path_to_data \
    --network_dir_path /your_path_to_compounds_reaction_network \
    --output_dir_path /your_path_to_save_results \
    --gene_expression_file_name your_data_file_name \
    --compounds_reactions_file_name your_reaction_data \
    --reactions_genes_file_name your_reactions_genes_data \
    --n_epoch 200
