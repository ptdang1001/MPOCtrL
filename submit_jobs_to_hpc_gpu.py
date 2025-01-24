# *- coding: utf-8 -*-
"""
Suppoese you are submitting jobs to HPC managed by SLURM, you can use this script to submit multiple jobs to HPC.
"""

import os
from datetime import datetime


# create output directory, if not exist create it
def create_output_dir(output_dir_path, data_name, network_name):
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    if data_name.endswith(".csv"):
        data_name = data_name.replace(".csv", "")
    elif data_name.endswith(".csv.gz"):
        data_name = data_name.replace(".csv.gz", "")
    network_name = network_name.replace("_cmMat.csv", "")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    new_output_dir_path = os.path.join(
        output_dir_path, f"{data_name}-{network_name}-Flux-{timestamp}"
    )
    if not os.path.exists(new_output_dir_path):
        os.makedirs(new_output_dir_path)
    return new_output_dir_path


def main():
    # define your data path
    input_dir_path = "/your/data/path"  # eg. /home/username/data

    # define your network path
    network_dir_path = "/your/network/path"  # eg. /home/username/network

    # define your save path
    output_dir_path = "/your/save/path"  # eg. /home/username/results

    # common network and gene list, customize based on your cases
    network_list = ["GGSL_V3_cmMat.csv", "M171_V3_connected_cmMat.csv"]
    network_gene_list = [
        "GGSL_V3_reactions_genes.json",
        "M171_V3_connected_reactions_genes.json",
    ]

    # data list, customize based on your data
    data_list = [
        "PAAD_data_mrna_seq_v2_rsem_log1p.csv.gz",
        "TCGA_PAAD_log1p.csv.gz",
    ]

    email_address = "xxxxxx@xxx.xx"  # your email address, will receive the job status

    # submit you data jobs to HPC - GPU
    for data in data_list:
        for network, network_gene in zip(network_list, network_gene_list):
            cur_output_dir_path = None
            cur_output_dir_path = create_output_dir(output_dir_path, data, network)
            command = (
                f"sbatch --mail-user={email_address} --mail-type=ALL "
                f"--output={cur_output_dir_path}/std_output.log --error={cur_output_dir_path}/std_error.log "
                f"run_on_hpc_gpu.sh {input_dir_path} {network_dir_path} {cur_output_dir_path} "
                f"{data} {network} {network_gene}"
            )
            print(f"Submitting job with command:\n{command}")
            os.system(command)
            print("\n")


if __name__ == "__main__":
    main()
