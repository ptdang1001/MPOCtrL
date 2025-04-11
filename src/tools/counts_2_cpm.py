import os

# import pandas as pd
import fireducks.pandas as pd
import numpy as np
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from tqdm import tqdm
import time
import argparse


def counts_2_cpm_log1p(file_path: str, log1p: bool = False) -> bool:
    """
    Process a CSV file to compute counts per million (CPM) and log1p transformation.
    Args:
        file_path (str): Path to the input CSV file
        log1p (bool): Whether to apply log1p transformation
    """

    print(f"Processing {file_path}...")

    # Step 1: Load the data
    df = None
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path, index_col=0)
    elif file_path.endswith(".csv.gz"):
        df = pd.read_csv(file_path, index_col=0, compression="gzip")
    else:
        raise ValueError("The input file must be a CSV or CSV.GZ file.")

    print(f"Loaded {file_path} with shape {df.shape}")
    print(df.head())

    # Step 2: Replace NaN and inf values with 0
    df = df.replace([np.nan, np.inf, -np.inf], 0)

    # Step 3: merge the rows if they have the same gene name, get the max
    df = df.groupby(df.index).max()

    # Step 4: Perform cpm normalization
    column_sums = df.sum(axis=0)  # Sum of each column
    column_sums = column_sums.replace(0, 1)  # Avoid division by zero
    df = df.div(column_sums, axis=1) * 1e6  # cpm normalization

    # Step 5: Apply log1p transformation
    if log1p:
        df = np.log1p(df)

    # step 6: shuffle the rows
    df = df.sample(frac=1, random_state=42)

    # step 7: shuffle the columns
    df = df.sample(frac=1, axis=1, random_state=42)

    # step 8: save the df in csv.gz format
    save_file = None
    if file_path.endswith(".csv"):
        if log1p:
            save_file = file_path.replace(".csv", "_cpm_log1p.csv.gz")
        else:
            save_file = file_path.replace(".csv", "_cpm.csv.gz")
    elif file_path.endswith(".csv.gz"):
        if log1p:
            save_file = file_path.replace(".csv.gz", "_cpm_log1p.csv.gz")
        else:
            save_file = file_path.replace(".csv.gz", "_cpm.csv.gz")
    else:
        raise ValueError("The input file must be a CSV or CSV.GZ file.")
    df.to_csv(save_file, compression="gzip")

    print(f"Saved {save_file} with shape {df.shape}")
    print(df.head())

    return True


def process_single_file(data_path: str, log1p: bool = False) -> bool:
    """
    Process a single file to compute counts per million (CPM) and log1p transformation.
    Args:
        data_path (str): Path to the input CSV file
        log1p (bool): Whether to apply log1p transformation
    """
    counts_2_cpm_log1p(file_path=data_path, log1p=log1p)

    return True


def process_multiple_files():
    """
    Process all CSV files in a directory in parallel using joblib.
    Args:
        data_dir_path: Directory containing the files
        gene_lengths: Series of gene lengths (optional)
    """
    data_dir_path = "./"
    # Find all CSV files in the directory

    files = [
        os.path.join(data_dir_path, f)
        for f in os.listdir(data_dir_path)
        if f.endswith(".csv") or f.endswith(".csv.gz")
    ]

    # Print summary of files
    print(f"Found {len(files)} files to process in {data_dir_path}")
    for i, f in enumerate(files[:5]):  # Show first 5 files
        print(f"  {i + 1}. {os.path.basename(f)}")
    if len(files) > 5:
        print(f"  ... and {len(files) - 5} more files")

    # Process files in parallel with progress bar

    start_time = time.time()
    """
    
    # Configure parallel processing
    n_jobs = min(len(files), cpu_count() - 1)  # Leave one CPU free
    print(f"Using {n_jobs} parallel jobs for processing {len(files)} files")
    
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(count_cpm_log1p)(file_path, gene_lengths) for file_path in files
    )

    # Calculate success rate
    success_count = sum(results)
    print(f"\nProcessing completed in {time.time() - start_time:.2f} seconds")
    print(f"Successfully processed {success_count} out of {len(files)} files")

    # Check for failures
    if success_count < len(files):
        print(f"WARNING: Failed to process {len(files) - success_count} files")
    """
    for f in tqdm(files):
        print(f"Processing {f}...")
        counts_2_cpm_log1p(os.path.join(data_dir_path, f), log1p=False)
        print(f"{f} processed successfully")
    print(f"Processing completed in {time.time() - start_time:.2f} seconds")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter gene expression data, keeping top dense samples."
    )
    parser.add_argument(
        "--data_name",
        type=str,
        help="Path to the gene expression data file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./",
        help="Directory containing the gene expression data files",
    )
    parser.add_argument(
        "--if_log1p",
        type=bool,
        default=False,
        help="Apply log1p transformation to the data",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data_name = args.data_name
    data_dir = args.data_dir
    if_log1p = args.if_log1p
    data_path = os.path.join(data_dir, data_name)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File {data_path} does not exist.")

    if if_log1p:
        print(f"CPM Processing {data_path} with log1p...")
    else:
        print(f"CPM Processing {data_path} without log1p...")

    process_single_file(file_path=data_path, log1p=if_log1p)


if __name__ == "__main__":
    main()
