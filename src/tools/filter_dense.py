import fireducks.pandas as pd
import numpy as np
import os
import argparse
from rich.console import Console
from rich.progress import Progress

console = Console()


def print_zero_statistics(data: pd.DataFrame):
    """
    Print detailed statistics about zeros and non-zeros in a DataFrame.

    Parameters:
    -----------
    data : pandas.DataFrame
        The input DataFrame to analyze
    console : rich.console.Console
        Rich console for pretty printing
    """
    try:
        # Get shape information
        rows, cols = data.shape
        total_elements = rows * cols

        # Calculate zeros
        non_zero_counts = np.count_nonzero(data.values, axis=0)
        total_non_zeros = np.sum(non_zero_counts)
        total_zeros = total_elements - total_non_zeros

        # Calculate percentages
        zero_percentage = (total_zeros / total_elements) * 100
        non_zero_percentage = 100 - zero_percentage

        # Calculate stats per sample
        sample_density = pd.Series(non_zero_counts, index=data.columns)
        density_percentage = (sample_density / rows) * 100

        # Calculate stats per gene (row)
        gene_non_zeros = np.count_nonzero(data.values, axis=1)
        gene_density = pd.Series(gene_non_zeros, index=data.index)
        gene_density_percentage = (gene_density / cols) * 100

        # Print stats
        console.print("\n[bold blue]Zero Statistics Summary:[/bold blue]")
        console.print(f"[green]Data Shape:[/green] {rows:,} genes × {cols:,} samples")
        console.print(f"[green]Total Elements:[/green] {total_elements:,}")
        console.print(
            f"[green]Zero Elements:[/green] {total_zeros:,} ({zero_percentage:.2f}%)"
        )
        console.print(
            f"[green]Non-zero Elements:[/green] {total_non_zeros:,} ({non_zero_percentage:.2f}%)"
        )

        console.print("\n[bold blue]Sample Statistics:[/bold blue]")
        console.print(
            f"[green]Average non-zeros per sample:[/green] {sample_density.mean():.2f} ({density_percentage.mean():.2f}%)"
        )
        console.print(
            f"[green]Most dense sample:[/green] {sample_density.idxmax()} with {sample_density.max():,} non-zeros ({density_percentage.max():.2f}%)"
        )
        console.print(
            f"[green]Least dense sample:[/green] {sample_density.idxmin()} with {sample_density.min():,} non-zeros ({density_percentage.min():.2f}%)"
        )

        console.print("\n[bold blue]Gene Statistics:[/bold blue]")
        console.print(
            f"[green]Average non-zeros per gene:[/green] {gene_density.mean():.2f} ({gene_density_percentage.mean():.2f}%)"
        )
        console.print(
            f"[green]Most expressed gene:[/green] {gene_density.idxmax()} with {gene_density.max():,} non-zeros ({gene_density_percentage.max():.2f}%)"
        )
        console.print(
            f"[green]Least expressed gene:[/green] {gene_density.idxmin()} with {gene_density.min():,} non-zeros ({gene_density_percentage.min():.2f}%)"
        )

        # Histogram information
        sample_density_bins = [0, 10, 25, 50, 75, 90, 100]
        sample_hist = np.histogram(
            density_percentage, bins=np.array(sample_density_bins)
        )[0]

        console.print("\n[bold blue]Sample Density Distribution:[/bold blue]")
        for i in range(len(sample_hist) - 1):
            pct = (sample_hist[i] / cols) * 100
            console.print(
                f"[green]{sample_density_bins[i]}%-{sample_density_bins[i+1]}%:[/green] {sample_hist[i]:,} samples ({pct:.2f}%)"
            )

    except Exception as e:
        console.print(f"[bold red]Error calculating zero statistics: {str(e)}")


def filter_dense_samples(
    data_dir: str = None,
    data_name: str = None,
    top_percentages: list = None,
    output_dir: str = None,
):
    """
    Process gene expression data by filtering for top dense samples.

    Parameters:
    -----------
    data_dir : str, default=None
        Directory containing the input file
    data_name : str
        Name of the input file (csv, csv.gz, parquet)
    top_percentages : list, default=None
        List of percentages for top dense samples to keep (e.g., 10 means top 10%)
    output_dir : str, default=None
        Directory to save the output files. If None, saves in the same directory as input

    Returns:
    --------
    dict
        Dictionary mapping percentages to paths of saved files
    """
    # Initialize console
    console.print("[bold blue]Starting the filtering process...")
    if not data_name or not data_dir:
        console.print("[bold red]Error: Please provide a valid data file or directory.")
        return {}
    if not top_percentages:
        console.print("[bold red]Error: Please provide a list of top percentages.")
        return {}

    data_name_path = os.path.join(data_dir, data_name)
    # Check if the file exists
    if not os.path.exists(data_name_path):
        console.print(f"[bold red]Error: File not found: {data_name_path}")
        return {}

    # Set output directory
    if output_dir is None:
        output_dir = data_dir

    # read the file in csv, csv.gz, or parquet format
    data = None
    if data_name.endswith(".csv"):
        data = pd.read_csv(data_name_path, index_col=0)
    elif data_name.endswith(".csv.gz"):
        data = pd.read_csv(data_name_path, compression="gzip", index_col=0)
    elif data_name.endswith(".parquet"):
        data = pd.read_parquet(data_name_path)
    else:
        console.print(f"[bold red]Error: Unsupported file format: {data_name}")
        return {}

    console.print(
        f"Data loaded. Shape: {data.shape[0]} genes × {data.shape[1]} samples"
    )

    # Print zero statistics
    print_zero_statistics(data)

    # Calculate non-zero counts for each sample (column)
    console.print("Calculating sample densities...")
    try:
        # Count non-zero entries in each column (sample)
        non_zero_counts = np.count_nonzero(data.values, axis=0)
        total_elements = data.shape[0] * data.shape[1]
        zero_elements = total_elements - np.sum(non_zero_counts)

        # Check if there are any zeros in the data
        if zero_elements == 0:
            console.print(
                "[bold yellow]Warning: No zeros found in the data. All samples have 100% density."
            )
            console.print(
                "[bold yellow]Filtering by density isn't meaningful since all samples are equally dense."
            )
            return {}

        # Create a Series with sample names as index
        sample_density = pd.Series(non_zero_counts, index=data.columns)

        # Sort samples by density in descending order
        sorted_density = sample_density.sort_values(ascending=False)

    except Exception as e:
        console.print(f"[bold red]Error calculating densities: {str(e)}")
        return {}

    data_base_name = ""
    if data_name.endswith(".csv"):
        data_base_name = data_name[:-4]
    elif data_name.endswith(".csv.gz"):
        data_base_name = data_name[:-7]
    elif data_name.endswith(".parquet"):
        data_base_name = data_name[:-8]
    else:
        raise ValueError("Unsupported file format")

    with Progress() as progress:
        task = progress.add_task("Processing...", total=len(top_percentages))

        for percentage in sorted(top_percentages):
            # Calculate how many samples to keep
            samples_to_keep = int(np.ceil(len(data.columns) * percentage / 100))

            # Get the top samples
            top_samples = sorted_density.head(samples_to_keep).index.tolist()

            # Create filtered dataframe
            df_filtered = data[top_samples]

            # print zero statistics for filtered data
            print_zero_statistics(df_filtered)

            # Create a filename with metadata
            genes_count = df_filtered.shape[0]
            samples_count = df_filtered.shape[1]

            new_filename = f"{data_base_name}_dense_genes{genes_count}_samples{samples_count}.csv.gz"
            save_path = os.path.join(output_dir, new_filename)

            # Save the filtered dataframe
            df_filtered.to_csv(save_path, compression="gzip")

            console.print(
                f"Saved top {percentage}% dense ({samples_count} samples) to [green]{os.path.basename(save_path)}[/green]"
            )
            progress.update(task, advance=1)

    console.print("[bold green]Processing completed successfully!")
    return True


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
        help="Directory containing gene expression data files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for filtered data",
    )
    parser.add_argument(
        "--percentages",
        type=str,
        default="5,10,20,30,40,50,60",
        help="Percentages to filter by",
    )
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Check for valid input
    # print current args using rich
    console.print(f"Arguments: {args}")

    # Parse percentages
    try:
        percentages = [int(p.strip()) for p in args.percentages.split(",")]
    except Exception as e:
        # Handle any other exceptions
        console.print(
            f"[bold yellow]Warning: An error occurred: {e}. Using default [5, 10, 20, 30, 40, 50, 60]."
        )
        percentages = [5, 10, 20, 30, 40, 50, 60]

    # filter dense samples
    console.print(f"Filtering dense samples with percentages: {percentages} ...")
    filter_dense_samples(
        data_dir=args.data_dir,
        data_name=args.data_name,
        top_percentages=percentages,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
