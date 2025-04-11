import numpy as np
import fireducks.pandas as pd
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
import os
import argparse
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich.table import Table
import time
import warnings
from kneed import KneeLocator
from joblib import Parallel, delayed
import multiprocessing

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize console
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
        sample_density_bins = [0, 5, 10, 25, 50, 75, 90, 100]
        sample_hist = np.histogram(
            density_percentage, bins=np.array(sample_density_bins)
        )[0]

        console.print("\n[bold blue]Sample Density Distribution:[/bold blue]")
        for i in range(len(sample_hist)):
            pct = (sample_hist[i] / cols) * 100
            console.print(
                f"[green]{sample_density_bins[i]}%-{sample_density_bins[i+1]}%:[/green] {sample_hist[i]:,} samples ({pct:.2f}%)"
            )

    except Exception as e:
        console.print(f"[bold red]Error calculating zero statistics: {str(e)}")


def compute_nmf_for_rank(data_array, rank):
    """
    Compute NMF for a specific rank.

    Parameters:
    -----------
    data_array : numpy.ndarray
        Input data as numpy array
    rank : int
        Rank to use for NMF

    Returns:
    --------
    tuple
        (rank, error, error_message)
    """
    try:
        nmf = NMF(
            n_components=rank,
            init="random",
            random_state=42,
            max_iter=500,
            solver="cd",
            beta_loss="frobenius",
        )
        W = nmf.fit_transform(data_array)
        H = nmf.components_
        reconstructed = W @ H
        error = mean_squared_error(data_array, reconstructed)
        return rank, error, None
    except Exception as e:
        return rank, float("inf"), str(e)


def find_optimal_rank(data, max_rank=50, return_results=True):
    """
    Find the optimal rank for NMF by identifying the elbow point in the reconstruction error curve.
    Uses parallel processing to speed up computation.

    Parameters:
    -----------
    data : pandas.DataFrame
        Input gene expression data (genes x samples)
    max_rank : int, default=50
        Maximum rank to consider
    return_results : bool, default=True
        Whether to return the NMF model and imputed data at the optimal rank

    Returns:
    --------
    dict
        Dictionary containing the optimal rank, error values, and if return_results is True,
        the NMF model and imputed data at the optimal rank
    """
    console.print(
        "[bold blue]Finding optimal NMF rank using parallel processing...[/bold blue]"
    )

    # Ensure max_rank doesn't exceed min(n_samples, n_features)
    max_possible_rank = min(data.shape[0], data.shape[1])
    if max_rank > max_possible_rank:
        console.print(
            f"[yellow]Warning: max_rank ({max_rank}) exceeds data dimensions. Using {max_possible_rank} instead.[/yellow]"
        )
        max_rank = max_possible_rank

    # Range of ranks to try (start from 1)
    ranks = list(range(1, max_rank + 1))

    # Convert to numpy array for speed
    data_array = data.values

    # Determine the number of cores to use (leave one core free)
    num_cores = min(max_rank, multiprocessing.cpu_count() - 1)
    console.print(f"[cyan]Using {num_cores} CPU cores for parallel processing[/cyan]")

    # Run NMF in parallel with improved progress tracking
    all_results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        # Create task with steps for each rank
        task = progress.add_task("[cyan]Fitting NMF models...", total=len(ranks))

        # Use a chunking approach to show incremental progress
        chunk_size = max(1, len(ranks) // 2)  # Show progress in roughly 10 steps
        rank_chunks = [
            ranks[i : i + chunk_size] for i in range(0, len(ranks), chunk_size)
        ]

        for chunk in rank_chunks:
            # Process a chunk of ranks in parallel
            chunk_results = Parallel(n_jobs=num_cores, backend="loky")(
                delayed(compute_nmf_for_rank)(data_array, rank) for rank in chunk
            )
            all_results.extend(chunk_results)

            # Update progress bar after each chunk
            progress.update(task, advance=len(chunk))

            # Show some immediate feedback about what ranks were processed
            min_rank = min(chunk)
            max_rank = max(chunk)
            if min_rank == max_rank:
                progress.console.print(f"    Processed rank {min_rank}")
            else:
                progress.console.print(f"    Processed ranks {min_rank}-{max_rank}")

        # Ensure progress is completed
        progress.update(task, completed=len(ranks))

    # Process results (same as before)
    all_results.sort(key=lambda x: x[0])  # Sort by rank
    reconstruction_errors = [error for _, error, _ in all_results]
    error_messages = {rank: msg for rank, _, msg in all_results if msg is not None}

    # Print any errors that occurred
    for rank, msg in error_messages.items():
        console.print(f"[bold red]Error fitting NMF with rank {rank}: {msg}[/bold red]")

    # Find the elbow point using KneeLocator
    try:
        kneedle = KneeLocator(
            ranks,
            reconstruction_errors,
            curve="convex",
            direction="decreasing",
            S=1.0,
        )
        optimal_rank = kneedle.elbow

        if optimal_rank is None:
            # Try with different parameters if elbow not found
            kneedle = KneeLocator(
                ranks,
                reconstruction_errors,
                curve="convex",
                direction="decreasing",
                S=0.5,
            )
            optimal_rank = kneedle.elbow

    except Exception as e:
        console.print(
            f"[bold yellow]Warning: KneeLocator failed: {str(e)}[/bold yellow]"
        )
        optimal_rank = None

    # If optimal rank not found, use heuristic
    if optimal_rank is None:
        # Find where the rate of change slows down significantly
        diffs = np.diff(reconstruction_errors)
        diffs = np.append(diffs, diffs[-1])  # Pad to match original size
        rel_changes = diffs / np.array(reconstruction_errors)
        optimal_rank_idx = np.argmin(abs(rel_changes - np.median(rel_changes)))
        optimal_rank = ranks[optimal_rank_idx]
        console.print(
            f"[yellow]Could not detect elbow point with KneeLocator. Using alternative method to find rank={optimal_rank}.[/yellow]"
        )

    console.print(f"[green]Optimal rank found: [bold]{optimal_rank}[/bold][/green]")

    # Calculate imputed data at optimal rank if requested
    imputed_data_at_optimal = None
    if return_results:
        console.print(
            f"[cyan]Calculating imputed data with optimal rank {optimal_rank}...[/cyan]"
        )
        try:
            optimal_nmf = NMF(
                n_components=optimal_rank,
                init="random",
                random_state=42,
                max_iter=500,
                solver="cd",
            )
            W_optimal = optimal_nmf.fit_transform(data_array)
            H_optimal = optimal_nmf.components_
            reconstructed_optimal = W_optimal @ H_optimal
            imputed_data_at_optimal = pd.DataFrame(
                reconstructed_optimal, index=data.index, columns=data.columns
            )
        except Exception as e:
            console.print(
                f"[bold red]Error calculating imputed data at optimal rank: {str(e)}[/bold red]"
            )

    # Prepare results
    results_dict = {
        "optimal_rank": optimal_rank,
        "ranks": ranks,
        "errors": reconstruction_errors,
    }

    # Add imputed data if requested
    if return_results and imputed_data_at_optimal is not None:
        results_dict["imputed_data"] = imputed_data_at_optimal

    return results_dict


def impute_genes(data, rank=0, max_rank=50):
    """
    Impute missing values in gene expression data using NMF.

    Parameters:
    -----------
    data : pandas.DataFrame
        Input gene expression data (genes x samples)
    rank : int, default=0
        Specific rank to use for NMF. If 0, optimal rank will be determined automatically
    max_rank : int, default=50
        Maximum rank to consider if rank is 0

    Returns:
    --------
    dict
        Dictionary containing the imputed data, original data, rank used,
        and other useful information
    """
    start_time = time.time()

    console.print("\n[bold blue]Gene Imputation with NMF[/bold blue]")
    console.print(f"Input data shape: {data.shape[0]} samples × {data.shape[1]} genes")

    # Handle missing values if any
    if data.isna().sum().sum() > 0:
        missing_count = data.isna().sum().sum()
        console.print(
            f"[yellow]Input data contains {missing_count} NaN values. These will be replaced with zeros.[/yellow]"
        )
        data = data.fillna(0)

    imputed_data = None
    results = None
    # If rank is specified, directly fit NMF
    if rank > 0:
        console.print(f"Using specified rank: [bold]{rank}[/bold]")

        with console.status("[cyan]Fitting NMF model..."):
            nmf = NMF(
                n_components=rank,
                init="random",
                random_state=42,
                max_iter=500,
                solver="cd",
            )
            W = nmf.fit_transform(data.values)
            H = nmf.components_
            reconstructed = W @ H
            imputed_data = pd.DataFrame(
                reconstructed, index=data.index, columns=data.columns
            )
            results = {
                "rank": rank,
                "imputed_data": imputed_data,
                "reconstruction_error": mean_squared_error(
                    data.values, imputed_data.values
                ),
            }
    else:
        # Find optimal rank and get imputed data
        rank_results = find_optimal_rank(data, max_rank=max_rank, return_results=True)

        if "imputed_data" not in rank_results:
            console.print(
                "[bold red]Failed to calculate imputed data at optimal rank.[/bold red]"
            )
            return None

        results = {
            "rank": rank_results["optimal_rank"],
            "original_data": data,
            "imputed_data": rank_results["imputed_data"],
            "reconstruction_error": rank_results["errors"][
                rank_results["optimal_rank"] - 2
            ],  # -2 because ranks start from 2
        }

    # Calculate metrics
    mse = mean_squared_error(data.values, results["imputed_data"].values)

    # Print summary
    elapsed_time = time.time() - start_time
    console.print(f"\n[bold green]Imputation Complete![/bold green]")
    console.print(f"Rank used: [bold]{results['rank']}[/bold]")
    console.print(f"Mean Squared Error: {mse:.6f}")
    console.print(f"Process completed in {elapsed_time:.2f} seconds")

    # Display summary table
    table = Table(title="Data Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Input Shape", f"{data.shape[0]} genes × {data.shape[1]} samples")
    table.add_row("NMF Rank", str(results["rank"]))
    table.add_row("MSE", f"{mse:.6f}")
    table.add_row("Processing Time", f"{elapsed_time:.2f} seconds")

    console.print(table)

    return results


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Gene imputation using NMF")
    parser.add_argument("--data_dir", type=str, help="Directory containing input data")
    parser.add_argument("--data_name", type=str, help="Input data file")
    parser.add_argument(
        "--rank", type=int, default=0, help="Specific rank to use for NMF"
    )
    parser.add_argument(
        "--max_rank", type=int, default=20, help="Maximum rank to consider"
    )
    return parser.parse_args()


def main():
    """
    Main function to run from command line.
    """
    args = parse_args()

    console.print(f"[bold blue]Gene Imputation Process[/bold blue]")
    console.print(f"Input file: {args.data_dir}")
    console.print(f"Data file: {args.data_name}")
    console.print(f"Rank: {args.rank}")
    console.print(f"Max rank: {args.max_rank}")

    # Load data
    data = None
    data_name_path = os.path.join(args.data_dir, args.data_name)
    if not os.path.exists(data_name_path):
        console.print(
            f"[bold red]Error: File {data_name_path} does not exist.[/bold red]"
        )
        return
    with console.status("[cyan]Loading data..."):
        try:
            if not os.path.exists(data_name_path):
                console.print(
                    f"[bold red]Error: File {data_name_path} does not exist.[/bold red]"
                )
                return
            console.print(f"Loading data from {data_name_path}...")
            # Load data
            if args.data_name.endswith(".csv"):
                data = pd.read_csv(data_name_path, index_col=0)
            elif args.data_name.endswith(".csv.gz"):
                data = pd.read_csv(data_name_path, index_col=0, compression="gzip")
            else:
                console.print(
                    "[bold red]Error: Unsupported file format. Only .csv and .csv.gz are supported.[/bold red]"
                )
                return

            console.print(
                f"Data loaded: {data.shape[0]} genes × {data.shape[1]} samples"
            )
        except Exception as e:
            console.print(f"[bold red]Error loading data: {str(e)}[/bold red]")
            return

    # Print zero statistics
    console.print("[bold blue]Calculating zero statistics...[/bold blue]")
    print_zero_statistics(data)

    # Perform imputation
    results = impute_genes(data.T.copy(), rank=args.rank, max_rank=args.max_rank)
    data_imputed = results.get("imputed_data", None)
    data_imputed = data_imputed.T.copy() if data_imputed is not None else None
    console.print(
        f"[bold blue]Imputation completed with rank {results['rank']}[/bold blue]"
    )
    print_zero_statistics(data_imputed)

    if data_imputed is None:
        console.print(
            "[bold red]Error: Imputation failed. Check the logs for details.[/bold red]"
        )
        return

    if data_imputed is not None:
        # Save results
        save_file_path = ""
        if data_name_path.endswith(".csv"):
            save_file_path = data_name_path.replace(".csv", "_imputed.csv.gz")
        elif data_name_path.endswith(".csv.gz"):
            save_file_path = data_name_path.replace(".csv.gz", "_imputed.csv.gz")
        else:
            save_file_path = data_name_path + "_imputed.csv.gz"
        with console.status("[cyan]Saving imputed data..."):
            try:
                data_imputed.to_csv(save_file_path, compression="gzip")
                console.print(f"[green]Imputed data saved to {save_file_path}[/green]")
            except Exception as e:
                console.print(
                    f"[bold red]Error saving imputed data: {str(e)}[/bold red]"
                )
                return


if __name__ == "__main__":
    main()
