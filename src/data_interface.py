# -*-coding:utf-8-*-


import os, json, warnings
from numba import njit
from datetime import datetime


import torch
from torch.utils.data import Dataset
#import pandas as pd
import fireducks.pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# global variables
SEP_SIGN = "*" * 100
warnings.filterwarnings("ignore")


@njit
def mean_std_to_mean_ratio(matrix):
    """
    Compute the mean of column-wise (standard deviation / mean) for the given matrix.

    Args:
        matrix (np.ndarray): Input matrix of shape (n_rows, n_cols).

    Returns:
        float: Mean of the column-wise ratios.
    """
    n_rows, n_cols = matrix.shape
    ratios = np.zeros(n_cols, dtype=np.float64)

    for col in range(n_cols):
        column = matrix[:, col]
        col_mean = np.mean(column)
        if col_mean != 0:  # Avoid division by zero
            col_std = np.std(column)
            ratios[col] = col_std / col_mean
        else:
            ratios[col] = 0.0  # Assign 0 if column mean is 0

    # Normalize the ratios by dividing by their sum
    ratios_sum = np.sum(ratios)
    if ratios_sum != 0:
        ratios /= ratios_sum  # Normalize
    else:
        ratios.fill(0.0)  # If sum is zero, set all normalized ratios to 0

    return np.mean(ratios)


@njit
def normalize_data_norm(matrix, target_norm=10.0, by="row"):
    """
    Normalize each row of a matrix to have a norm of 1.

    Args:
        matrix (np.ndarray): Input matrix of shape (n_rows, n_cols).

    Returns:
        np.ndarray: Matrix with normalized rows.
    """
    n_rows, n_cols = matrix.shape
    for i in range(n_rows):
        row_norm = np.sqrt(
            np.sum(matrix[i, :] ** 2)
        )  # Calculate the L2 norm of the row
        if row_norm != 0:  # Avoid division by zero
            matrix[i, :] /= row_norm
            matrix[i, :] *= target_norm


@njit
def normalize_data_sum(matrix, target_sum=10.0, by="col"):
    """
    Normalize a matrix by columns in-place so that the sum of each column is the target_sum.

    Args:
        matrix (np.ndarray): Input matrix to normalize (modified in-place).
        target_sum (float): Desired sum for each column.
    """
    if by == "row":
        matrix = matrix.T
    # Get the shape of the matrix
    rows, cols = matrix.shape

    # Loop through each column
    for col in range(cols):
        col_sum = np.sum(matrix[:, col])  # Calculate the sum of the column
        if col_sum != 0:  # Avoid division by zero
            scale_factor = target_sum / col_sum
            for row in range(rows):
                matrix[row, col] *= scale_factor
    if by == "row":
        matrix = matrix.T


def init_output_dir_path(args):
    reaction_network_name = args.compounds_reactions_file_name.split("_cmMat.csv")[0]
    print("Reaction Network Name:{0}".format(reaction_network_name))
    print(SEP_SIGN)
    # Get the current timestamp
    current_timestamp = datetime.now()
    # Format the timestamp
    formatted_timestamp = current_timestamp.strftime("%Y%m%d%H%M%S")
    data_file_name = args.gene_expression_file_name.split(".csv")[0]
    folder_name = f"{data_file_name}-{reaction_network_name}-{args.experiment_name}_{formatted_timestamp}"
    output_dir_path = os.path.join(args.output_dir_path, folder_name)
    # if folder already exists, add a number to the folder name
    if os.path.exists(output_dir_path):
        random_number = np.random.randint(1, 999)
        folder_name = (
            f"{data_file_name}-{reaction_network_name}-{args.experiment_name}_"
            f"{formatted_timestamp}_{str(random_number).zfill(3)}"
        )
        output_dir_path = os.path.join(args.output_dir_path, folder_name)
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    return output_dir_path, folder_name


def load_gene_expression(args):
    read_path = os.path.join(args.input_dir_path, args.gene_expression_file_name)
    gene_expression = None

    if read_path.endswith(".csv.gz"):
        gene_expression = pd.read_csv(read_path, index_col=0, compression="gzip")
    elif read_path.endswith(".csv"):
        gene_expression = pd.read_csv(read_path, index_col=0)
    else:
        print("Wrong Gene Expression File Name!")
        return False

    # replace the nan with zero
    gene_expression = gene_expression.fillna(0.0)

    # remove the rows which are all zero
    gene_expression = gene_expression.loc[~(gene_expression == 0).all(axis=1), :]

    # remove the cols which are all zero
    gene_expression = gene_expression.loc[:, ~(gene_expression == 0).all(axis=0)]

    # remove duplicated rows
    gene_expression = gene_expression[~gene_expression.index.duplicated(keep="first")]

    # remove duplicated cols
    gene_expression = gene_expression.loc[
        :, ~gene_expression.columns.duplicated(keep="first")
    ]

    gene_expression = gene_expression.T

    print(SEP_SIGN)
    # choose 5 random row index
    n_rdm = 5
    rdm_row_idx = np.random.choice(gene_expression.index, n_rdm)
    # choose 5 random col index
    rdm_col_idx = np.random.choice(gene_expression.columns, n_rdm)

    # print the gene expression data, the random 5 rows and 5 cols
    print("Gene Expression Data shape:{0}".format(gene_expression.shape))
    print(
        "Gene Expression Data sample:\n{0}".format(
            gene_expression.loc[rdm_row_idx, rdm_col_idx]
        )
    )

    print(SEP_SIGN)

    return gene_expression


def load_compounds_reactions(args):
    read_path = os.path.join(args.network_dir_path, args.compounds_reactions_file_name)
    compounds_reactions = pd.read_csv(read_path, index_col=0)

    compounds_reactions.index = compounds_reactions.index.map(lambda x: str(x))
    compounds_reactions = compounds_reactions.astype(int)

    """
    print(SEP_SIGN)
    print("\nCompounds:{0}\n".format(compounds_reactions.index.values))
    print("\nReactions:{0}\n".format(compounds_reactions.columns.values))
    print("\nCompounds_Reactions shape:{0}\n".format(compounds_reactions.shape))
    print("\n compounds_Reactions sample:\n {0} \n".format(compounds_reactions))
    print(SEP_SIGN)
    """
    return compounds_reactions


def load_reactions_genes(args):
    read_path = os.path.join(args.network_dir_path, args.reactions_genes_file_name)
    # Opening JSON file
    f = open(read_path)
    # returns JSON object as
    # a dictionary
    reactions_genes = json.load(f)
    # Closing file
    f.close()

    """
    print(SEP_SIGN)
    print("\n Reactions and contained genes:\n {0} \n".format(reactions_genes))
    print(SEP_SIGN)
    """

    return reactions_genes


def remove_allZero_rowAndCol(factors_nodes):
    # remove all zero rows and columns
    factors_nodes = factors_nodes.loc[~(factors_nodes == 0).all(axis=1), :]
    factors_nodes = factors_nodes.loc[:, ~(factors_nodes == 0).all(axis=0)]
    return factors_nodes


def remove_margin_compounds(factors_nodes):
    n_factors, _ = factors_nodes.shape
    keep_idx = []
    print(SEP_SIGN)
    for i in range(n_factors):
        if (factors_nodes.iloc[i, :] >= 0).all() or (
            factors_nodes.iloc[i, :] <= 0
        ).all():
            # print("Remove Compound:{0}".format(factors_nodes.index.values[i]))
            continue
        else:
            keep_idx.append(i)
    factors_nodes = factors_nodes.iloc[keep_idx, :]
    factors_nodes = remove_allZero_rowAndCol(factors_nodes)
    # print(SEP_SIGN)
    # print(SEP_SIGN)
    # print("\n compounds_modules sample:\n {0} \n".format(factors_nodes))
    # print(SEP_SIGN)
    return factors_nodes


def get_data_with_intersection_gene(gene_expression, reactions_genes):
    all_genes_in_gene_expression = set(gene_expression.columns.values.tolist())
    all_genes_in_reactions = []
    for _, genes in reactions_genes.items():
        all_genes_in_reactions.extend(genes)
    all_genes_in_reactions = set(all_genes_in_reactions)

    intersection_genes = all_genes_in_gene_expression.intersection(
        all_genes_in_reactions
    )

    if len(intersection_genes) == 0:
        return None, None

    reactions_genes_new = {}
    print("Current Reaction - Intersection Genes")
    for reaction_i, genes in reactions_genes.items():
        cur_genes_intersection = None
        cur_genes_intersection = set(genes).intersection(intersection_genes)
        print(f"{reaction_i} - {list(cur_genes_intersection)}")
        if len(cur_genes_intersection) != 0:
            reactions_genes_new[reaction_i] = list(cur_genes_intersection)
        else:
            reactions_genes_new[reaction_i] = None

    return gene_expression[list(intersection_genes)], reactions_genes_new


def data_pre_processing(gene_expression, reactions_genes, compounds_reactions_df):
    # for the compounds_reactions adj matrix
    # remove outside compounds and reactions
    compounds_reactions_df = remove_margin_compounds(compounds_reactions_df)
    # remove the all zero rows
    # remove the all zero columns
    compounds_reactions_df = remove_allZero_rowAndCol(compounds_reactions_df)

    # get the intersection reactions bewteen reactions genes and compounds reactions
    reactions_genes = {
        reaction_i: reactions_genes[reaction_i]
        for reaction_i in compounds_reactions_df.columns
    }
    # get the data with intersection genes
    gene_expression, reactions_genes = get_data_with_intersection_gene(
        gene_expression, reactions_genes
    )

    # if there is no intersection genes, just return
    if gene_expression is None:
        # print("\n No Intersection of Genes between Data and Reactions! \n")
        return None, None, None

    return gene_expression, reactions_genes, compounds_reactions_df


def min_max_normalize(df):
    #log_df = np.log1p(df)
    min_vals = df.min(axis=0)
    max_vals = df.max(axis=0)
    normalized_df = (df - min_vals) / (max_vals - min_vals + 1e-8)
    return normalized_df


def z_score_normalization(data):
    """Apply Z-score normalization to each column."""
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)

    # Avoid division by zero for constant columns
    stds[stds == 0] = 1
    normalized_data = (data - means) / stds
    return normalized_data


@njit
def fill_zeros_with_mean(data):
    """Fill zero values with the mean of non-zero values in each column."""
    for i in range(data.shape[1]):
        col = data[:, i]
        non_zero_values = col[col != 0]

        if non_zero_values.size > 0:
            mean_value = non_zero_values.mean()
            for j in range(col.size):
                if col[j] == 0:
                    col[j] = mean_value
    return data


def normalize_gene_expression(gene_expression, reactions_genes):
    n_samples = gene_expression.shape[0]
    reactions_geneExpressionMean = {}
    reactions_gene_expression_normalized = {}

    for reaction, genes in reactions_genes.items():
        cur_data = None
        if genes is not None:
            cur_data = gene_expression.loc[:, genes].values
            cur_data = min_max_normalize(cur_data)
            reactions_gene_expression_normalized[reaction] = cur_data
        else:
            reactions_gene_expression_normalized[reaction] = cur_data
    return reactions_gene_expression_normalized


class CombinedDataset(Dataset):
    def __init__(self, reactions_x_y):
        """
        Initialize the dataset with a dictionary of reactions and their corresponding X, Y data.

        Args:
            reactions_x_y (dict): A dictionary where keys are reaction identifiers and values are tuples (x, y).
                                  x and y should be numpy arrays. x is a matrix, y is a vector.
        """
        # Precompute tensors and store them in a dictionary
        self.reactions_x_y = {
            reaction_i: (
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
            )
            for reaction_i, (x, y) in reactions_x_y.items()
        }

        # Use the first reaction to determine the number of samples
        self.n_samples = next(iter(self.reactions_x_y.values()))[0].shape[0]

    def __len__(self):
        # Assuming all datasets have the same length
        return self.n_samples

    def __getitem__(self, idx):
        """
        Get the batch for the given index, organized as {reaction_i: [x, y]}.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary where each key is a reaction identifier, and the value is a list [x, y].
        """
        return {
            reaction_i: {"X": x[idx], "Y": y[idx]}
            for reaction_i, (x, y) in self.reactions_x_y.items()
        }


def split_data(
    reactions_normalizedNpData_dict, samples_reactions_df, flag, test_size=0.2
):
    train_data = {}
    test_data = {}
    if flag == "train_val":
        for reaction_i, data_np in reactions_normalizedNpData_dict.items():
            if data_np is None:
                continue
            y = samples_reactions_df[reaction_i].values
            X_train, X_test, y_train, y_test = train_test_split(
                data_np, y, test_size=test_size, random_state=42
            )
            train_data[reaction_i] = [X_train, y_train]
            test_data[reaction_i] = [X_test, y_test]
    elif flag == "predict":
        for reaction_i, data_np in reactions_normalizedNpData_dict.items():
            if data_np is None:
                continue
            y = samples_reactions_df[reaction_i].values
            train_data[reaction_i] = [data_np, y]
    return train_data, test_data


def prepare_dataloader_mpo(reactions_X_Y, Y):
    # there is a dummy y in reactions_trainData, use the y to replace it
    for reaction_i, (X, _) in reactions_X_Y.items():
        reactions_X_Y[reaction_i] = [X, Y[reaction_i].values]
    return reactions_X_Y
