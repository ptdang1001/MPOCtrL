import os, json

# import pysnooper
import numpy as np
import pandas as pd
from numba import njit
import matplotlib.pyplot as plt


@njit
def merge_matrix(matrix_mpoctrl, matrix_mpo):
    """
    Merge two 2D numpy arrays:
    - If one value is zero, keep the non-zero value.
    - If both are non-zero, take their mean.
    - If both are zero, keep zero.
    """
    # Ensure arrays are of the same shape
    assert matrix_mpoctrl.shape == matrix_mpo.shape, "Arrays must have the same shape"

    rows, cols = matrix_mpoctrl.shape
    merged_array = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            if matrix_mpoctrl[i, j] == 0 and matrix_mpo[i, j] != 0:
                merged_array[i, j] = matrix_mpo[i, j]
            elif matrix_mpo[i, j] == 0 and matrix_mpoctrl[i, j] != 0:
                merged_array[i, j] = matrix_mpoctrl[i, j]
            elif matrix_mpoctrl[i, j] != 0 and matrix_mpo[i, j] != 0:
                merged_array[i, j] = matrix_mpoctrl[i, j]*0.8 + matrix_mpo[i,j]*0.2

    return merged_array

def compute_matabolite(samples_reactions_df, compounds_reactions_df):
    """
    Compute the matabolites remaining in the samples after the reactions are performed.
    """
    # Convert DataFrames to NumPy arrays for faster computation
    mat1 = samples_reactions_df.to_numpy()
    mat2 = compounds_reactions_df.to_numpy()

    # Compute the new matrix
    # Broadcasting mat1[:, None, :] to multiply each row of mat1 with all rows of mat2
    new_matrix = np.einsum("ij,kj->ik", mat1, mat2)

    # Convert the result back to a DataFrame
    result_df = pd.DataFrame(
        new_matrix,
        index=samples_reactions_df.index,
        columns=compounds_reactions_df.index,
    )
    result_df = -result_df
    return result_df


def save_dict_to_json(dict, json_file_path):

    def ndarray_to_list(ndarray):
        if isinstance(ndarray, np.ndarray):
            return ndarray.tolist()
        return ndarray

    for key, value in dict.items():
        dict[key] = ndarray_to_list(value)

    # save the dict to a json file in a formatted way
    with open(json_file_path, "w") as json_file:
        json.dump(dict, json_file, indent=4)


def normalize_vector(vector):
    """
    Normalize a vector to ensure its sum is 1.
    """
    total = np.sum(vector)
    if total == 0:
        return vector
    return vector / total


def average_model_weights(state_dict1, state_dict2):
    """
    average the weights of two models.
    Args:
        state_dict1 (dict): model 1 state_dict。
        state_dict2 (dict): model 2 state_dict。
    Returns:
        dict: state_dict of the averaged model.
    """
    averaged_state_dict = {}
    for key in state_dict1.keys():
        if key in state_dict2:  # if the key is in the state_dict2
            averaged_state_dict[key] = state_dict1[key] * 0.7 + state_dict2[key] * 0.3
        else:
            averaged_state_dict[key] = state_dict1[key]
    return averaged_state_dict


def filter_by_indices(loss_dict_mpo, loss_dict_npmpo, key="train_totalLoss_list", n=5):
    """
    Retain only the values in all vectors corresponding to the indices
    that are not among the largest or smallest n values of a specified vector.

    Parameters:
    - loss_dict: dict, key is the name of the loss vector, value is a numpy array.
    - key: str, the key of the vector from which to determine indices (default is "train_totalLoss_list").
    - n: int, the number of largest and smallest values to exclude (default is 5).

    Returns:
    - filtered_loss_dict: dict, the filtered loss dictionary with only selected indices retained.
    - retained_indices: list, the indices that were retained.
    """
    if key not in loss_dict_mpo:
        raise ValueError(f"Key '{key}' not found in loss_dict.")

    # Extract the target vector
    target_vector = loss_dict_mpo[key]
    if len(target_vector) <= 2 * n:
        raise ValueError(
            f"Vector '{key}' must have more than {2 * n} elements to filter."
        )

    # Find indices of the largest n and smallest n values
    largest_indices = np.argsort(target_vector)[-n:]  # Indices of the largest n values
    smallest_indices = np.argsort(target_vector)[:n]  # Indices of the smallest n values

    # Determine the indices to retain
    all_indices = set(range(len(target_vector)))
    indices_to_exclude = set(largest_indices).union(smallest_indices)
    retained_indices = all_indices - indices_to_exclude
    if 0 not in retained_indices:
        retained_indices.add(0)
    if 1 not in retained_indices:
        retained_indices.add(1)
    retained_indices = sorted(retained_indices)

    # Create the filtered dictionary
    filtered_loss_dict_mpo = {}
    for k, vector in loss_dict_mpo.items():
        filtered_loss_dict_mpo[k] = vector[retained_indices]
    filtered_loss_dict_nompo = {}
    for k, vector in loss_dict_npmpo.items():
        filtered_loss_dict_nompo[k] = vector[retained_indices]

    return filtered_loss_dict_mpo, filtered_loss_dict_nompo


def plot_loss(loss_dict, output_dir_path):
    """
    Plots a 2x5 grid of subplots showing training and validation curves based on provided keys.

    Parameters:
        data_dict (dict): A dictionary with the following keys:
                          [
                              "train_totalLoss_list", "train_imbalanceLoss_list",
                              "train_cvLoss_list", "train_sampleCorLoss_list",
                              "train_reactionCorLoss_list",
                              "val_totalLoss_list", "val_imbalanceLoss_list",
                              "val_cvLoss_list", "val_sampleCorLoss_list",
                              "val_reactionCorLoss_list"
                          ]
                          Each key holds a NumPy array of values.
    """
    loss_dict, _ = filter_by_indices(loss_dict, loss_dict)
    # Define mapping of subplot titles to keys
    keys_map = [
        ("train_totalLoss_list", "val_totalLoss_list", "Total Loss"),
        ("train_imbalanceLoss_list", "val_imbalanceLoss_list", "Imbalance Loss"),
        (
            "train_reactionCor_list",
            "val_reactionCor_list",
            "Reaction Correlation",
        ),
        (
            "train_sampleCor_list",
            "val_sampleCor_list",
            "Sample Correlation",
        ),
        ("train_cv_list", "val_cv_list", "CV"),
    ]

    # Create a 1x5 grid
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle("MPOCtrL Training and Validation Losses", fontsize=16)

    for i, (train_key, val_key, title) in enumerate(keys_map):
        # Determine row and column for subplot
        ax = axes[i]

        # Extract training and validation data
        train_values = loss_dict[train_key]
        val_values = loss_dict[val_key]
        if "Cor" in train_key or "cv" in train_key:
            train_values = np.array(train_values)
            val_values = np.array(val_values)
        else:
            train_values = normalize_vector(train_values)
            val_values = normalize_vector(val_values)

        # Plot the curves
        ax.plot(train_values, label="Train", color="blue")
        ax.plot(val_values, label="Validation", color="orange")

        min_val = min(np.min(train_values), np.min(val_values))
        max_val = max(np.max(train_values), np.max(val_values))
        ax.set_ylim(min_val - 0.1 * abs(min_val), max_val + 0.1 * abs(max_val))

        # Add titles and labels
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss Value" if i == 0 else "")
        ax.grid(True)
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.show()
    # save the plot
    save_path = os.path.join(output_dir_path, "loss_curves.png")
    plt.savefig(save_path)
    plt.close()
