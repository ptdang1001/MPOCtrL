# -*- coding: utf8 -*

from multiprocessing import Pool, cpu_count
import numpy as np
from numba import njit


@njit
def fill_zeros_with_random_simple(matrix):
    rows, cols = matrix.shape

    # Count zeros
    zero_count = 0
    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] == 0:
                zero_count += 1

    # Generate all random numbers at once
    random_numbers = np.random.random(zero_count)

    # Fill zeros
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] == 0:
                matrix[i, j] = random_numbers[idx]
                idx += 1


@njit
def calculate_imbalance_loss_and_find_lowest_idx(
    samples_variables_np, factors_variables_np
):
    _, n_columns = samples_variables_np.shape
    sum_scale = 200.0 if n_columns > 100 else 50.0 if n_columns > 50 else 10.0
    normalize_data_sum(samples_variables_np, target_sum=sum_scale, by="row")
    """
    Calculate the loss based on element-wise multiplication between rows of samples_variables_np
    and all rows of factors_variables_np, and return the row index of samples_variables_np with the lowest loss.

    Args:
        samples_variables_np (np.ndarray): Matrix of shape (n_samples, n_variables).
        factors_variables_np (np.ndarray): Matrix of shape (n_factors, n_variables).

    Returns:
        tuple: (lowest_loss, row_idx_with_lowest_loss)
    """
    n_samples, n_variables_1 = samples_variables_np.shape
    n_factors, n_variables_2 = factors_variables_np.shape

    # Ensure the number of columns matches
    if n_variables_1 != n_variables_2:
        raise ValueError(
            "samples_variables_np and factors_variables_np must have the same number of columns."
        )

    lowest_loss = np.inf  # Start with an infinitely large loss
    row_idx_with_lowest_loss = -1  # Initialize the row index

    # Loop through each row of samples_variables_np
    for i in range(n_samples):
        current_loss = 0.0
        # Loop through each row of factors_variables_np
        for j in range(n_factors):
            row_sum = 0.0  # Sum of the current element-wise multiplied row
            for k in range(n_variables_1):
                row_sum += samples_variables_np[i, k] * factors_variables_np[j, k]
            current_loss += row_sum  # Accumulate the loss for the current row of samples_variables_np

        # Update the lowest loss and corresponding row index
        if current_loss < 0: current_loss = -current_loss
        if current_loss < lowest_loss:
            lowest_loss = current_loss
            row_idx_with_lowest_loss = i
        #print("current_loss: ", current_loss)
        #print("lowest_loss: ", lowest_loss)
        #print('\n')

    return row_idx_with_lowest_loss

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


class DirectedFactorGraph:
    def __init__(self, factors_variables):
        self.factors_variables = factors_variables
        self._factor_names = factors_variables.index.values
        self._variable_names = factors_variables.columns.values
        self._factors = {}
        self._variables = {}

        self._init_factors()
        self._init_variables()

    def init_1_factor(self, factor):
        idx = np.where(self.factors_variables.loc[factor, :] == 1)[0]
        parent = list(np.take(self._variable_names, idx)) if len(idx) else []
        idx = np.where(self.factors_variables.loc[factor, :] == -1)[0]
        child = list(np.take(self._variable_names, idx)) if len(idx) else []
        return factor, {"parent_variables": parent, "child_variables": child}

    # @pysnooper.snoop()
    def _init_factors(self):
        res = []
        n_processes = min(cpu_count(), len(self._factor_names))
        with Pool(n_processes) as p:
            res.append(p.map(self.init_1_factor, self._factor_names))

        # for factor,parent_child in res:
        for factor, parent_child in res[0]:
            self._factors[factor] = parent_child

    def init_i_variable(self, variable):
        idx = np.where(self.factors_variables[variable] == -1)[0]
        parent = list(np.take(self._factor_names, idx)) if len(idx) else []
        idx = np.where(self.factors_variables[variable] == 1)[0]
        child = list(np.take(self._factor_names, idx)) if len(idx) else []
        return variable, {"parent_factors": parent, "child_factors": child}

    # @pysnooper.snoop()
    def _init_variables(self):
        res = []
        n_processes = min(cpu_count(), len(self._variable_names))
        with Pool(n_processes) as p:
            res.append(p.map(self.init_i_variable, self._variable_names))

        for variable, parent_child in res[0]:
            self._variables[variable] = parent_child
