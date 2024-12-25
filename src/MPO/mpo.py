# -*-coding:utf-8-*-

import pandas as pd
from multiprocessing import Pool, cpu_count

# my libs
from MPO.utils.data_interface import DirectedFactorGraph
from MPO.utils.data_interface import calculate_imbalance_loss_and_find_lowest_idx
from MPO.utils.data_interface import normalize_data_sum
from MPO.utils.data_interface import fill_zeros_with_random_simple
from MPO.utils.model_interface import MessagePassingOptimization


def get_one_sample_flux(
    sample_name, factors_variables_df, variable_old, factors, variables, main_branch, args
):
    mpo = MessagePassingOptimization(variable_old.copy(), factors, variables, main_branch, args)

    mpo.run()

    epochs_variables_mpo_np = pd.DataFrame.from_dict(mpo._variables_dict_new, orient="index").to_numpy()
    epoch_with_lowest_loss=calculate_imbalance_loss_and_find_lowest_idx(epochs_variables_mpo_np, factors_variables_df.to_numpy())

    return sample_name, epochs_variables_mpo_np[epoch_with_lowest_loss]


# @pysnooper.snoop()
def run_mpo(factors_variables_df, samples_variables_df, main_branch, args):
    directed_factor_graph = DirectedFactorGraph(factors_variables_df)  # This is a bipartite directed factor graph
    variables_list = factors_variables_df.columns.tolist()
    samples_variables_df=samples_variables_df[variables_list]

    # Use itertuples to iterate over the DataFrame
    tasks = [
        (
            row[0],  # sample name
            factors_variables_df,  # compounds reactions adj matrix
            dict(zip(variables_list,row[1:])),
            directed_factor_graph._factors,
            directed_factor_graph._variables,
            main_branch,
            args,
        )
        for row in samples_variables_df.itertuples(index=True, name=None)
    ]
    # get current number of processes
    n_processes = min(cpu_count(), samples_variables_df.shape[0])
    results_list = []
    with Pool(n_processes) as pool:
        # Use starmap to process all rows in parallel
        results_list = pool.starmap(get_one_sample_flux, tasks)

    samples_variables_mpo_df = {
        sample_name: variables_predicted for sample_name, variables_predicted in results_list
    }

    samples_variables_mpo_df = pd.DataFrame.from_dict(samples_variables_mpo_df, orient="index")
    samples_variables_mpo_df.columns = variables_list
    sample_names = samples_variables_df.index
    samples_variables_mpo_df = samples_variables_mpo_df.loc[sample_names, :]

    return samples_variables_mpo_df


# run mpo
def mpo(factors_variables_df, samples_variables_df, main_branch, args):
    _,n_variables = samples_variables_df.shape
    fill_zeros_with_random_simple(samples_variables_df.values)

    sum_scale = 200.0 if n_variables > 100 else 50.0 if n_variables > 50 else 20.0
    normalize_data_sum(samples_variables_df.values,target_sum=sum_scale,by='row')

    samples_variables_mpo_df = None
    samples_variables_mpo_df = run_mpo(
        factors_variables_df.copy(), samples_variables_df.copy(), main_branch, args
    )

    samples_variables_mpo_df = samples_variables_mpo_df.abs()
    normalize_data_sum(samples_variables_mpo_df.values,target_sum=sum_scale,by='row')

    return samples_variables_mpo_df
