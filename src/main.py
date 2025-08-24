# -*-coding:utf-8-*-

# built-in library
import os, argparse, warnings, shutil

# Third-party library
#import pandas as pd
import fireducks.pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from multiprocessing import cpu_count

# my library
from data_interface import load_gene_expression
from data_interface import load_compounds_reactions
from data_interface import load_reactions_genes
from data_interface import data_pre_processing
from data_interface import normalize_gene_expression
from data_interface import split_data
from data_interface import prepare_dataloader_mpo
from data_interface import CombinedDataset


from model_interface import AdaptiveModel
from model_interface import AdaptiveMultipleModels

from MPO.mpo import mpo

from utils import plot_loss
from utils import average_model_weights
from utils import save_dict_to_json
from utils import compute_matabolite
from utils import merge_matrix


# global variables
SEP_SIGN = "*" * 100
# set the device to choose GPU or CPU
DEVICE = "auto" if torch.cuda.is_available() else "cpu"
# ignore some warning messages
warnings.filterwarnings("ignore")
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
# set the global random seed to make the results reproducible
# set the random seed to 42
L.seed_everything(42)  # set the random seed to 42


# @pysnooper.snoop()
def train(
    args,
    reactions_normalizedNpData_dict,
    compounds_reactions_df,
    reactions_genes_dict,
    n_samples,
    n_reactions,
):
    # set the random seed
    #L.seed_everything(args.seed)
    reactions_list = compounds_reactions_df.columns.tolist()
    reactions_geneLength = {
        reaction_i: (
            (
                1
                if len(reactions_genes_dict[reaction_i]) == 1
                else len(reactions_genes_dict[reaction_i])
            )
            if reactions_genes_dict[reaction_i]
            else 0
        )
        for reaction_i in reactions_list
    }

    # define a group of models, # of models = # of reactions
    models = {
        reaction_i: (AdaptiveModel(input_dim=n_genes) if n_genes > 0 else None)
        for reaction_i, n_genes in reactions_geneLength.items()
    }

    # define the multi model container
    multi_model_container = AdaptiveMultipleModels(
        models,
        compounds_reactions_df.values,
        reactions_list,
        "train_model",
    )
    # define the multi model container for NN-MPO
    multi_model_container_mpoctrl = AdaptiveMultipleModels(
        models,
        compounds_reactions_df.values,
        reactions_list,
        "train_model_mpo",
    )

    # split the data into train and validation
    # y_dummy is a dummy variable as a place hoder, it is not used in the training
    y_dummy = pd.DataFrame(np.zeros((n_samples, n_reactions)), columns=reactions_list)
    train_dataset, val_dataset = split_data(
        reactions_normalizedNpData_dict, y_dummy, "train_val", test_size=0.3
    )
    train_dataloader = CombinedDataset(train_dataset)
    val_dataloader = CombinedDataset(val_dataset)

    train_dataloader = DataLoader(
        train_dataloader,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataloader,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )

    # create the folder to save the model
    model_save_path = os.path.join(args.output_dir_path, "model_training_log")
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # monitor the validation loss and save the best model
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_save_path,
        monitor="val_total_loss",
        filename="best-model-{epoch:03d}-{val_total_loss:.3f}",
        save_top_k=1,
        mode="min",
    )

    checkpoint_callback_mpo = None
    trainer_mpo = None

    for i in tqdm(range(args.n_epoch // 10 + 1)):
        # define the multi model trainer
        trainer = Trainer(
            default_root_dir=model_save_path,
            max_epochs=10,
            accelerator=DEVICE,
            devices="auto",
            strategy="auto",
            enable_progress_bar=False,
            logger=False,
            enable_model_summary=False,
            callbacks=[
                checkpoint_callback,
            ],
        )
        multi_model_container.train()
        if i == 0:
            # train the model
            trainer.fit(multi_model_container, train_dataloader, val_dataloader)
        else:
            # load the weighted model from the best model of multi_model_container and multi_model_container_mpoctrl
            model_weights = torch.load(checkpoint_callback.best_model_path)[
                "state_dict"
            ]
            model_weights_mpo = torch.load(checkpoint_callback_mpo.best_model_path)[
                "state_dict"
            ]
            averaged_weights = average_model_weights(model_weights, model_weights_mpo)
            multi_model_container.load_state_dict(averaged_weights)
            trainer.fit(multi_model_container, train_dataloader, val_dataloader)

        # """
        # get the train data loader outputs from the trained model_mpo
        multi_model_container.eval()
        trainer.test(multi_model_container, train_dataloader)
        samples_reactions_train = multi_model_container.predictions
        multi_model_container.clear_cache()
        samples_reactions_train_df = pd.DataFrame(
            samples_reactions_train, columns=reactions_list
        )
        # train the model_mpo
        samples_reactions_train_df = mpo(
            compounds_reactions_df, samples_reactions_train_df, [], args
        )

        # get the val data loader outputs from the trained model_1
        trainer.test(multi_model_container, val_dataloader)
        samples_reactions_val = multi_model_container.predictions
        multi_model_container.clear_cache()
        samples_reactions_val_df = pd.DataFrame(
            samples_reactions_val, columns=reactions_list
        )
        samples_reactions_val_df = mpo(
            compounds_reactions_df, samples_reactions_val_df, [], args
        )

        # prepaer data loader mpo for model_mpo
        train_dataloader_mpo = prepare_dataloader_mpo(
            train_dataset, samples_reactions_train_df
        )
        val_dataloader_mpo = prepare_dataloader_mpo(
            val_dataset, samples_reactions_val_df
        )
        train_dataloader_mpo = CombinedDataset(train_dataloader_mpo)
        val_dataloader_mpo = CombinedDataset(val_dataloader_mpo)
        train_dataloader_mpo = DataLoader(
            train_dataloader_mpo,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
        )
        val_dataloader_mpo = DataLoader(
            val_dataloader_mpo,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
        )

        model_save_path_mpo = os.path.join(
            args.output_dir_path, "model_training_log_mpo"
        )
        if not os.path.exists(model_save_path_mpo):
            os.makedirs(model_save_path_mpo)
        else:
            # delete the best model in checkpoint_callback_2
            shutil.rmtree(model_save_path_mpo)
            os.mkdir(model_save_path_mpo)

        checkpoint_callback_mpo = ModelCheckpoint(
            dirpath=model_save_path_mpo,
            monitor="val_total_loss",
            filename="best-model-{epoch:03d}-{val_total_loss:.3f}",
            save_top_k=1,
            mode="min",
        )

        trainer_mpo = Trainer(
            default_root_dir=model_save_path_mpo,
            max_epochs=10,
            accelerator=DEVICE,
            devices="auto",
            strategy="auto",
            enable_progress_bar=False,
            logger=False,
            enable_model_summary=False,
            callbacks=[
                checkpoint_callback_mpo,
            ],
        )

        # train the model_mpo
        multi_model_container_mpoctrl.load_state_dict(
            torch.load(checkpoint_callback.best_model_path)["state_dict"]
        )
        trainer_mpo.fit(
            multi_model_container_mpoctrl, train_dataloader_mpo, val_dataloader_mpo
        )

    train_totalLoss_list = multi_model_container.train_totalLoss_list
    train_totalLoss_list = np.stack(train_totalLoss_list)
    train_imbalanceLoss_list = multi_model_container.train_imbalanceLoss_list
    train_imbalanceLoss_list = np.stack(train_imbalanceLoss_list)
    train_cv_list = multi_model_container.train_cv_list
    train_cv_list = np.stack(train_cv_list)
    train_sampleCor_list = multi_model_container.train_sampleCor_list
    train_sampleCor_list = np.stack(train_sampleCor_list)
    train_reactionCor_list = multi_model_container.train_reactionCor_list
    train_reactionCor_list = np.stack(train_reactionCor_list)

    val_totalLoss_list = multi_model_container.val_totalLoss_list
    val_totalLoss_list = np.stack(val_totalLoss_list)
    val_imbalanceLoss_list = multi_model_container.val_imbalanceLoss_list
    val_imbalanceLoss_list = np.stack(val_imbalanceLoss_list)
    val_cv_list = multi_model_container.val_cv_list
    val_cv_list = np.stack(val_cv_list)
    val_sampleCor_list = multi_model_container.val_sampleCor_list
    val_sampleCor_list = np.stack(val_sampleCor_list)
    val_reactionCor_list = multi_model_container.val_reactionCor_list
    val_reactionCor_list = np.stack(val_reactionCor_list)

    loss_dict = {
        "train_totalLoss_list": train_totalLoss_list,
        "train_imbalanceLoss_list": train_imbalanceLoss_list,
        "train_cv_list": train_cv_list,  # cross-validation loss
        "train_sampleCor_list": train_sampleCor_list,  # sample correlation loss
        "train_reactionCor_list": train_reactionCor_list,  # reaction correlation loss
        "val_totalLoss_list": val_totalLoss_list,
        "val_imbalanceLoss_list": val_imbalanceLoss_list,
        "val_cv_list": val_cv_list,  # cross-validation loss
        "val_sampleCor_list": val_sampleCor_list,  # sample correlation loss
        "val_reactionCor_list": val_reactionCor_list,  # reaction correlation loss
    }
    return loss_dict, multi_model_container


def inference(
    args,
    reactions_normalizedNpData_dict,
    compounds_reactions_df,
    n_samples,
    n_reactions,
    multi_model_container,
):
    # split the data into train and validation
    y_dummy = pd.DataFrame(
        np.zeros((n_samples, n_reactions)), columns=compounds_reactions_df.columns
    )
    full_dataloader, _ = split_data(
        reactions_normalizedNpData_dict, y_dummy, "predict", test_size=0.0
    )
    full_dataloader = CombinedDataset(full_dataloader)

    full_dataloader = DataLoader(
        full_dataloader,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
    )

    trainer = Trainer(
        logger=False, max_epochs=1, accelerator=DEVICE, devices=1, strategy="auto"
    )
    trainer.test(multi_model_container, full_dataloader)
    predictions = multi_model_container.predictions
    multi_model_container.clear_cache()

    return predictions


# @pysnooper.snoop()
def main(args):
    # print the input parameters
    print(f"{SEP_SIGN} \nCurrent Input parameters:\n{args}\n {SEP_SIGN}")
    print(f"Current CPU cores: {cpu_count()} GPT devices: {torch.cuda.device_count()}")

    # load the reactions and the contained genes, stored in a json file
    reactions_genes_dict = load_reactions_genes(args)

    # load gene expression data
    # geneExpression is the gene expression data,
    # cols:=samples/cells, rows:=genes,
    # but the data will be transposed to rows:=samples/cells,
    # cols:=genes automatically
    gene_expression_data = load_gene_expression(
        args.input_dir_path, args.gene_expression_file_name, reactions_genes_dict
    )
    n_samples, n_genes = gene_expression_data.shape

    # load the compounds and the reactions data, it is an adj matrix
    # compouns_reactions is the adj matrix of the factor graph (reaction graph)
    # rows:=compounds, columns:=reactions, entries are 0,1,-1
    compounds_reactions_df = load_compounds_reactions(args)
    n_compounds, n_reactions = compounds_reactions_df.shape

    # data pre-processing, remove the genes which are not in the reactions_genes
    # and remove the reactions which are not in the compounds_reactions
    # and remove the compounds which are not in the compounds_reactions
    # and remove the samples which are not in the gene_expression_data
    gene_expression_data, reactions_genes_dict, compounds_reactions_df = (
        data_pre_processing(
            gene_expression_data, reactions_genes_dict, compounds_reactions_df
        )
    )
    print(f"Compounds Reactions ADJ Matrix: \n{compounds_reactions_df}\n")

    if gene_expression_data is None:
        print("\nNo Intersection of Genes between Data and (reactions)Reactions! \n")
        return False

    # normalize the gene expression data
    # return a dictionary, key:=reaction, value:=normalized gene expression data
    reactions_normalizedNpData_dict = normalize_gene_expression(
        gene_expression_data, reactions_genes_dict
    )

    # initialize the output dir path
    # prepare the input and output dir
    # args.output_dir_path, _ = init_output_dir_path(args)

    # train the model with MPO
    loss_dict_mpoctrl, multi_model_container_mpoctrl = train(
        args,
        reactions_normalizedNpData_dict,
        compounds_reactions_df,
        reactions_genes_dict,
        n_samples,
        n_reactions,
    )

    # plot the loss curves
    plot_loss(loss_dict_mpoctrl, args.output_dir_path)

    # save the loss results in json file
    loss_dict_mpoctrl_save_path = os.path.join(args.output_dir_path, "loss_dict.json")
    save_dict_to_json(loss_dict_mpoctrl, loss_dict_mpoctrl_save_path)

    samples_reactions_mpoctrl = inference(
        args,
        reactions_normalizedNpData_dict,
        compounds_reactions_df,
        n_samples,
        n_reactions,
        multi_model_container_mpoctrl,
    )

    samples_reactions_mpoctrl_df = pd.DataFrame(
        samples_reactions_mpoctrl,
        index=gene_expression_data.index,
        columns=compounds_reactions_df.columns,
    )

    args.mpo_n_epoch = 50
    samples_reactions_mpo_df = mpo(
        compounds_reactions_df, samples_reactions_mpoctrl_df, [], args
    )
    mpo_save_path = os.path.join(args.output_dir_path, "mpo.csv")
    samples_reactions_mpo_df.to_csv(mpo_save_path, index=True, header=True)

    samples_reactions_mpoctrl_np = merge_matrix(
        samples_reactions_mpoctrl_df.values, samples_reactions_mpo_df.values
    )
    samples_reactions_mpoctrl_df = pd.DataFrame(
        samples_reactions_mpoctrl_np,
        index=gene_expression_data.index,
        columns=compounds_reactions_df.columns,
    )

    # save the MPOCtrL flux results
    flux_res_save_path = os.path.join(args.output_dir_path, "flux.csv")
    samples_reactions_mpoctrl_df.to_csv(
        flux_res_save_path,
        index=True,
        header=True,
    )

    # compute the matabolites remaining in the samples after the reactions are performed
    matabolites_df = compute_matabolite(
        samples_reactions_mpoctrl_df, compounds_reactions_df
    )
    matabolites_save_path = os.path.join(args.output_dir_path, "matabolite.csv")
    matabolites_df.to_csv(matabolites_save_path, index=True, header=True)

    return True


def parse_arguments(parser):
    # global parameters
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--input_dir_path", type=str, default="./inputs/", help="The inputs directory."
    )
    parser.add_argument(
        "--network_dir_path",
        type=str,
        default="./inputs/",
        help="The inputs directory.",
    )
    parser.add_argument(
        "--output_dir_path",
        type=str,
        default="./outputs/",
        help="The outputs directory, you can find all outputs in this directory.",
    )
    parser.add_argument(
        "--gene_expression_file_name",
        type=str,
        default="NA",
        help="The scRNA-seq file name.",
    )
    parser.add_argument(
        "--compounds_reactions_file_name",
        type=str,
        default="NA",
        help="The table describes relationship between compounds and reactions. Each row is an intermediate metabolite and each column is metabolic reaction.",
    )
    parser.add_argument(
        "--reactions_genes_file_name",
        type=str,
        default="NA",
        help="The json file contains genes for each reaction. We provide human and mouse two models in scFEA.",
    )

    parser.add_argument("--experiment_name", type=str, default="Flux")

    # parameters for scFEA
    parser.add_argument(
        "--n_epoch",
        type=int,
        default=200,
        help="User defined Epoch for scFEA training.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=9999999, help="Batch size, scfea."
    )
    # parameters for bp_balance
    parser.add_argument(
        "--mpo_n_epoch",
        type=int,
        default=10,
        help="User defined Epoch for Message Passing Optimizer.",
    )
    parser.add_argument(
        "--mpo_stop_threshold",
        type=float,
        default=0.0001,
        help="delta for the stopping criterion",
    )
    parser.add_argument(
        "--mpo_learning_rate",
        type=float,
        default=0.7,
        help="mpo learning rate for the update step",
    )
    parser.add_argument(
        "--mpo_beta_2", type=float, default=0.5, help="beta_2 for main branch"
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPOCtrL")

    # global args
    args = parse_arguments(parser)

    main(args)
