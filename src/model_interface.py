# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import lightning as L


class AdaptiveLayer(nn.Module):
    """
    Layer with a learnable gating mechanism to control its activation.
    """

    def __init__(self, input_dim, output_dim):
        super(AdaptiveLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.gate = nn.Parameter(
            torch.zeros(output_dim)
        )  # Learnable gate parameter for each feature

    def forward(self, x):
        gate_activation = (
            torch.sigmoid(self.gate) + 1e-6
        )  # Add small epsilon to avoid zero
        linear_output = self.linear(x)
        return gate_activation * linear_output

def replace_nan_predictions(predictions, default_value=0.0):
    if torch.isnan(predictions).any():
        # print("NaN detected in predictions. Replacing with default value.")
        predictions[torch.isnan(predictions)] = default_value
    return predictions
    

class AdaptiveModel(L.LightningModule):
    def __init__(self, input_dim, dropout_rate=0.1, output_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate

        self.activation = nn.LeakyReLU()
        self.output_activation = nn.ReLU()

        # Replace all fully connected layers with AdaptiveLayer
        self.adaptive_layer_1 = AdaptiveLayer(input_dim, 2 * input_dim)
        self.bn1 = nn.BatchNorm1d(2 * input_dim)
        self.dropout1 = nn.Dropout(self.dropout_rate)

        self.adaptive_layer_2 = AdaptiveLayer(2 * input_dim, 4 * input_dim)
        self.bn2 = nn.BatchNorm1d(4 * input_dim)
        self.dropout2 = nn.Dropout(self.dropout_rate)

        self.adaptive_layer_3 = AdaptiveLayer(4 * input_dim, 8 * input_dim)
        self.bn3 = nn.BatchNorm1d(8 * input_dim)
        self.dropout3 = nn.Dropout(self.dropout_rate)

        self.fc_out = nn.Linear(8 * input_dim, output_dim)

    def skip_connection(self, x_pre):
        # repeat the x_pre 2 times to make the x_repeated's lengthe is 2 times of x_pre
        x_repeated = x_pre.repeat(1, 2)
        return x_repeated

    def forward(self, x):
        x1 = self.adaptive_layer_1(x)
        x1 = self.bn1(x1)
        x1 += self.skip_connection(x)
        x1 = self.activation(x1)
        if self.input_dim > 2:
            x1 = self.dropout1(x1)

        # Second block
        x2 = self.adaptive_layer_2(x1)
        x2 = self.bn2(x2)
        x2 += self.skip_connection(x1)  # Skip connection added before activation
        x2 = self.activation(x2)
        x2 = self.dropout2(x2)

        # Third block
        x3 = self.adaptive_layer_3(x2)
        x3 = self.bn3(x3)
        x3 += self.skip_connection(x2)  # Skip connection added before activation
        x3 = self.activation(x3)
        x3 = self.dropout3(x3)

        # Output layer
        x4 = self.fc_out(x3)
        x4 = self.output_activation(x4)

        x4 = replace_nan_predictions(x4)
        
        return x4


class AdaptiveMultipleModels(L.LightningModule):
    def __init__(self, models, compounds_reactions_np, reaction_names, train_flag):
        super().__init__()
        self.automatic_optimization = False  # Disable automatic optimization
        self.train_flag = train_flag
        self.models = nn.ModuleDict(
            {reaction_name: model for reaction_name, model in models.items()}
        )
        self.compounds_reactions_tensor = compounds_reactions_np
        self.reaction_names = reaction_names
        self.n_reactions = len(reaction_names)

        self.entropy_lambda = nn.Parameter(
            torch.tensor(0.1)
        )  # entropy loss to control the model is not overfitting

        self.predictions = []
        self.train_imbalanceLoss_list = []
        self.val_imbalanceLoss_list = []
        self.train_cv_list = []
        self.val_cv_list = []
        self.train_totalLoss_list = []
        self.val_totalLoss_list = []
        self.train_sampleCor_list = []
        self.val_sampleCor_list = []
        self.train_reactionCor_list = []
        self.val_reactionCor_list = []

    def setup(self, stage=None):
        self.compounds_reactions_tensor = torch.tensor(
            self.compounds_reactions_tensor, dtype=torch.float32, device=self.device
        )

    def forward(self, x, reaction_name):
        outputs = self.models[reaction_name](x)
        return outputs

    def replace_nan_weights(self, value=1e-6):
        for reaction, model in self.models.items():
            if model is None:
                continue
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    # print(f"NaN detected in {name}. Replacing with small values.")
                    param.data[torch.isnan(param)] = value

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()

        Y_batch = []
        samples_reactions_batch = []
        samples_reactions_dataMean_batch = []
        batch_size = batch[list(batch.keys())[0]]["X"].shape[0]
        for reaction_name in self.reaction_names:
            # model is None, add 0s to the batch and 1s to the mean batch
            if self.models[reaction_name] is None:
                output = torch.zeros(
                    batch_size, 1, device=self.device, dtype=torch.float32
                )
                samples_reactions_batch.append(output)
                samples_reactions_dataMean_batch.append(
                    torch.ones(batch_size, device=self.device, dtype=torch.float32)
                )
                Y_batch.append(
                    torch.zeros(batch_size, device=self.device, dtype=torch.float32)
                )
                continue

            # if the model is not None, get the output
            X = batch[reaction_name]["X"]
            Y = batch[reaction_name]["Y"]
            Y_batch.append(Y)
            output = self.forward(X, reaction_name)
            samples_reactions_batch.append(output)
            samples_reactions_dataMean_batch.append(X.mean(dim=1))

        samples_reactions_batch = (
            torch.stack(samples_reactions_batch).transpose(0, 1).squeeze(2)
        )
        samples_reactions_dataMean_batch = torch.stack(
            samples_reactions_dataMean_batch
        ).transpose(0, 1)
        Y_batch = torch.stack(Y_batch).transpose(0, 1)

        # calculate the total loss
        total_loss = None
        imbalance_loss = None
        cv = None
        sample_cor = None
        reaction_cor = None
        if self.train_flag == "train_model":
            # constrained learning loss
            total_loss, imbalance_loss, sample_cor, reaction_cor, cv = (
                self.get_total_loss_cl(
                    samples_reactions_batch,
                    samples_reactions_dataMean_batch,
                    "train_step",
                )
            )
            self.train_imbalanceLoss_list.append(imbalance_loss.detach().cpu().numpy())
            self.train_cv_list.append(cv.detach().cpu().numpy())
            self.train_totalLoss_list.append(total_loss.detach().cpu().numpy())
            self.train_sampleCor_list.append(sample_cor.detach().cpu().numpy())
            self.train_reactionCor_list.append(reaction_cor.detach().cpu().numpy())

        elif self.train_flag == "train_model_mpo":
            total_loss = self.get_total_loss_sl(
                samples_reactions_batch,
                samples_reactions_dataMean_batch,
                Y_batch,
                "train_step",
            )

        self.log(
            "train_total_loss",
            total_loss.to(self.device),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # Directly after the manual_backward call in your training_step
        self.manual_backward(total_loss)
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad()

        self.replace_nan_weights()

        return total_loss

    def validation_step(self, batch, batch_idx):
        samples_reactions_batch = []
        samples_reactions_dataMean_batch = []
        batch_size = batch[list(batch.keys())[0]]["X"].shape[0]
        Y_batch = []
        for reaction_name in self.reaction_names:
            # model is None, add 0s to the batch and 1s to the mean batch
            if self.models[reaction_name] is None:
                output = torch.zeros(
                    batch_size, 1, device=self.device, dtype=torch.float32
                )
                samples_reactions_batch.append(output)
                samples_reactions_dataMean_batch.append(
                    torch.ones(batch_size, device=self.device, dtype=torch.float32)
                )
                Y_batch.append(
                    torch.zeros(batch_size, device=self.device, dtype=torch.float32)
                )
                continue

            # if the model is not None, get the output
            X = batch[reaction_name]["X"]
            Y = batch[reaction_name]["Y"]
            Y_batch.append(Y)
            output = self.forward(X, reaction_name)
            samples_reactions_batch.append(output)
            samples_reactions_dataMean_batch.append(X.mean(dim=1))

        samples_reactions_batch = (
            torch.stack(samples_reactions_batch).transpose(0, 1).squeeze(2)
        )
        samples_reactions_dataMean_batch = torch.stack(
            samples_reactions_dataMean_batch
        ).transpose(0, 1)
        Y_batch = torch.stack(Y_batch).transpose(0, 1)

        # calculate the total loss
        total_loss = None
        imbalance_loss = None
        cv = None
        sample_cor = None
        reaction_cor = None
        if self.train_flag == "train_model":
            # constrained learning loss
            total_loss, imbalance_loss, sample_cor, reaction_cor, cv = (
                self.get_total_loss_cl(
                    samples_reactions_batch,
                    samples_reactions_dataMean_batch,
                    "val_step",
                )
            )
            self.val_imbalanceLoss_list.append(imbalance_loss.detach().cpu().numpy())
            self.val_cv_list.append(cv.detach().cpu().numpy())
            self.val_totalLoss_list.append(total_loss.detach().cpu().numpy())
            self.val_sampleCor_list.append(sample_cor.detach().cpu().numpy())
            self.val_reactionCor_list.append(reaction_cor.detach().cpu().numpy())

        elif self.train_flag == "train_model_mpo":
            total_loss = self.get_total_loss_sl(
                samples_reactions_batch,
                samples_reactions_dataMean_batch,
                Y_batch,
                "val_step",
            )

        self.log(
            "val_total_loss",
            total_loss.to(self.device),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return total_loss

    def test_step(self, batch, batch_idx):
        samples_reactions_batch = []
        samples_reactions_dataMean_batch = []
        batch_size = batch[list(batch.keys())[0]]["X"].shape[0]
        Y_batch = []
        for reaction_name in self.reaction_names:
            # model is None, add 0s to the batch and 1s to the mean batch
            if self.models[reaction_name] is None:
                output = torch.zeros(
                    batch_size, 1, device=self.device, dtype=torch.float32
                )
                samples_reactions_batch.append(output)
                samples_reactions_dataMean_batch.append(
                    torch.ones(batch_size, device=self.device, dtype=torch.float32)
                )
                Y_batch.append(torch.zeros(batch_size, device=self.device))
                continue

            # if the model is not None, get the output
            X = batch[reaction_name]["X"]
            Y = batch[reaction_name]["Y"]
            Y_batch.append(Y)
            output = self.forward(X, reaction_name)
            samples_reactions_batch.append(output)
            samples_reactions_dataMean_batch.append(X.mean(dim=1))

        samples_reactions_batch = (
            torch.stack(samples_reactions_batch).transpose(0, 1).squeeze(2)
        )
        self.predictions.append(samples_reactions_batch)

        return True

    def on_test_epoch_end(self):
        self.predictions = torch.cat(self.predictions, dim=0).detach().cpu().numpy()
        # abs the predictions
        self.predictions = abs(self.predictions)

    def clear_cache(self):
        self.predictions = []

    def compute_pearson_correlation(self, matrix1, matrix2, dim=1):
        # Step 1: Normalize rows/columns (subtract mean, divide by std dev)
        matrix1_centered = matrix1 - matrix1.mean(dim=dim, keepdim=True)
        matrix2_centered = matrix2 - matrix2.mean(dim=dim, keepdim=True)

        matrix1_std = matrix1_centered.norm(dim=dim, keepdim=True)
        matrix2_std = matrix2_centered.norm(dim=dim, keepdim=True)

        # Step 2: Compute dot product for covariance and divide by product of norms
        covariance = (matrix1_centered * matrix2_centered).sum(dim=dim, keepdim=False)
        correlation = covariance / (matrix1_std * matrix2_std).squeeze(dim)

        # Step 3: Compute Pearson correlation distance
        # pearson_correlation_distance = 1 - correlation
        # fill na with 0
        correlation[torch.isnan(correlation)] = 0
        correlation = correlation.mean()

        return correlation

    def compute_imbalance_loss(self, samples_reactions):
        # 1. Expand dimensions to align for pairwise row-wise element-wise product
        # Shape after expansion: (1000, 1, 300) for matrix1, and (1, 1000, 300) for matrix2
        row_sum = samples_reactions.sum(dim=1) + 1e-8
        # row_norm = torch.norm(samples_reactions, dim=1)
        scale = 200.0 if self.n_reactions > 100 else 50.0 if self.n_reactions > 50 else 20.0

        # normalize the sample row to make the model is 1
        # samples_reactions = samples_reactions / row_norm.view(-1, 1)
        samples_reactions = samples_reactions / row_sum.view(-1, 1)
        samples_reactions = samples_reactions * scale

        # Ensure both matrices have the same number of columns
        # assert samples_reactions.size(1) == self.compounds_reactions_tensor.size(
        #    1
        # ), "Matrices must have the same number of columns."

        # Expand matrix1 and matrix2 for broadcasting
        expanded_matrix1 = samples_reactions.unsqueeze(1)  # Shape: (n1, 1, c)
        expanded_matrix2 = self.compounds_reactions_tensor.unsqueeze(0)  # Shape: (1, n2, c)

        # Compute element-wise product between each row of matrix1 and all rows of matrix2
        pairwise_products = expanded_matrix1 * expanded_matrix2  # Shape: (n1, n2, c)

        # Compute row sums for each combination of rows
        row_sums = pairwise_products.sum(dim=2)  # Shape: (n1, n2)

        # Compute squared sums and mean for each row in matrix1
        squared_sums = row_sums**2  # Shape: (n1, n2)
        row_means = squared_sums.mean(dim=1)  # Shape: (n1)

        # Compute the overall mean of the means
        imbalance_loss = row_means.mean()  # Scalar
        return imbalance_loss

    def compute_gradient_negative_loss(self):
        total_gradient_penalty = 0
        total_gradient_penalty = sum(
            torch.sum(torch.relu(-param.grad))
            for model in self.models.values()
            if model is not None
            for param in model.parameters()
            if param.grad is not None
        )
        return total_gradient_penalty

    def compute_model_entropy_loss(self):
        entropy_loss = -torch.sum(
            torch.sigmoid(self.entropy_lambda)
            * torch.log(torch.sigmoid(self.entropy_lambda) + 1e-8)
        )
        return entropy_loss

    def compute_cv(self, samples_reactions):
        cv = samples_reactions.std(dim=0) / (samples_reactions.mean(dim=0) + 1e-8)
        # fille na with 0
        cv[torch.isnan(cv)] = 0
        cv = abs(cv)
        cv = cv.mean()
        return cv

    # calculate the total constrained loss
    def get_total_loss_cl(self, samples_reactions, samples_reactions_geneMean, step):

        # *************************************** data fitting loss calculation **********************************************
        # column coefficient of variation
        cv = self.compute_cv(samples_reactions)

        # pearson correlation distance loss by column/reaction
        reaction_cor = self.compute_pearson_correlation(
            samples_reactions, samples_reactions_geneMean
        )
        #if reaction_cor < 0:
        #    reaction_cor = -reaction_cor

        # pearson correlation distance loss by row/sample
        sample_cor = self.compute_pearson_correlation(
            samples_reactions.T, samples_reactions_geneMean.T
        )
        #if sample_cor < 0:
        #    sample_cor = -sample_cor

        # imbalance loss
        imbalance_loss = self.compute_imbalance_loss(samples_reactions)

        # gradient positive penalty
        gradient_penalty = self.compute_gradient_negative_loss()

        # calculate the entropy loss
        entropy_loss = self.compute_model_entropy_loss()

        loss_list = [
            imbalance_loss,
            1 - reaction_cor,
            1 - sample_cor,
            gradient_penalty,
            entropy_loss,
        ]
        non_zero_loss_list = [loss for loss in loss_list if loss != 0]
        total_loss = sum(non_zero_loss_list) / len(non_zero_loss_list)
        return (total_loss, imbalance_loss, sample_cor, reaction_cor, cv)

    # calculate the total loss for the model mpo, supervised learning
    def get_total_loss_sl(self, samples_reactions, samples_reactions_geneMean, Y, step):

        # *************************************** data fitting loss calculation ************
        # MSE loss
        mse_loss = F.mse_loss(samples_reactions, Y)

        # column coefficient of variation loss
        cv = self.compute_cv(samples_reactions)

        # pearson correlation loss by column
        reaction_cor = self.compute_pearson_correlation(
            samples_reactions, samples_reactions_geneMean
        )
        #if reaction_cor < 0:
        #    reaction_cor = -reaction_cor

        # pearson correlation loss by row
        sample_cor = self.compute_pearson_correlation(
            samples_reactions.T, samples_reactions_geneMean.T
        )
        #if sample_cor < 0:
        #    sample_cor = -sample_cor

        # imbalance loss
        imbalance_loss = self.compute_imbalance_loss(samples_reactions)
        # gradient positive penalty
        gradient_penalty = self.compute_gradient_negative_loss()

        entropy_loss = self.compute_model_entropy_loss()
        loss_list = [
            mse_loss,
            imbalance_loss,
            1 - reaction_cor,
            1 - sample_cor,
            gradient_penalty,
            entropy_loss,
        ]
        non_zero_loss_list = [loss for loss in loss_list if loss != 0]
        total_loss = sum(non_zero_loss_list) / len(non_zero_loss_list)
        return total_loss

    def configure_optimizers(self):
        optimizers = [
            Adam(self.models[reaction_name].parameters(), lr=0.0005, weight_decay=1e-5)
            for reaction_name in self.reaction_names
            if self.models[reaction_name] is not None
        ]
        return optimizers
