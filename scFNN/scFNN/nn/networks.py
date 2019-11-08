import abc

import six
import torch
import torch.nn as nn
from torch import optim

from scFNN.nn.farfalle import SingleHeadAttention, NeuronPool, get_normalization_function
from scFNN.nn.modules import exp_module, ZeroInflatedNegativeBinomialLoss, NegativeBinomialLoss, \
    log_transform_module, identity_module, ZeroInflatedNegativeBinomialLossLogVer, \
    NegativeBinomialLossLogVer, SelectionWithKeyInputNeuronPool, SelectionWithKeyOutputNeuronPool, LibrarySizeEstimator


@six.add_metaclass(abc.ABCMeta)
class AbstractNetwork(nn.Module):
    def __init__(self):
        super(AbstractNetwork, self).__init__()
        self.epoch = 0

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def loss(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def get_full_state_dict(self, *args, **kwargs):
        pass

    def _init_optimization_modules(self, learning_rate, early_stopping_patience):
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', verbose=True,
                                                              patience=5, threshold=1e-4,
                                                              min_lr=1e-8, factor=1 / 4)
        self.early_stopping_patience = early_stopping_patience
        self.best_test_loss = None
        self.early_stopping_counter = 0

    def train_on(self, data):
        self.optimizer.zero_grad()

        output = self.forward(data)
        loss = self.loss(data, output)
        final_loss = loss["loss"]
        final_loss.backward()

        self.optimizer.step()

        for loss_type in loss:
            loss[loss_type] = loss[loss_type].item()

        return loss

    def finalize_train(self, train_loss, test_loss):
        self.scheduler.step(test_loss["loss"])

        if self.best_test_loss is not None and test_loss["loss"] >= self.best_test_loss:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.early_stopping_patience:
                return "stop"
            return "EarlyStopping: %i / %i" % (self.early_stopping_counter, self.early_stopping_patience)
        else:
            self.best_test_loss = test_loss["loss"]
            self.early_stopping_counter = 0

    def get_output(self, data):
        output = self.forward(data)
        return output

    @staticmethod
    def load_from_full_state(state):
        class_name = state.pop("model")
        state_dict = state.pop("torch_state_dict")
        model = eval(class_name)(**state)
        model.load_state_dict(state_dict)
        return model

    def set_epoch(self, epoch):
        self.epoch = epoch


class FNNAutoEncoder(AbstractNetwork):
    def __init__(self, input_dim, input_keys, gene_embedding_dim, encoder_sizes, decoder_sizes,
                 learning_rate=1e-2, early_stopping_patience=10,
                 dropout_rate=0.2, attention_normalization="l2", hidden_activation="nn.LeakyReLU(0.2)",
                 input_scaling="log-scale", reconstruction_loss="zinb", log_ver=True,
                 library_size_normalization=True):
        super(FNNAutoEncoder, self).__init__()

        self.input_dim = input_dim
        self.input_keys = input_keys
        self.gene_embedding_dim = gene_embedding_dim
        self.encoder_sizes = encoder_sizes
        self.decoder_sizes = decoder_sizes
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.library_size_factor = 1e6
        self.dropout_rate = dropout_rate
        self.attention_normalization = attention_normalization
        self.hidden_activation = hidden_activation
        self.input_scaling = input_scaling
        self.log_ver = log_ver
        self.library_size_normalization = library_size_normalization
        self.reconstruction_loss = reconstruction_loss
        self.linked_zinb_parameters = False
        if reconstruction_loss == "linked-zinb":
            self.linked_zinb_parameters = True
            self.reconstruction_loss = "zinb"

        self.normalization = get_normalization_function(attention_normalization, self.gene_embedding_dim)
        self.attention = SingleHeadAttention(self.normalization, self.gene_embedding_dim)

        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.library_size_estimator = LibrarySizeEstimator([self.input_dim, ])

        self.input_net = SelectionWithKeyInputNeuronPool(
            self.input_keys, self.gene_embedding_dim, linear_transform=True)

        encoder_neuron_pools = [
            NeuronPool(
                n_neurons=size, embedding_dim=self.gene_embedding_dim,
                transformation=("BatchNorm", hidden_activation, "PointwiseLinear"),
                axon_dendrite_linked_embedding=True,
                synaptic_module=self.attention
            )
            for size in encoder_sizes
        ]

        self.encoder = nn.Sequential(*encoder_neuron_pools)

        decoder_neuron_pools = [
            NeuronPool(
                n_neurons=size, embedding_dim=self.gene_embedding_dim,
                transformation=("BatchNorm", hidden_activation, "PointwiseLinear"),
                axon_dendrite_linked_embedding=True,
                synaptic_module=self.attention
            )
            for size in decoder_sizes
        ]

        self.decoder = nn.Sequential(*decoder_neuron_pools)

        if self.linked_zinb_parameters:
            self.output = SelectionWithKeyOutputNeuronPool(
                input_keys, self.gene_embedding_dim, self.attention, linear_transform=True)
            self.output_fork = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=10, kernel_size=1),
                                             nn.Conv1d(in_channels=10, out_channels=10, kernel_size=1),
                                             nn.Conv1d(in_channels=10, out_channels=10, kernel_size=1),
                                             nn.Conv1d(in_channels=10, out_channels=3, kernel_size=1))
        else:
            self.output_mean = SelectionWithKeyOutputNeuronPool(
                input_keys, self.gene_embedding_dim, self.attention, linear_transform=True)
            self.output_r = SelectionWithKeyOutputNeuronPool(
                input_keys, self.gene_embedding_dim, self.attention, linear_transform=True)
            self.output_pi = SelectionWithKeyOutputNeuronPool(
                input_keys, self.gene_embedding_dim, self.attention, linear_transform=True)

        if self.log_ver:
            self.mean_module = identity_module()
            self.r_module = identity_module()
        else:
            self.mean_module = exp_module()
            self.r_module = nn.Softplus()
        self.pi_logit_module = identity_module()

        if input_scaling == "log-scale":
            self.input_scaling_module = log_transform_module()
        else:
            raise NotImplementedError()

        if self.reconstruction_loss == "mse":
            if log_ver:
                self.reconstruction_loss_module = nn.MSELoss(reduction="sum")
            else:
                raise NotImplementedError()
        elif self.reconstruction_loss == "nb":
            if log_ver:
                self.reconstruction_loss_module = NegativeBinomialLossLogVer(reduction="sum")
            else:
                self.reconstruction_loss_module = NegativeBinomialLoss(reduction="sum")
        elif self.reconstruction_loss == "zinb":
            if log_ver:
                self.reconstruction_loss_module = ZeroInflatedNegativeBinomialLossLogVer(reduction="sum")
            else:
                self.reconstruction_loss_module = ZeroInflatedNegativeBinomialLoss(reduction="sum")
        else:
            raise NotImplementedError()

        self._init_optimization_modules(learning_rate=self.learning_rate,
                                        early_stopping_patience=self.early_stopping_patience)

    def normalize(self, x):
        # library_sizes = (3 * x.sum(dim=1) + self.library_size_estimator(x)) / 4
        library_sizes = x.sum(dim=1)
        normalized_x = torch.mm(torch.diag(self.library_size_factor / library_sizes), x)

        return normalized_x, library_sizes

    def encode(self, x, keys):
        transformed_input = self.input_scaling_module(x)
        input_pair = self.input_net((transformed_input, None), keys)
        encoded_layer = self.encoder(input_pair)

        return encoded_layer

    def decode(self, x, keys):
        decoded = self.decoder(x)
        if self.linked_zinb_parameters:
            outputs, _ = self.output(decoded, keys)
            outputs = self.output_fork(outputs.unsqueeze(-2))
            mean_nodes = self.mean_module(outputs[:, 0, :])
            r_nodes = self.r_module(outputs[:, 1, :])
            pi_nodes = self.pi_logit_module(outputs[:, 2, :])
        else:
            mean_nodes = self.mean_module(self.output_mean(decoded, keys)[0])
            r_nodes = self.r_module(self.output_r(decoded, keys)[0])
            pi_nodes = self.pi_logit_module(self.output_pi(decoded, keys)[0])

        return mean_nodes, r_nodes, pi_nodes

    def denormalize(self, mean_nodes, library_sizes):
        """
        Reverts normalization done using normalize function
        """
        if self.log_ver:
            mean_nodes = torch.log(library_sizes / self.library_size_factor).unsqueeze(-1) + mean_nodes
        else:
            mean_nodes = torch.mm(torch.diag(library_sizes / self.library_size_factor), mean_nodes)
        return mean_nodes

    def forward(self, data):
        x = data.get("X")
        keys = data.get("keys")

        x = self.dropout(x)

        # Library size Correction
        if self.library_size_normalization:
            x, library_sizes = self.normalize(x)

        encoded_layer = self.encode(x, keys)
        mean_nodes, r_nodes, pi_nodes = self.decode(encoded_layer, keys)

        normal_mean_nodes = mean_nodes
        if self.library_size_normalization:
            mean_nodes = self.denormalize(mean_nodes=mean_nodes, library_sizes=library_sizes)

        # For external use only :D
        if self.log_ver:
            mean = torch.exp(mean_nodes)
            normal_mean_nodes = torch.exp(normal_mean_nodes)
            r = torch.exp(r_nodes)
        else:
            mean = mean_nodes
            r = r_nodes
        pi = torch.sigmoid(pi_nodes)

        # Output part
        return {"mean_nodes": mean_nodes, "r_nodes": r_nodes, "pi_nodes": pi_nodes,
                "mean": mean, "r": r, "pi": pi, "pred": mean, "normal_mean_nodes": normal_mean_nodes}

    def loss(self, data, output):
        x = data.get("X")

        mean_nodes = output.get("mean_nodes")
        r_nodes = output.get("r_nodes")
        pi_nodes = output.get("pi_nodes")

        if self.reconstruction_loss == "mse":
            if self.log_ver:
                loss = self.reconstruction_loss_module(target=torch.log(x + 1), input=mean_nodes)
            else:
                raise NotImplementedError()
        elif self.reconstruction_loss == "nb":
            if self.log_ver:
                loss = self.reconstruction_loss_module(x=x, mean_log=mean_nodes, r_log=r_nodes)
            else:
                loss = self.reconstruction_loss_module(x=x, mean=mean_nodes, r=r_nodes)
        elif self.reconstruction_loss == "zinb":
            if self.log_ver:
                loss = self.reconstruction_loss_module(x=x, mean_log=mean_nodes, r_log=r_nodes, pi_logit=pi_nodes)
            else:
                loss = self.reconstruction_loss_module(x=x, mean=mean_nodes, r=r_nodes, pi_logit=pi_nodes)
        else:
            raise NotImplementedError()

        return {"loss": loss}

    def get_full_state_dict(self):
        reconstruction_loss = self.reconstruction_loss
        if self.linked_zinb_parameters:
            reconstruction_loss = "linked-zinb"
        return {
            "model": "FNNAutoEncoder",
            "torch_state_dict": self.state_dict(),
            "input_dim": self.input_dim,
            "input_keys": self.input_keys,
            "gene_embedding_dim": self.gene_embedding_dim,
            "encoder_sizes": self.encoder_sizes,
            "decoder_sizes": self.decoder_sizes,
            "learning_rate": self.learning_rate,
            "early_stopping_patience": self.early_stopping_patience,
            "dropout_rate": self.dropout_rate,
            "attention_normalization": self.attention_normalization,
            "hidden_activation": self.hidden_activation,
            "input_scaling": self.input_scaling,
            "log_ver": self.log_ver,
            "library_size_normalization": self.library_size_normalization,
            "reconstruction_loss": reconstruction_loss,
        }
