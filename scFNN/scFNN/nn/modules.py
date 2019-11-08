import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from scFNN.nn.functions import negative_binomial, zero_inflated_negative_binomial, \
    zero_inflated_negative_binomial_log_version, negative_binomial_log_ver


class SequentialOfLinearLayers(nn.Module):
    def __init__(self, input_size, layer_sizes,
                 bias_indicator=None, bias_indicators=None,
                 bath_norm_indicator=None, bath_norm_indicators=None,
                 dropout_rate=None, dropout_rates=None,
                 activation_module=None, activation_modules=None):
        """
        :param input_size: size of input
        :param layer_sizes: a list containing size of layers
        :param bias_indicator: whether to use bias or not (use if it is same among all layers)
        :param bias_indicators: a list indicating whether to use bias in each layer or not (use
                                if it is same among all layers)
        :param bath_norm_indicator: whether to use batch normalization or not (use if it is same among all layers)
        :param bath_norm_indicators: a list indicating whether to use batch normalization in each layer or not (use
                                if it is same among all layers)
        :param dropout_rate: dropout rate among all layers (use if dropout is same among all layers)
        :param dropout_rates: a list containing dropout rate of layers (in case dropout is variable among layers)
        :param activation_module: activation function among all layers (use if it is same among all layers)
        :param activation_modules: a list containing activation function of layers (if it is variable among layers)
        """
        super(SequentialOfLinearLayers, self).__init__()

        if dropout_rate is not None:
            assert dropout_rates is None
            dropout_rates = [dropout_rate for _ in layer_sizes]

        if bath_norm_indicator is not None:
            assert bath_norm_indicators is None
            bath_norm_indicators = [bath_norm_indicator for _ in layer_sizes]

        if bias_indicator is not None:
            assert bias_indicators is None
            bias_indicators = [bias_indicator for _ in layer_sizes]

        if activation_module is not None:
            assert activation_modules is None
            activation_modules = [activation_module for _ in layer_sizes]

        self.layers = []
        for i in range(len(layer_sizes)):

            if dropout_rates is not None and dropout_rates[i] > 0:
                self.layers.append(nn.Dropout(dropout_rates[i]))

            self.layers.append(nn.Linear(([input_size] + layer_sizes)[i], layer_sizes[i], bias=bias_indicators[i]))

            if bath_norm_indicators is not None and bath_norm_indicators[i]:
                self.layers.append(nn.BatchNorm1d(layer_sizes[i]))

            if activation_modules is not None:
                self.layers.append(eval(activation_modules[i]))

        self.network = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.network(x)


class NegativeBinomialLoss(nn.Module):
    def __init__(self, reduction="sum"):
        super(NegativeBinomialLoss, self).__init__()
        self.reduction = reduction

    def forward(self, x, mean, r):
        ret = - negative_binomial(k=x, m=mean, r=r)
        if self.reduction is None:
            return ret
        ret = torch.mean(ret) if self.reduction == 'mean' else torch.mean(ret, dim=-1).sum()
        return ret


class MyMSELoss(nn.Module):
    def __init__(self, reduction="sum"):
        super(MyMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, x, mean):
        ret = (x - mean) ** 2
        if self.reduction is None:
            return ret
        ret = torch.mean(ret) if self.reduction == 'mean' else torch.mean(ret, dim=-1).sum()
        return ret


class NegativeBinomialLossLogVer(nn.Module):
    def __init__(self, reduction="sum"):
        super(NegativeBinomialLossLogVer, self).__init__()
        self.reduction = reduction

    def forward(self, x, mean_log, r_log):
        ret = - negative_binomial_log_ver(k=x, m_log=mean_log, r_log=r_log)
        if self.reduction is None:
            return ret
        ret = torch.mean(ret) if self.reduction == 'mean' else torch.mean(ret, dim=-1).sum()
        return ret


class ZeroInflatedNegativeBinomialLoss(nn.Module):
    def __init__(self, reduction="sum"):
        super(ZeroInflatedNegativeBinomialLoss, self).__init__()
        self.reduction = reduction

    def forward(self, x, mean, r, pi_logit):
        ret = - zero_inflated_negative_binomial(k=x, m=mean, r=r, pi_logit=pi_logit)
        if self.reduction is None:
            return ret
        ret = torch.mean(ret) if self.reduction == 'mean' else torch.mean(ret, dim=-1).sum()
        return ret


class ZeroInflatedNegativeBinomialLossLogVer(nn.Module):
    def __init__(self, reduction="sum"):
        super(ZeroInflatedNegativeBinomialLossLogVer, self).__init__()
        self.reduction = reduction

    def forward(self, x, mean_log, r_log, pi_logit):
        ret = - zero_inflated_negative_binomial_log_version(k=x, m_log=mean_log, r_log=r_log, pi_logit=pi_logit)
        if self.reduction is None:
            return ret
        ret = torch.mean(ret) if self.reduction == 'mean' else torch.mean(ret, dim=-1).sum()
        return ret


class WrapperModule(nn.Module):
    def __init__(self, transformation, description=""):
        super(WrapperModule, self).__init__()
        self.transformation = transformation
        self.description = description

    def forward(self, x):
        return self.transformation(x)

    def __repr__(self):
        return self.description


class SelectionWithKeyInputNeuronPool(nn.Module):
    def __init__(self, keys, embedding_dim, linear_transform=True):
        super(SelectionWithKeyInputNeuronPool, self).__init__()

        # Set instance parameters
        self.keys = keys
        self.n_neurons = len(keys)
        self.embedding_dim = embedding_dim
        self.linear_transform = linear_transform

        # Define embeddings of axons and dendrites
        self.input_axon_embeddings = nn.Parameter(torch.empty(self.n_neurons, embedding_dim))

        # Linear transform part
        if self.linear_transform:
            self.scale = nn.Parameter(torch.empty([self.n_neurons, ]))
            self.bias = nn.Parameter(torch.empty([self.n_neurons, ]))

        # Define mapping between keys and indices
        self.mapping = pd.DataFrame(data=np.arange(len(keys)).reshape(1, -1), columns=keys)

        # Set initial values of module parameters
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.input_axon_embeddings, mode='fan_out')

        if self.linear_transform:
            self.scale.data.uniform_(1, 1)
            self.bias.data.zero_()

    def init_from_another_module(self, other):
        common_genes = np.intersect1d(self.keys, other.keys)

        with torch.no_grad():
            self.input_axon_embeddings.data[self.map(common_genes)] = \
                other.input_axon_embeddings.data[other.map(common_genes)]
            if self.linear_transform and other.linear_transform:
                self.scale.data[self.map(common_genes)] = other.scale.data[other.map(common_genes)]
                self.bias.data[self.map(common_genes)] = other.bias.data[other.map(common_genes)]

    def map(self, keys):
        return self.mapping[keys].values[0]

    def forward(self, input_pair, keys=None):
        assert keys is not None

        inputs, _ = input_pair
        indices = self.map(keys)
        if self.linear_transform:
            inputs = self.bias[indices] + self.scale[indices] * inputs
        return inputs, self.input_axon_embeddings[indices] + 0.

    def extra_repr(self):
        return 'n_neurons={n_neurons}, embedding_dim={embedding_dim},' \
               'linear_transform={linear_transform}\n'.format(**self.__dict__) + \
               'input_axon_embeddings=Tensor({})'.format(self.input_axon_embeddings.shape)


class LibrarySizeEstimator(nn.Module):
    def __init__(self, input_size):
        super(LibrarySizeEstimator, self).__init__()

        self.input_size = input_size
        self.log_weights = nn.Parameter(torch.Tensor(*self.input_size))

        self._reset_parameters()

    def _reset_parameters(self):
        self.log_weights.data.zero_()

    def forward(self, x):
        return F.linear(x, torch.exp(self.log_weights))


class SelectionWithKeyOutputNeuronPool(nn.Module):
    def __init__(self, keys, embedding_dim, synaptic_module, linear_transform):
        super(SelectionWithKeyOutputNeuronPool, self).__init__()

        # Set instance parameters
        self.keys = keys
        self.n_neurons = len(keys)
        self.embedding_dim = embedding_dim
        self.linear_transform = linear_transform

        # Define embeddings of axons and dendrites
        self.dendrite_embeddings = nn.Parameter(torch.empty(self.n_neurons, embedding_dim))

        # Linear transform part
        if self.linear_transform:
            self.scale = nn.Parameter(torch.empty([self.n_neurons, ]))
            self.bias = nn.Parameter(torch.empty([self.n_neurons, ]))

        # Define mapping between keys and indices
        self.mapping = pd.DataFrame(data=np.arange(len(keys)).reshape(1, -1), columns=keys)

        # Define synaptic module that connects dendrites to axons
        self.synaptic_module = synaptic_module

        # Set initial values of module parameters
        self._reset_dendrite_parameters()

    def _reset_dendrite_parameters(self):
        nn.init.kaiming_uniform_(self.dendrite_embeddings, mode='fan_in')

        if self.linear_transform:
            self.scale.data.uniform_(1, 1)
            self.bias.data.zero_()

    def init_from_another_module(self, other):
        common_genes = np.intersect1d(self.keys, other.keys)

        with torch.no_grad():
            self.dendrite_embeddings.data[self.map(common_genes)] = \
                other.dendrite_embeddings.data[other.map(common_genes)]
            if self.linear_transform and other.linear_transform:
                self.scale.data[self.map(common_genes)] = other.scale.data[other.map(common_genes)]
                self.bias.data[self.map(common_genes)] = other.bias.data[other.map(common_genes)]

    def map(self, keys):
        return self.mapping[keys].values[0]

    def forward(self, input_pair, keys=None):
        assert keys is not None

        inputs, input_embeddings = input_pair
        indices = self.map(keys)

        dendrite_pulse, _ = self.synaptic_module(
            q=self.dendrite_embeddings[indices],
            k=input_embeddings,
            v=inputs.unsqueeze(-1),
        )

        axon_outputs = dendrite_pulse.squeeze(-1)

        if self.linear_transform:
            axon_outputs = self.bias[indices] + self.scale[indices] * axon_outputs

        return axon_outputs, None

    def extra_repr(self):
        return 'n_neurons={n_neurons}, embedding_dim={embedding_dim},' \
               'linear_transform={linear_transform}\n'.format(**self.__dict__) + \
               'dendrite_embeddings=Tensor({})'.format(self.dendrite_embeddings.shape)


def identity_module():
    return WrapperModule(lambda x: x, description="identity")


def exp_module():
    return WrapperModule(lambda x: torch.exp(x), description="exp(x)")


def log_transform_module():
    return WrapperModule(lambda x: torch.log(x + 1), description="log(x + 1)")
