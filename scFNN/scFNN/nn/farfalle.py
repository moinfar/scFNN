import abc

import six
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointwiseLinearLayer(nn.Module):
    def __init__(self, shape, low=-1, high=1, bias=True):
        super(PointwiseLinearLayer, self).__init__()

        self.weight = nn.Parameter(torch.empty(*shape))
        self.bias = nn.Parameter(torch.empty(*shape)) if bias else None

        self.low = low
        self.high = high
        self._reset_parameters()

    def _reset_parameters(self):
        self.weight.data.uniform_(self.low, self.high)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        x = self.weight.unsqueeze(0) * x
        if self.bias is not None:
            x = self.bias.unsqueeze(0) + x
        return x

    def extra_repr(self):
        return 'shape={shape}, init=uniform({low}, {high}), bias={bias}'.format(
            shape=self.weight.shape, low=self.low, high=self.high, bias=self.bias is not None)


def get_normalization_function(normalization, embedding_dim, normalization_dim=-1):
    if callable(normalization):
        return normalization
    if normalization == "identity":
        return lambda x: x
    if normalization == "l1":
        return lambda x: F.normalize(x, p=1, dim=normalization_dim)
    if normalization == "l2":
        return lambda x: F.normalize(x, p=2, dim=normalization_dim)
    if normalization == "softmax":
        return lambda x: F.softmax(x / embedding_dim ** 0.5, dim=normalization_dim)
    raise ValueError("Normalization `{}` not defined".format(normalization))


def get_transformation_module(transformation, shape):
    if isinstance(transformation, list) or isinstance(transformation, tuple):
        return nn.Sequential(*[get_transformation_module(t, shape) for t in transformation])
    if isinstance(transformation, nn.Module):
        return transformation
    if transformation == "Identity" or transformation is None:
        return nn.Identity()
    if transformation == "PointwiseLinear":
        return PointwiseLinearLayer(shape)
    if transformation == "LayerNorm":
        return nn.LayerNorm(shape, elementwise_affine=True)
    if transformation == "BatchNorm":
        assert len(shape) == 1
        return nn.BatchNorm1d(shape[0], affine=True)
    elif isinstance(transformation, str):
        return eval(transformation)
    raise ValueError("Transformation `{}` not defined".format(transformation))


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, normalization):
        super(ScaledDotProductAttention, self).__init__()
        self.normalization = normalization

    def forward(self, q, k, v, cached_attn=None):
        if cached_attn is None:
            attn = torch.bmm(q, k.transpose(1, 2))  # n_head x lq x lk
            attn = self.normalization(attn)
        else:
            attn = cached_attn

        output = torch.bmm(v, attn.transpose(1, 2))

        return output, attn


class SingleHeadAttention(nn.Module):
    def __init__(self, normalization, d_model, linear_transform=False):
        super().__init__()

        self.attention = ScaledDotProductAttention(normalization=normalization)
        self.d_model = d_model
        if linear_transform:
            # Transforming both queries and keys will mean
            # we will compute qT * w_qsT * w_ks * k
            # Instead we will directly train W = w_qsT * w_ks.
            self.w = nn.Linear(self.d_model, self.d_model, bias=False)
        else:
            self.w = nn.Identity()

    def forward(self, q, k, v, cached_attn=None):
        len_q, _ = q.size()
        sz_b, len_v, d_v = v.size()
        v = v.permute(0, 2, 1).unsqueeze(0).contiguous().view(1, -1, len_v)  # 1 x (b * dv) x lv

        if cached_attn is None:
            q = self.w(q.unsqueeze(0))
            k = k.unsqueeze(0)

            output, attn = self.attention(q, k, v)
        else:
            output, attn = self.attention(None, None, v, cached_attn=cached_attn)

        output = output.view(sz_b, d_v, len_q)
        output = output.permute(0, 2, 1)  # b x lq x dv

        return output, attn


class FixedInputNeuronPool(nn.Module):
    def __init__(self, n_neurons, embedding_dim,
                 transformation=("PointwiseLinear", "nn.Dropout(p=0.2)")):
        super(FixedInputNeuronPool, self).__init__()

        # Set instance parameters
        self.n_neurons = n_neurons
        self.embedding_dim = embedding_dim

        # Define embeddings of axons and dendrites
        self.input_axon_embeddings = nn.Parameter(torch.empty(n_neurons, embedding_dim))

        # Define transformations
        self.transformation = get_transformation_module(transformation, [n_neurons, ])

        # Set initial values of module parameters
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.input_axon_embeddings, mode='fan_out')

    def forward(self, input_pair):
        inputs, _ = input_pair
        inputs = self.transformation(inputs)

        return inputs, self.input_axon_embeddings + 0.

    def extra_repr(self):
        return 'n_neurons={n_neurons}, embedding_dim={embedding_dim},\n'.format(**self.__dict__) + \
               'input_axon_embeddings=Tensor({})'.format(self.input_axon_embeddings.shape)


@six.add_metaclass(abc.ABCMeta)
class AbstractNeuronPoolWithSynapse(nn.Module):
    def __init__(self, n_neurons, embedding_dim, synaptic_module, transformation, cache_enabled=False):

        super(AbstractNeuronPoolWithSynapse, self).__init__()

        # Set instance parameters
        self.n_neurons = n_neurons
        self.embedding_dim = embedding_dim
        self.cache_enabled = cache_enabled

        # Define embeddings of axons and dendrites
        self.dendrite_embeddings = nn.Parameter(torch.empty(n_neurons, embedding_dim))

        # Define transformations and activations
        self.transformation = get_transformation_module(transformation, [n_neurons, ])

        # Define synaptic module that connects dendrites to axons
        self.synaptic_module = synaptic_module

        # Define caching mechanism for synaptic state
        if self.cache_enabled:
            self.synaptic_state_cache = None

            def clear_synaptic_state_cache(*args):
                self.synaptic_state_cache = None

            self.clear_synaptic_state_cache = clear_synaptic_state_cache
            self.dendrite_embeddings.register_hook(self.clear_synaptic_state_cache)
            self.cached_input_embeddings = None

        # Set initial values of module parameters
        self._reset_dendrite_parameters()

    def _reset_dendrite_parameters(self):
        nn.init.kaiming_uniform_(self.dendrite_embeddings, mode='fan_in')

    @abc.abstractmethod
    def get_axon_embeddings(self):
        pass

    def forward(self, input_pair):
        inputs, input_embeddings = input_pair

        synaptic_state_cache = None

        if self.cache_enabled:
            if self.cached_input_embeddings is not None and \
                    torch.equal(self.cached_input_embeddings, input_embeddings):
                synaptic_state_cache = self.synaptic_state_cache

        dendrite_pulse, self.synaptic_state_cache = self.synaptic_module(
            q=self.dendrite_embeddings,
            k=input_embeddings,
            v=inputs.unsqueeze(-1),
            cached_attn=synaptic_state_cache
        )
        self.cached_input_embeddings = input_embeddings if self.cache_enabled else None

        dendrite_pulse = dendrite_pulse.squeeze(-1)

        axon_outputs = self.transformation(dendrite_pulse)

        return axon_outputs, self.get_axon_embeddings()

    def extra_repr(self):
        return 'n_neurons={n_neurons}, embedding_dim={embedding_dim},\n'.format(**self.__dict__) + \
               'dendrite_embeddings=Tensor({})'.format(self.dendrite_embeddings.shape)


class FixedOutputNeuronPool(AbstractNeuronPoolWithSynapse):
    def __init__(self, n_neurons, embedding_dim, synaptic_module=None,
                 transformation="PointwiseLinear", cache_enabled=False):

        super(FixedOutputNeuronPool, self).__init__(
            n_neurons=n_neurons, embedding_dim=embedding_dim, synaptic_module=synaptic_module,
            transformation=transformation, cache_enabled=cache_enabled)

    def get_axon_embeddings(self):
        return None


class NeuronPool(AbstractNeuronPoolWithSynapse):
    def __init__(self, n_neurons, embedding_dim, synaptic_module,
                 transformation=("PointwiseLinear", "nn.LeakyReLU(0.2)", "PointwiseLinear"),
                 axon_dendrite_linked_embedding=False, cache_enabled=False):

        super(NeuronPool, self).__init__(
            n_neurons=n_neurons, embedding_dim=embedding_dim, synaptic_module=synaptic_module,
            transformation=transformation, cache_enabled=cache_enabled)

        self.axon_dendrite_linked_embedding = axon_dendrite_linked_embedding

        # Define embeddings of axons
        if not self.axon_dendrite_linked_embedding:
            self.axon_terminal_embeddings = nn.Parameter(torch.empty(n_neurons, embedding_dim))
            if self.cache_enabled:
                self.axon_terminal_embeddings.register_hook(self.clear_synaptic_state_cache)
        else:
            self.axon_terminal_embeddings = self.dendrite_embeddings

        # Set initial values of module parameters
        self._reset_axon_parameters()

    def _reset_axon_parameters(self):
        if not self.axon_dendrite_linked_embedding:
            nn.init.kaiming_uniform_(self.axon_terminal_embeddings, mode='fan_in')

    def get_axon_embeddings(self):
        return self.axon_terminal_embeddings + 0.

    def extra_repr(self):
        return 'n_neurons={n_neurons}, embedding_dim={embedding_dim}, ' \
               'axon_dendrite_linked_embedding={axon_dendrite_linked_embedding},\n'.format(**self.__dict__) + \
               'axon_terminal_embeddings=Tensor({})\n'.format(self.axon_terminal_embeddings.shape) + \
               'dendrite_embeddings=Tensor({})'.format(self.dendrite_embeddings.shape)

