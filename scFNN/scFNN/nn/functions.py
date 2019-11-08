import math
import numpy as np
import torch
import torch.nn.functional as F

from scFNN.general.conf import settings


def log_eps(x):
    epsilon = settings.EPSILON

    return torch.log(x + epsilon)


def lgamma_eps(x):
    epsilon = settings.EPSILON

    return torch.lgamma(x + epsilon)


def frac_eps(x, y):
    epsilon = settings.EPSILON

    return x / (y + epsilon)


def softplus_eps(x):
    epsilon = settings.EPSILON

    return F.softplus(x + epsilon)


def negative_binomial(k, m, r):
    # choice_part = log(binom(k, k+r-1))
    choice_part = lgamma_eps(k + r) - lgamma_eps(k + 1) - lgamma_eps(r)
    # log_m_r = log(m+r)
    log_m_r = log_eps(m + r)
    # log_pow_k = log(p ^ k) = log((m/(m+r)) ^ k)
    log_pow_k = k * (log_eps(m) - log_m_r)
    # log_pow_r = log((1 - p) ^ r) = log((r/(m+r)) ^ r)
    log_pow_r = r * (log_eps(r) - log_m_r)

    return choice_part + log_pow_k + log_pow_r


def negative_binomial_log_ver(k, m_log, r_log):
    # r :D
    r = torch.exp(r_log)

    # choice_part = log(binom(k, k+r-1))
    choice_part = lgamma_eps(k + r) - lgamma_eps(k + 1) - lgamma_eps(r)
    # log_pow_k = log(p ^ k) = log((m/(m+r)) ^ k)
    log_pow_k = - k * softplus_eps(r_log - m_log)
    # log_pow_r = log((1 - p) ^ r) = log((r/(m+r)) ^ r)
    log_pow_r = - r * softplus_eps(m_log - r_log)

    return choice_part + log_pow_k + log_pow_r


def zero_inflated_negative_binomial(k, m, r, pi_logit):
    zero_threshold = 1e-4

    # choice_part = log(binom(k, k+r-1))
    choice_part = lgamma_eps(k + r) - lgamma_eps(k + 1) - lgamma_eps(r)
    # log_m_r = log(m+r)
    log_m_r = log_eps(m + r)
    # log_pow_k = log(p ^ k) = log((m/(m+r)) ^ k)
    log_pow_k = k * (log_eps(m) - log_m_r)
    # log_pow_r = log((1 - p) ^ r) = log((r/(m+r)) ^ r)
    log_pow_r = r * (log_eps(r) - log_m_r)

    # log_pi = log(pi)
    log_pi = - softplus_eps(- pi_logit)
    # log_1_minus_pi = log(1-pi)
    log_1_minus_pi = - softplus_eps(pi_logit)
    # log_zero = log(pi + (1 - pi) * (r/(m+r)) ^ r)
    zero_part = log_pi + softplus_eps(- pi_logit + log_pow_r)

    # negative_binom = negative_binomial(k, m, r)
    negative_binom_part = log_1_minus_pi + choice_part + log_pow_k + log_pow_r

    reconstruction_loss = torch.where(k < zero_threshold, zero_part, negative_binom_part)

    return reconstruction_loss


def zero_inflated_negative_binomial_log_version(k, m_log, r_log, pi_logit):
    zero_threshold = 1e-4

    assert not (m_log != m_log).any()
    assert not (r_log != r_log).any()
    assert not (pi_logit != pi_logit).any()

    # r :D
    r = torch.exp(r_log)

    # choice_part = log(binom(k, k+r-1))
    choice_part = lgamma_eps(k + r) - lgamma_eps(k + 1) - lgamma_eps(r)
    # log_pow_k = log(p ^ k) = log((m/(m+r)) ^ k)
    log_pow_k = - k * softplus_eps(r_log - m_log)
    # log_pow_r = log((1 - p) ^ r) = log((r/(m+r)) ^ r)
    log_pow_r = - r * softplus_eps(m_log - r_log)
    # TODO: use logsigmoid function instead of softplus
    # log_pi = log(pi)
    log_pi = - softplus_eps(- pi_logit)
    # log_1_minus_pi = log(1-pi)
    log_1_minus_pi = - softplus_eps(pi_logit)
    # log_zero = log(pi + (1 - pi) * (r/(m+r)) ^ r)
    zero_part = log_pi + softplus_eps(- pi_logit + log_pow_r)

    # negative_binom = negative_binomial(k, m, r)
    negative_binom_part = log_1_minus_pi + choice_part + log_pow_k + log_pow_r

    reconstruction_loss = torch.where(k < zero_threshold, zero_part, negative_binom_part)

    return reconstruction_loss


def cross_entropy_of_two_gaussians(mu_p, log_sigma_2_p, mu_q, log_sigma_2_q):
    return 1 / 2 * (math.log(2 * math.pi) + 2 * log_sigma_2_q +
                    torch.exp(2 * (log_sigma_2_p - log_sigma_2_q)) +
                    frac_eps((mu_p - mu_q) ** 2, torch.exp(2 * log_sigma_2_q)))


def kullback_leibler_of_two_gaussians(mu_p, log_sigma_p, mu_q, log_sigma_q):
    return 1 / 2 * (2 * (log_sigma_q - log_sigma_p) +
                    torch.exp(log_sigma_p - log_sigma_q) +
                    frac_eps((mu_p - mu_q) ** 2, torch.exp(2 * log_sigma_q)) - 1)


def entropy_of_gaussian(log_sigma):
    return 1 / 2 * (1 + math.log(2 * math.pi) + 2 * log_sigma)


def log_p_normal(loc, log_scale, observation):
    return - frac_eps((observation - loc) ** 2, 2 * torch.exp(log_scale) ** 2)\
           - log_scale - 0.5 * math.log(2 * math.pi)


def col_sums(data_loader, dtype="numpy"):
    output = None
    for batch_data in data_loader:
        data = {"X": batch_data[0]}

        sums = data.sum(dim=0)

        if output is None:
            output = sums
        else:
            output += sums

    if dtype == "numpy":
        return output.cpu().numpy()
    else:
        return output


def row_sums(data_loader, data_index=0, dtype="numpy"):
    output = None
    for batch_data in data_loader:
        data = batch_data[data_index]

        sums = data.sum(dim=1)

        if output is None:
            output = sums
        else:
            output = torch.cat([output, sums])

    if dtype == "numpy":
        return output.cpu().numpy()
    else:
        return output


def pandas_col_sums(data_loader):
    output = None
    for batch_data in data_loader:
        data = {"X": batch_data[0]}

        sums = data.sum(axis=0).values

        if output is None:
            output = sums
        else:
            output += sums

    return output


def pandas_row_sums(data_loader, data_index=0):
    output = None
    for batch_data in data_loader:
        data = batch_data[data_index]

        sums = data.sum(axis=1).values

        if output is None:
            output = sums
        else:
            output = np.concatenate([output, sums])


    return output