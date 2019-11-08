import numpy as np


normalizations = {
    "none": lambda x: x,
    "l1": lambda x: 1e6 * x.divide(x.sum(axis=0), axis=1),
    "l2": lambda x: 1e6 * x.divide((x ** 2).sum(axis=0) ** 0.5, axis=1)
}


transformations = {  # Zero Should be mapped to zero
    "none": lambda x: x,
    "log": lambda x: np.log2(1 + x),
    "sqrt": lambda x: np.sqrt(x) + np.sqrt(x + 1) - 1
}


def shuffle_and_rename_columns(data, prefix="cell", disabled=False):
    original_columns = data.columns.values

    if not disabled:
        column_permutation = np.random.permutation(range(len(data.columns.values)))
    else:
        column_permutation = range(len(data.columns.values))
    permuted_data = data[data.columns.values[column_permutation]]

    if not disabled:
        permuted_data.columns = ["%s_%d" % (prefix, i) for i in range(len(original_columns))]

    return permuted_data, original_columns, column_permutation


def rearrange_and_rename_columns(data, original_columns, column_permutation):
    reverse_permutation = np.argsort(column_permutation)

    rearranged_data = data[data.columns.values[reverse_permutation]]
    rearranged_data.columns = original_columns

    return rearranged_data
