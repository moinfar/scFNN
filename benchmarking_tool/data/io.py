import pandas as pd

from utils.base import load_gzip_pickle


def read_csv(filename, index_col=0, header=0):
    return pd.read_csv(filename, sep=None, index_col=index_col, header=header, engine="python")


def write_csv(data, filename):
    if filename.endswith(".csv"):
        data.to_csv(filename, sep=",", index_label="")
    elif filename.endswith(".tsv"):
        data.to_csv(filename, sep="\t", index_label="")
    elif filename.endswith(".csv.gz"):
        data.to_csv(filename, sep=",", index_label="", compression="gzip")
    elif filename.endswith(".tsv.gz"):
        data.to_csv(filename, sep="\t", index_label="", compression="gzip")
    else:
        raise NotImplementedError("Unrecognized format for file %s" % filename)


def read_table_file(filename):
    if filename.endswith(".csv") or filename.endswith(".tsv") or \
            filename.endswith(".csv.gz") or filename.endswith(".tsv.gz"):
        return read_csv(filename)
    elif filename.endswith(".pkl.gz"):
        return load_gzip_pickle(filename)
    else:
        raise NotImplementedError("Unrecognized format for file %s" % filename)
