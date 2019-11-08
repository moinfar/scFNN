import os
import argparse

import numpy as np
import pandas as pd

from uncurl import max_variance_genes, run_state_estimation


def parse_args():
    parser = argparse.ArgumentParser(description='A docker wrapper around UNCURL.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--input", type=str, default="input_count.csv", help="input count file to be imputed", metavar="FILE")
    parser.add_argument("-o", "--outputdir", type=str, default="/output", help="directory which output data will be stored in", metavar="FILE")
    parser.add_argument("--gene-subset", type=str, default="non_zero", help="subset of genes used in algorithm (non_zero/max_variance)", metavar="STRING")
    parser.add_argument("--clusters", type=int, default=10, help="number of clusters", metavar="INT")
    parser.add_argument("--dist", type=str, default="Poiss", help="distribution model (Poiss/LogNorm/Gaussian/NB)", metavar="STRING")
    parser.add_argument("--max-iters", type=int, default=30, help="max_iters parameter", metavar="INT")
    parser.add_argument("--inner-max-iters", type=int, default=100, help="inner_max_iters parameter", metavar="INT")
    parser.add_argument("--initialization", type=str, default="tsvd", help="initialization", metavar="STRING")
    parser.add_argument("--threads", type=int, default=8, help="number of threads", metavar="INT")

    return parser.parse_args()


def make_sure_dir_exists(dire_name):
    import errno

    try:
        os.makedirs(dire_name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e


args = parse_args()
print("run with these parametres: %s" % str(args))

data = pd.read_csv(args.input, index_col=0)

if args.gene_subset == 'non_zero':
    genes_subset = np.sum(data.values, axis=1) != 0  # select nonzero genes
elif args.gene_subset == 'max_variance':
    genes_subset = max_variance_genes(data.values, nbins=5, frac=0.2) # select genes with max variance
else:
    raise NotImplementedError("optin `%s` for `gene_subset` not defined." % args.gene_subset)

data_subset = data.iloc[genes_subset,:]
M, W, ll = run_state_estimation(data_subset.values, clusters=args.clusters,
                                dist=args.dist, disp=True,
                                max_iters=args.max_iters,
                                inner_max_iters=args.inner_max_iters,
                                initialization=args.initialization,
                                threads=args.threads)

print("ll: %f" % ll)

data.iloc[genes_subset, :] = np.matmul(M, W) # imputation

make_sure_dir_exists(args.outputdir)

np.savetxt("genes_subset.csv", genes_subset, delimiter=",")
np.savetxt("M.csv", M, delimiter=",")
np.savetxt("W.csv", W, delimiter=",")

data.to_csv(os.path.join(args.outputdir, "uncurl_output.csv"))
