import os
import magic
import argparse
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description='MAGIC', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('-i', '--input', metavar='FILE', default='input_count.csv', type=str,
                   help='File path of input data file.')
    p.add_argument('-o', '--outputdir', metavar='DIR', default='/output', type=str,
                   help='Directory which output should be stored in.')

    p.add_argument('-k', default=10, type=int,
                   help='number of nearest neighbors on which to build kernel.')
    p.add_argument('-a', default=15, type=int,
                   help='sets decay rate of kernel tails.')
    p.add_argument('-t', default="auto", type=str,
                   help='power to which the diffusion operator is powered.')
    p.add_argument('--n-pca', default=100, type=int,
                   help='Number of principal components to use for calculating neighborhoods.')
    p.add_argument('--knn-dist', default="euclidean", type=str,
                   help='Any metric from scipy.spatial.distance.')
    p.add_argument('--n-jobs', default=1, type=int,
                   help='The number of jobs to use for the computation. If -1 all CPUs are used.')

    try:
        return p.parse_args()
    except ArgumentParserError:
        raise


def make_sure_dir_exists(dire_name):
    import errno

    try:
        os.makedirs(dire_name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e


args = parse_args()
if args.t != "auto":
    args.t = int(args.t)
print("run with these parametres: %s" % str(args))


# Main Part

X = pd.read_csv(args.input, index_col=0)
X = X.transpose()

magic_operator = magic.MAGIC(k=args.k, a=args.a, t=args.t, n_pca=args.n_pca,
                             knn_dist=args.knn_dist, n_jobs=args.n_jobs)

X_magic = magic_operator.fit_transform(X, genes="all_genes")
X_magic = X_magic.transpose()

make_sure_dir_exists(args.outputdir)
X_magic.to_csv(os.path.join(args.outputdir, "magic_output.csv"))
