import os
import argparse
import pandas as pd

from model import SAUCIE
from loader import Loader


def parse_args():
    p = argparse.ArgumentParser(description='SAUCIE', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('-i', '--input', metavar='FILE', default='input_count.csv', type=str,
                   help='File path of input data file.')
    p.add_argument('-o', '--outputdir', metavar='DIR', default='/output', type=str,
                   help='Directory which output should be stored in.')

    p.add_argument('-s', '--steps', default=1000, type=int,
                   help='Number of iterations.')
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
print("run with these parametres: %s" % str(args))

# Main Part

X = pd.read_csv(args.input, index_col=0)
data = X.values.T

saucie = SAUCIE(data.shape[1])
loadtrain = Loader(data, shuffle=True)
saucie.train(loadtrain, steps=args.steps)

loadeval = Loader(data, shuffle=False)
embedding = saucie.get_embedding(loadeval)
number_of_clusters, clusters = saucie.get_clusters(loadeval)
reconstruction = saucie.get_reconstruction(loadeval)

X_emb = pd.DataFrame(embedding, index=X.columns.values, columns=["D1", "D2"])
X_clusters = pd.DataFrame(clusters, index=X.columns.values, columns=["cluster"])
X_imp = pd.DataFrame(reconstruction.T, index=X.index.values, columns=X.columns.values)

make_sure_dir_exists(args.outputdir)
X_emb.to_csv(os.path.join(args.outputdir, "embedding.csv"))
X_clusters.to_csv(os.path.join(args.outputdir, "clusters.csv"))
X_imp.to_csv(os.path.join(args.outputdir, "saucie_output.csv"))

print("Done!")
