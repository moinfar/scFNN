import os
import argparse
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description='MAGIC', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('-i', '--input', metavar='FILE', default='input_count.csv', type=str,
                   help='File path of input data file.')
    p.add_argument('-o', '--outputdir', metavar='DIR', default='/output', type=str,
                   help='Directory which output should be stored in.')

    p.add_argument('-n', '--n-epochs', default=400, type=int,
                   help='number of epoch in training the NN.')
    p.add_argument('--lr', default=1e-3, type=float,
                   help='learning rate.')
    p.add_argument('-s', '--train-size', default=0.75, type=float,
                   help='faction of data used in training the NN.')

    p.add_argument('--cuda', action='store_true')

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

from scvi.dataset.csv import GeneExpressionDataset
from scvi.models import VAE
from scvi.inference import UnsupervisedTrainer

X = pd.read_csv(args.input, index_col=0)
X = X.transpose()

dataset = GeneExpressionDataset(*GeneExpressionDataset.get_attributes_from_matrix(X.values, labels=None),
                                gene_names=X.columns.values, cell_types=None)
vae = VAE(dataset.nb_genes, n_batch=0)

trainer = UnsupervisedTrainer(vae,
                              dataset,
                              train_size=args.train_size,
                              use_cuda=args.cuda,
                              frequency=5)
trainer.train(n_epochs=args.n_epochs, lr=args.lr)

full = trainer.create_posterior(trainer.model, dataset, indices=np.arange(len(dataset)))
latent, _, _ = full.sequential().get_latent()
imputed_values = full.sequential().imputation()

make_sure_dir_exists(args.outputdir)
filename_latent=os.path.join(args.outputdir, "latent.csv")
filename_imputation=os.path.join(args.outputdir, "imputed_values.csv")

pd.DataFrame(latent.T, columns=X.index.values).to_csv(filename_latent)
pd.DataFrame(imputed_values.T, columns=X.index.values, index=X.columns.values).to_csv(filename_imputation)

print("Done!")
