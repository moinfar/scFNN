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

    p.add_argument('-d', '--latent-code-dim', default=50, type=int,
                   help='the feature dimension outputted by scScope.')
    p.add_argument('-T', default=2, type=int,
                   help='depth of recurrence used in deep learning framework.')
    p.add_argument('-b', '--batch-size', default=64, type=int,
                   help='number of cells used in each training iteration.')
    p.add_argument('-n', '--n-epochs', default=100, type=int,
                   help='number of epoch in training the NN.')
    p.add_argument('--lr', default=1e-3, type=float,
                   help='learning rate.')
    p.add_argument('--beta1', default=0.05, type=float,
                   help='learning rate.')
    p.add_argument('--no-mask', action='store_false',
                   help="flag indicating whether to use only (all/non-zero) entries in calculating losses.")

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
X = X.transpose()

import scscope as DeepImpute

DI_model = DeepImpute.train(
                        X.values,
                        args.latent_code_dim,
                        use_mask=not args.no_mask,
                        batch_size=args.batch_size,
                        max_epoch=args.n_epochs,
                        epoch_per_check=args.n_epochs + 1,
                        T=args.T,
                        exp_batch_idx_input=[],
                        encoder_layers=[],
                        decoder_layers=[],
                        learning_rate=args.lr,
                        beta1=args.beta1,
                        num_gpus=1)

latent_code, imputed_val, _ = DeepImpute.predict(X.values, DI_model)

make_sure_dir_exists(args.outputdir)
filename_latent=os.path.join(args.outputdir, "latent.csv")
filename_imputation=os.path.join(args.outputdir, "imputed_values.csv")

pd.DataFrame(latent_code.T, columns=X.index.values).to_csv(filename_latent)
pd.DataFrame(imputed_val.T, columns=X.index.values, index=X.columns.values).to_csv(filename_imputation)

print("Done!")
