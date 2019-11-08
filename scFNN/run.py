import argparse
import atexit
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split, DataLoader
from torchviz import make_dot

from scFNN.data.io import write_csv
from scFNN.data.torch_loader import CsvFileDataSet, PredefinedDataSet
from scFNN.general.conf import settings
from scFNN.general.utils import log, dump_gzip_pickle, load_gzip_pickle
from scFNN.general.utils import make_sure_dir_exists
from scFNN.nn.networks import FNNAutoEncoder, AbstractNetwork
from scFNN.nn.training import get_output, test
from scFNN.nn.training import run

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_parser():
    main_parser = argparse.ArgumentParser(description="CLI for testing FNN single-cell analyzer.")
    main_parser.set_defaults(default_function=main_parser.print_help)

    # Run Parameters
    main_parser.add_argument('data_set', metavar='DATASET', type=str,
                             help='Dataset to train network on (either an address or a dataset name)')

    main_parser.add_argument('output_dir', metavar='OUTDIR', type=str,
                             help='Address to put output files in')

    main_parser.add_argument('--gene-emb-dim', '-g', metavar='N', type=int,
                             default=256, help='Embedding Dimension of genes')

    main_parser.add_argument('--encoder-sizes', metavar='N', type=str,
                             default="1024,128,64", help='Number of hidden neurons in encoder')

    main_parser.add_argument('--decoder-sizes', metavar='N', type=str,
                             default="128,1024", help='Number of hidden neurons in decoder')

    main_parser.add_argument('--norm', '-n', metavar='NORM', type=str,
                             default="l2", choices=['identity', 'l1', 'l2', 'softmax'],
                             help='Attention normalization function')

    main_parser.add_argument('--act', '-a', metavar='ACT', type=str,
                             default="nn.ELU()", help='Hidden activation function')

    main_parser.add_argument('--loss', '-l', metavar='LOSS', type=str,
                             default="zinb", help='Loss for last layer')

    main_parser.add_argument('--dropout', '-d', metavar='%', type=float,
                             default=0., help='Input dropout probability')

    main_parser.add_argument('--org', '-o', metavar='ORGANISM', type=str,
                             default="human", choices=['human', 'mouse'],
                             help='Organism under study')

    main_parser.add_argument('--upstream', '-u', metavar='FILE', type=str,
                             default=None, help='Model used for initializing embedding values')

    main_parser.add_argument('--no-log-ver', action="store_false",
                             help='Add if activation of mean and r are exp-softplus instead of exp-exp')

    main_parser.add_argument('--no-lib-norm', action="store_false",
                             help='Add if you want to disable library size normalization')

    main_parser.add_argument('--n-epochs', '-N', metavar='N', type=int,
                             default=100, help='Number of epochs the network is trined')

    main_parser.add_argument('--opt', '-O', metavar='OPT', type=str, default='adam',
                             choices=['adam', 'sgd', 'rmsprop'], help='Optimization algorithm')

    main_parser.add_argument('--lr', '-L', metavar='LR', type=float, default=0.01,
                             help='Learning rate of the optimizer')

    main_parser.add_argument('--batch-size', '-B', metavar='BS', type=int, default=32,
                             help='Batch size')

    main_parser.add_argument('--device', '-D', metavar='DEVICE', type=str,
                             default=("cuda" if torch.cuda.is_available() else "cpu"),
                             help='Device on which the model is trained')

    main_parser.add_argument('--seed', '-S', metavar='SEED', type=int, default=-1,
                             help='Random Seed (set -1 to disable -- default = -1)')

    main_parser.add_argument('--ignore-output', action="store_true",
                             help='Add if you want to ignore output and save model only')

    main_parser.add_argument('--ignore-model', action="store_true",
                             help='Add if you want to ignore saving the model')

    main_parser.add_argument('--evaluate-only', action="store_true",
                             help='Add if you want to only evaluate a model')

    main_parser.add_argument('--debug', action="store_true",
                             help='Keeps IPython interpreter running after work')

    main_parser.add_argument('--mean-only', action="store_true",
                             help='Only saves mean.csv.gz')

    return main_parser


if __name__ == '__main__':
    # Parsing
    parser = generate_parser()
    args = parser.parse_args()

    print("Running with arguments:")
    for key in vars(args):
        if key == "default_function":
            continue
        print(key, ": ", getattr(args, key))
    print("---")

    # Reproducibility options
    if args.seed != -1:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True

    org = args.org if hasattr(args, "org") else None

    all_gene_ids = None

    # Loading dataset
    if args.data_set in settings.data_sets.keys():
        data_set = PredefinedDataSet(args.data_set, ref=all_gene_ids)
    else:
        data_set = CsvFileDataSet(args.data_set, ref=all_gene_ids)

    train_dataset, validation_dataset = random_split(data_set, [len(data_set) - int(0.1 * len(data_set)),
                                                                int(0.1 * len(data_set))])

    def collate_function(batch_list):
        result = tuple([pd.concat(batch_entry, axis=0)
                        for batch_entry in zip(*batch_list)])
        return result

    # Data Loader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=collate_function,
        shuffle=True,
        drop_last=False)
    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=collate_function,
        shuffle=False,
        drop_last=False)

    model = FNNAutoEncoder(
        input_dim=data_set.n_genes,
        input_keys=data_set.genes,
        gene_embedding_dim=args.gene_emb_dim,
        encoder_sizes=eval("[%s]" % args.encoder_sizes),
        decoder_sizes=eval("[%s]" % args.decoder_sizes),
        learning_rate=args.lr,
        early_stopping_patience=10,
        dropout_rate=args.dropout,
        attention_normalization=args.norm,
        hidden_activation=args.act, reconstruction_loss=args.loss,
        log_ver=args.no_log_ver, library_size_normalization=args.no_lib_norm)

    if args.upstream is not None:
        full_state = load_gzip_pickle(args.upstream)
        upstream = AbstractNetwork.load_from_full_state(full_state)

        print("{} genes in common with upstream".format(len(np.intersect1d(model.input_keys, upstream.input_keys))))

        model.input_net.init_from_another_module(upstream.input_net)
        if model.linked_zinb_parameters:
            model.output.init_from_another_module(upstream.output)
        else:
            model.output_mean.init_from_another_module(upstream.output_mean)
            model.output_r.init_from_another_module(upstream.output_r)
            model.output_pi.init_from_another_module(upstream.output_pi)

        del upstream
        del full_state

    print("Training model: ", model)

    data = {"X": torch.from_numpy(pd.concat([data_set[0][0], data_set[1][0], data_set[2][0]]).values),
            "keys": data_set[0][0].columns.values}
    dot = make_dot(model.loss(data, model(data))["loss"], params=dict(model.named_parameters()))
    dot.format = 'svg'
    dot.render("output.svg")

    # CUDA options
    if args.device == "cuda":
        model.cuda()

    # Define save procedure
    @atexit.register
    def save_model_and_outputs():
        args.output_dir = os.path.abspath(args.output_dir)
        make_sure_dir_exists(args.output_dir)
        log("Saving output results to {}".format(args.output_dir))

        if not args.ignore_model:
            dump_gzip_pickle(model.get_full_state_dict(), os.path.join(args.output_dir, "model.pkl.gz"))

        if not args.ignore_output:
            data_loader = DataLoader(
                dataset=data_set,
                batch_size=args.batch_size,
                num_workers=4,
                collate_fn=collate_function,
                shuffle=False,
                drop_last=False)

            all_outputs = get_output(model, data_loader)
            if args.mean_only:
                features = ["mean"]
            for feature in ["mean", "r", "pi", "normal_mean_nodes"]:
                if feature in all_outputs:
                    print("Saving %s ..." % feature)
                    output = all_outputs[feature]
                    output = pd.DataFrame(output.transpose(), index=data_set.genes, columns=data_set.cells)
                    output = output.round(3)
                    write_csv(output, os.path.join(args.output_dir, "{}.csv.gz".format(feature)))

        if args.debug:
            import IPython
            IPython.embed()

    # Training
    if not args.evaluate_only:
        run(model, args.n_epochs, train_loader, validation_loader)
    else:
        data_loader = DataLoader(
            dataset=data_set,
            batch_size=args.batch_size,
            num_workers=4,
            collate_fn=collate_function,
            shuffle=False,
            drop_last=False)

        test(model, data_loader)
