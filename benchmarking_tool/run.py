import argparse
from general.conf import settings

from main import generate_cell_cycle_test, handle_main_arguments, evaluate_random_mask_test, \
    evaluate_cell_cycle_test, generate_random_mask_test, generate_down_sample_test, evaluate_down_sample_test, \
    generate_clustering_test, evaluate_clustering_test, evaluate_paired_data_test, generate_paired_data_test, \
    evaluate_cite_seq_test, generate_cite_seq_test, visualize_cell_cycle_evaluation, visualize_clustering_evaluation, \
    visualize_random_mask_evaluation, visualize_down_sample_evaluation, visualize_paired_data_evaluation, \
    visualize_cite_seq_evaluation
from utils.base import log


def generate_parser():
    main_parser = argparse.ArgumentParser(description="A benchmarking suite to evaluate "
                                                      "single-cell RNA-seq imputation algorithms.")

    main_parser.set_defaults(default_function=main_parser.print_help)
    main_parser.add_argument('id', metavar='ID', type=str,
                             help='unique ID to identify current benchmark.')

    # Define sub commands
    subparsers = main_parser.add_subparsers(help='action to perform')
    parser_generate = subparsers.add_parser('generate', help='generate a count file to impute')
    subparsers_generate = parser_generate.add_subparsers(help='type of benchmark')
    parser_evaluate = subparsers.add_parser('evaluate', help='evaluate an imputed count file')
    subparsers_evaluate = parser_evaluate.add_subparsers(help='type of benchmark')
    parser_visualize = subparsers.add_parser('visualize', help='visualize an already evaluated experiment')
    subparsers_visualize = parser_visualize.add_subparsers(help='type of benchmark')

    # Define global arguments
    main_parser.add_argument('--seed', '-S', metavar='N', type=int,
                             help='Seed for random generator (random if not provided)')
    main_parser.add_argument('--debug', '-D', action='store_true',
                             help='Prints debugging info')

    # Define generate commands
    parser_generate.set_defaults(default_function=parser_generate.print_help)
    parser_generate.add_argument('--output', '-o', metavar='COUNT_FILE',
                                 type=str, required=True,
                                 help='Address where noisy count matrix will be stored in')
    parser_generate.add_argument('--preserve-columns', '-p', action='store_true',
                                 help='Preserve column orders and labels if using a real dataset.')

    parser_generate_cell_cycle = subparsers_generate.add_parser('cell-cycle')
    parser_generate_cell_cycle.set_defaults(function=generate_cell_cycle_test)
    parser_generate_cell_cycle.add_argument('--rm-ercc', action='store_true',
                                            help='Remove ERCC rows from generated expression table.')
    parser_generate_cell_cycle.add_argument('--rm-mt', action='store_true',
                                            help='Remove mitochondrial genes from generated expression table.')
    parser_generate_cell_cycle.add_argument('--rm-lq', action='store_true',
                                            help='Remove low quality cells.')

    parser_generate_clustering = subparsers_generate.add_parser('clustering')
    parser_generate_clustering.set_defaults(function=generate_clustering_test)
    parser_generate_clustering.add_argument('--data-set', '-d', metavar='DATASET',
                                            type=str, default='CORTEX_3005',
                                            help='Dataset to be used in this benchmark')

    parser_generate_random_mask = subparsers_generate.add_parser('random-mask')
    parser_generate_random_mask.set_defaults(function=generate_random_mask_test)
    parser_generate_random_mask.add_argument('--data-set', '-d', metavar='DATASET',
                                             type=str, default='10xPBMC4k',
                                             help='Dataset to be used in this benchmark')
    parser_generate_random_mask.add_argument('--dropout-count', '-c', metavar='N',
                                             type=int, required=True,
                                             help='Number of dropouts to introduce')
    parser_generate_random_mask.add_argument('--min-expression', '-m', metavar='M',
                                             type=int, default=10,
                                             help='Minimum expression for an entry to be dropped')
    parser_generate_random_mask.add_argument('--hvg-frac', '-f', metavar='F',
                                             type=float, default=0.05,
                                             help='Fraction of genes that are considered as HVG '
                                                  '(entries from non-HVG gees will not be dropped)')
    parser_generate_random_mask.add_argument('--n-samples', '-n', metavar='N',
                                             type=int, default=0,
                                             help='Number of samples (cells) used from dataset'
                                                  '(Enter 0 to use all samples.)')

    parser_generate_down_sample = subparsers_generate.add_parser('down-sample')
    parser_generate_down_sample.set_defaults(function=generate_down_sample_test)
    parser_generate_down_sample.add_argument('--data-set', '-d', metavar='DATASET',
                                             type=str, default='10xPBMC4k',
                                             help='Dataset to be used in this benchmark')
    parser_generate_down_sample.add_argument('--read-ratio', '-r', metavar='RATIO',
                                             type=float, required=True,
                                             help='Ratio of reads compared to original dataset')
    parser_generate_down_sample.add_argument('--n-samples', '-n', metavar='N',
                                             type=int, default=0,
                                             help='Number of samples (cells) used from dataset'
                                                  '(Enter 0 to use all samples.)')
    parser_generate_down_sample.add_argument('--replace', action='store_true',
                                             help='Add in case of random replacement')

    parser_generate_paired_data = subparsers_generate.add_parser('paired-data')
    parser_generate_paired_data.set_defaults(function=generate_paired_data_test)
    parser_generate_paired_data.add_argument('--data-set', '-d', metavar='DS',
                                             type=str, default='SRP041736',
                                             help='Dataset which has paired data (DS-HQ and DS-LQ must be implemented)')

    parser_generate_cite_seq = subparsers_generate.add_parser('cite-seq')
    parser_generate_cite_seq.set_defaults(function=generate_cite_seq_test)
    parser_generate_cite_seq.add_argument('--data-set', '-d', metavar='DS',
                                          type=str, default='CITE-CBMC',
                                          help='Dataset which has RNA and ADT data (must have RNA and ADT keys)')

    # Define evaluate commands
    parser_evaluate.set_defaults(default_function=parser_evaluate.print_help)
    parser_evaluate.add_argument('--input', '-i', metavar='IMPUTED_COUNT_FILE',
                                 type=str, required=True,
                                 help='Address of file containing imputed count matrix')
    parser_evaluate.add_argument('--result-dir', '-r', metavar='RESULT_DIR',
                                 type=str, required=True,
                                 help='The directory where the evaluation results will be stored in')
    parser_evaluate.add_argument('--visualization', '-v', choices=['none', 'pdf', 'html'], default='none',
                                 help='Plotting format.')

    parser_evaluate_cell_cycle = subparsers_evaluate.add_parser('cell-cycle')
    parser_evaluate_cell_cycle.set_defaults(function=evaluate_cell_cycle_test)
    parser_evaluate_cell_cycle.add_argument('--normalization', "-n", choices=['none', 'l1', 'l2'], default='l1',
                                            help='Normalization to be applied before transformation and evaluation.')
    parser_evaluate_cell_cycle.add_argument('--transformation', "-t", choices=['none', 'log', 'sqrt'], default='log',
                                            help='Transformation to be applied before evaluation.')
    parser_evaluate_cell_cycle.add_argument('--clear-cache', '--cc', action='store_true',
                                            help='Clear embedding cache if available.')

    parser_evaluate_clustering = subparsers_evaluate.add_parser('clustering')
    parser_evaluate_clustering.set_defaults(function=evaluate_clustering_test)
    parser_evaluate_clustering.add_argument('--normalization', "-n", choices=['none', 'l1', 'l2'], default='l1',
                                            help='Normalization to be applied before transformation and evaluation.')
    parser_evaluate_clustering.add_argument('--transformation', "-t", choices=['none', 'log', 'sqrt'], default='log',
                                            help='Transformation to be applied before evaluation.')
    parser_evaluate_clustering.add_argument('--clear-cache', '--cc', action='store_true',
                                            help='Clear embedding cache if available.')

    parser_evaluate_random_mask = subparsers_evaluate.add_parser('random-mask')
    parser_evaluate_random_mask.set_defaults(function=evaluate_random_mask_test)

    parser_evaluate_down_sample = subparsers_evaluate.add_parser('down-sample')
    parser_evaluate_down_sample.set_defaults(function=evaluate_down_sample_test)
    parser_evaluate_down_sample.add_argument('--transformation', "-t", choices=['none', 'log', 'sqrt'], default='log',
                                             help='Transformation to be applied before evaluation.')

    parser_evaluate_paired_data = subparsers_evaluate.add_parser('paired-data')
    parser_evaluate_paired_data.set_defaults(function=evaluate_paired_data_test)
    parser_evaluate_paired_data.add_argument('--normalization', "-n", choices=['none', 'l1', 'l2'], default='l1',
                                             help='Normalization to be applied before transformation and evaluation.')
    parser_evaluate_paired_data.add_argument('--transformation', "-t", choices=['none', 'log', 'sqrt'], default='log',
                                             help='Transformation to be applied before evaluation.')

    parser_evaluate_cite_seq = subparsers_evaluate.add_parser('cite-seq')
    parser_evaluate_cite_seq.set_defaults(function=evaluate_cite_seq_test)
    parser_evaluate_cite_seq.add_argument('--transformation', "-t", choices=['none', 'log', 'sqrt'], default='log',
                                          help='Transformation to be applied before evaluation.')

    # Define visualization commands
    parser_visualize.set_defaults(default_function=parser_visualize.print_help)
    parser_visualize.add_argument('--result-dir', '-r', metavar='RESULT_DIR',
                                  type=str, required=True,
                                  help='The directory where the evaluation results are stored in')
    parser_visualize.add_argument('--type', '-t', choices=['none', 'pdf', 'html'], default='html',
                                  help='Plotting format.')

    parser_visualize_cell_cycle = subparsers_visualize.add_parser('cell-cycle')
    parser_visualize_cell_cycle.set_defaults(function=visualize_cell_cycle_evaluation)

    parser_visualize_clustering = subparsers_visualize.add_parser('clustering')
    parser_visualize_clustering.set_defaults(function=visualize_clustering_evaluation)

    parser_visualize_random_mask = subparsers_visualize.add_parser('random-mask')
    parser_visualize_random_mask.set_defaults(function=visualize_random_mask_evaluation)

    parser_visualize_down_sample = subparsers_visualize.add_parser('down-sample')
    parser_visualize_down_sample.set_defaults(function=visualize_down_sample_evaluation)

    parser_visualize_paired_data = subparsers_visualize.add_parser('paired-data')
    parser_visualize_paired_data.set_defaults(function=visualize_paired_data_evaluation)

    parser_visualize_cite_seq= subparsers_visualize.add_parser('cite-seq')
    parser_visualize_cite_seq.set_defaults(function=visualize_cite_seq_evaluation)

    return main_parser


if __name__ == '__main__':
    parser = generate_parser()
    args = parser.parse_args()

    if settings.DEBUG:
        log("Running with arguments: " + str(args))

    handle_main_arguments(args)

    if 'function' in args:
        args.function(args)
    else:
        args.default_function()
