from evaluators.biological import CellCyclePreservationEvaluator
from evaluators.paired_data import PairedLQHQDataEvaluator, CITESeqEvaluator
from evaluators.clustering import ClusteringEvaluator
from evaluators.numerical import RandomMaskedLocationPredictionEvaluator, DownSampledDataReconstructionEvaluator
from general.conf import settings
from utils.base import generate_seed


def handle_main_arguments(args):
    settings.DEBUG = args.debug
    args.seed = int(args.seed) if args.seed else generate_seed()


def print_metric_results(results):
    for metric in sorted(results.keys()):
        print("%s: %4f" % (metric, results[metric]))


def generate_cell_cycle_test(args):
    uid = "%s_cell_cycle" % args.id
    evaluator = CellCyclePreservationEvaluator(uid)
    evaluator.prepare()
    evaluator.set_seed(args.seed)
    evaluator.generate_test_bench(args.output, preserve_columns=args.preserve_columns,
                                  rm_ercc=args.rm_ercc, rm_mt=args.rm_mt, rm_lq=args.rm_lq)


def generate_clustering_test(args):
    uid = "%s_clustering" % args.id
    evaluator = ClusteringEvaluator(uid, args.data_set)
    evaluator.prepare()
    evaluator.set_seed(args.seed)
    evaluator.generate_test_bench(args.output, preserve_columns=args.preserve_columns)


def generate_random_mask_test(args):
    uid = "%s_random_mask" % args.id
    evaluator = RandomMaskedLocationPredictionEvaluator(uid, args.data_set)
    evaluator.prepare()
    evaluator.set_seed(args.seed)
    evaluator.generate_test_bench(args.output, preserve_columns=args.preserve_columns,
                                  n_samples=args.n_samples, dropout_count=args.dropout_count,
                                  min_expression=args.min_expression, hvg_frac=args.hvg_frac)


def generate_down_sample_test(args):
    uid = "%s_down_sample" % args.id
    evaluator = DownSampledDataReconstructionEvaluator(uid, args.data_set)
    evaluator.prepare()
    evaluator.set_seed(args.seed)
    evaluator.generate_test_bench(args.output, preserve_columns=args.preserve_columns, n_samples=args.n_samples,
                                  read_ratio=args.read_ratio, replace=args.replace)


def generate_paired_data_test(args):
    uid = "%s_paired_data" % args.id
    evaluator = PairedLQHQDataEvaluator(uid, args.data_set)
    evaluator.prepare()
    evaluator.set_seed(args.seed)
    evaluator.generate_test_bench(args.output, preserve_columns=args.preserve_columns)


def generate_cite_seq_test(args):
    uid = "%s_cite_seq" % args.id
    evaluator = CITESeqEvaluator(uid, args.data_set)
    evaluator.prepare()
    evaluator.set_seed(args.seed)
    evaluator.generate_test_bench(args.output, preserve_columns=args.preserve_columns)


def evaluate_cell_cycle_test(args):
    uid = "%s_cell_cycle" % args.id
    evaluator = CellCyclePreservationEvaluator(uid)
    evaluator.set_seed(args.seed)
    results = evaluator.evaluate_result(args.input, args.result_dir, visualization=args.visualization, clear_cache=args.clear_cache,
                                        normalization=args.normalization, transformation=args.transformation)
    print_metric_results(results)


def evaluate_clustering_test(args):
    uid = "%s_clustering" % args.id
    evaluator = ClusteringEvaluator(uid)
    evaluator.set_seed(args.seed)
    results = evaluator.evaluate_result(args.input, args.result_dir, visualization=args.visualization, clear_cache=args.clear_cache,
                                        normalization=args.normalization, transformation=args.transformation)
    print_metric_results(results)


def evaluate_random_mask_test(args):
    uid = "%s_random_mask" % args.id
    evaluator = RandomMaskedLocationPredictionEvaluator(uid)
    evaluator.set_seed(args.seed)
    results = evaluator.evaluate_result(args.input, args.result_dir, visualization=args.visualization)
    print_metric_results(results)


def evaluate_down_sample_test(args):
    uid = "%s_down_sample" % args.id
    evaluator = DownSampledDataReconstructionEvaluator(uid)
    evaluator.set_seed(args.seed)
    results = evaluator.evaluate_result(args.input, args.result_dir, visualization=args.visualization,
                                        transformation=args.transformation)
    print_metric_results(results)


def evaluate_paired_data_test(args):
    uid = "%s_paired_data" % args.id
    evaluator = PairedLQHQDataEvaluator(uid)
    evaluator.set_seed(args.seed)
    results = evaluator.evaluate_result(args.input, args.result_dir, visualization=args.visualization,
                                        normalization=args.normalization, transformation=args.transformation)
    print_metric_results(results)


def evaluate_cite_seq_test(args):
    uid = "%s_cite_seq" % args.id
    evaluator = CITESeqEvaluator(uid)
    evaluator.set_seed(args.seed)
    results = evaluator.evaluate_result(args.input, args.result_dir, visualization=args.visualization,
                                        transformation=args.transformation)
    print_metric_results(results)


def visualize_cell_cycle_evaluation(args):
    uid = "%s_cell_cycle" % args.id
    evaluator = CellCyclePreservationEvaluator(uid)
    evaluator.set_seed(args.seed)
    evaluator.visualize_result(args.result_dir, output_type=args.type)
    print("Done.")


def visualize_clustering_evaluation(args):
    uid = "%s_clustering" % args.id
    evaluator = ClusteringEvaluator(uid)
    evaluator.set_seed(args.seed)
    evaluator.visualize_result(args.result_dir, output_type=args.type)
    print("Done.")


def visualize_random_mask_evaluation(args):
    uid = "%s_random_mask" % args.id
    evaluator = RandomMaskedLocationPredictionEvaluator(uid)
    evaluator.set_seed(args.seed)
    evaluator.visualize_result(args.result_dir, output_type=args.type)
    print("Done.")


def visualize_down_sample_evaluation(args):
    uid = "%s_down_sample" % args.id
    evaluator = DownSampledDataReconstructionEvaluator(uid)
    evaluator.set_seed(args.seed)
    evaluator.visualize_result(args.result_dir, output_type=args.type)
    print("Done.")


def visualize_paired_data_evaluation(args):
    uid = "%s_paired_data" % args.id
    evaluator = PairedLQHQDataEvaluator(uid)
    evaluator.set_seed(args.seed)
    evaluator.visualize_result(args.result_dir, output_type=args.type)
    print("Done.")


def visualize_cite_seq_evaluation(args):
    uid = "%s_cite_seq" % args.id
    evaluator = CITESeqEvaluator(uid)
    evaluator.set_seed(args.seed)
    evaluator.visualize_result(args.result_dir, output_type=args.type)
    print("Done.")
