import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr

from data.data_set import get_data_set_class
from data.io import write_csv, read_table_file
from data.operations import shuffle_and_rename_columns, rearrange_and_rename_columns, transformations
from evaluators.base import AbstractEvaluator
from general import settings
from utils.base import make_sure_dir_exists, log, dump_gzip_pickle, load_gzip_pickle


class RandomMaskedLocationPredictionEvaluator(AbstractEvaluator):
    def __init__(self, uid, data_set_name=None):
        super(RandomMaskedLocationPredictionEvaluator, self).__init__(uid)

        self.data_set_name = data_set_name
        self.data_set = None

    def prepare(self, **kwargs):
        if self.data_set_name is None:
            raise ValueError("data_set_name can not be None.")

        self.data_set = get_data_set_class(self.data_set_name)()
        self.data_set.prepare()

    def _load_data(self, n_samples):
        assert "data" in self.data_set.keys()

        data = self.data_set.get("data")

        if n_samples is None or n_samples == 0:
            pass
        else:
            subset = np.random.choice(data.shape[1], n_samples, replace=False)
            data = data.iloc[:, subset].copy()

        return data

    @staticmethod
    def get_hvg_genes(data, hvg_frac):
        gene_means = np.log(data.mean(axis=1) + 1e-8)
        gene_vars = 2 * np.log(data.std(axis=1) + 1e-8)

        hvg_indices = set()

        for x in np.arange(0, gene_means.max(), 0.5):
            related_indices = np.where(np.logical_and(x <= gene_means, gene_means < x + 1))[0]
            if related_indices.shape[0] == 0:
                continue
            threshold = np.percentile((gene_vars - gene_means)[related_indices], 100 * (1 - hvg_frac))
            hvg_indices.update(list(np.where(np.logical_and.reduce((x <= gene_means,
                                                                    gene_means < x + 1,
                                                                    gene_vars - gene_means >= threshold)))[0]))

        return list(sorted(list(hvg_indices)))

    def generate_test_bench(self, count_file_path, **kwargs):
        n_samples = kwargs['n_samples']
        dropout_count = kwargs['dropout_count']
        min_expression = kwargs['min_expression']
        hvg_frac = kwargs['hvg_frac']
        preserve_columns = kwargs['preserve_columns']

        count_file_path = os.path.abspath(count_file_path)
        data = self._load_data(n_samples)

        hvg_indices = self.get_hvg_genes(data, hvg_frac)

        # Generate elimination mask
        non_zero_locations = []

        data_values = data.values
        for x in hvg_indices:
            for y in range(data.shape[1]):
                if data_values[x, y] >= min_expression:
                    non_zero_locations.append((x, y))
        del data_values

        mask = np.zeros_like(data)

        masked_locations = [non_zero_locations[index] for index in
                            np.random.choice(len(non_zero_locations),
                                             dropout_count, replace=False)]

        for (x, y) in masked_locations:
            mask[x, y] = 1

        mask = pd.DataFrame(mask, index=data.index, columns=data.columns)

        # Elimination
        low_quality_data = data * (1 - mask.values)

        is_nonzero = np.sum(low_quality_data, axis=1) > 0
        mask = mask[is_nonzero].copy()
        data = data[is_nonzero].copy()
        low_quality_data = low_quality_data[is_nonzero].copy()

        # Shuffle columns
        low_quality_data, original_columns, column_permutation = \
            shuffle_and_rename_columns(low_quality_data, disabled=preserve_columns)

        # Save hidden data
        make_sure_dir_exists(settings.STORAGE_DIR)
        hidden_data_file_path = os.path.join(settings.STORAGE_DIR, "%s.hidden.pkl.gz" % self.uid)
        dump_gzip_pickle([data.to_sparse(), mask.to_sparse(), original_columns, column_permutation],
                         hidden_data_file_path)
        log("Benchmark hidden data saved to `%s`" % hidden_data_file_path)

        make_sure_dir_exists(os.path.dirname(count_file_path))
        write_csv(low_quality_data, count_file_path)
        log("Count file saved to `%s`" % count_file_path)

    def _load_hidden_state(self):
        hidden_data_file_path = os.path.join(settings.STORAGE_DIR, "%s.hidden.pkl.gz" % self.uid)
        sparse_data, sparse_mask, original_columns, column_permutation = \
            load_gzip_pickle(hidden_data_file_path)
        data = sparse_data.to_dense()
        mask = sparse_mask.to_dense()

        del sparse_data
        del sparse_mask

        return data, mask, original_columns, column_permutation

    def evaluate_result(self, processed_count_file_path, result_dir, visualization, **kwargs):
        make_sure_dir_exists(os.path.join(result_dir, "files"))
        info = []

        # Load hidden state and data
        data, mask, original_columns, column_permutation = self._load_hidden_state()

        # Load imputed data
        imputed_data = read_table_file(processed_count_file_path)

        # Restore column names and order
        imputed_data = rearrange_and_rename_columns(imputed_data, original_columns, column_permutation)

        # Replace negative values with zero
        imputed_data = imputed_data.clip(lower=0)

        # Evaluation
        log_diff = np.abs(transformations["log"](data) - transformations["log"](imputed_data))
        sqrt_diff = np.abs(transformations["sqrt"](data) - transformations["sqrt"](imputed_data))

        mse_on_log = float(np.sum(np.sum(mask * np.where(data != 0, 1, 0) * np.square(log_diff))) /
                           np.sum(np.sum(mask * np.where(data != 0, 1, 0))))
        mae_on_log = float(np.sum(np.sum(mask * np.where(data != 0, 1, 0) * np.abs(log_diff))) /
                           np.sum(np.sum(mask * np.where(data != 0, 1, 0))))
        mse_on_sqrt = float(np.sum(np.sum(mask * np.where(data != 0, 1, 0) * np.square(sqrt_diff))) /
                            np.sum(np.sum(mask * np.where(data != 0, 1, 0))))
        mae_on_sqrt = float(np.sum(np.sum(mask * np.where(data != 0, 1, 0) * np.abs(sqrt_diff))) /
                            np.sum(np.sum(mask * np.where(data != 0, 1, 0))))

        metric_results = {
            'RMSE_sqrt': mse_on_sqrt ** 0.5,
            'MAE_sqrt': mae_on_sqrt,
            'RMSE_log': mse_on_log ** 0.5,
            'MAE_log': mae_on_log
        }

        masked_locations = []
        mask_values = mask.values
        for x in range(mask_values.shape[0]):
            for y in range(mask_values.shape[1]):
                if mask_values[x, y] == 1:
                    masked_locations.append((x, y))

        original_values = []
        predicted_values = []
        for (x, y) in masked_locations:
            original_values.append(data.iloc[x, y])
            predicted_values.append(imputed_data.iloc[x, y])

        original_values = np.asarray(original_values)
        predicted_values = np.asarray(predicted_values)

        predictions_df = pd.DataFrame({'original': original_values, 'predicted': predicted_values})
        write_csv(predictions_df, os.path.join(result_dir, "files", "predictions.csv"))
        info.append({'filename': "predictions.csv",
                     'description': 'Original masked values along predicted values',
                     'plot_description': 'Predicted values vs. original masked values',
                     })

        write_csv(pd.DataFrame(info), os.path.join(result_dir, "files", "info.csv"))

        # Save results to a file
        result_path = os.path.join(result_dir, "result.txt")
        with open(result_path, 'w') as file:
            file.write("## METRICS:\n")
            for metric in sorted(metric_results):
                file.write("%s\t%4f\n" % (metric, metric_results[metric]))

            file.write("##\n## ADDITIONAL INFO:\n")
            file.write("# GENE\tCELL\tGOLD_STANDARD\tRESULT:\n")
            for (x, y) in masked_locations:
                file.write("# %s\t%s\t%f\t%f\n" % (data.index.values[x],
                                                   data.columns.values[y],
                                                   data.iloc[x, y],
                                                   imputed_data.iloc[x, y]))

        log("Evaluation results saved to `%s`" % result_path)

        if visualization != "none":
            self.visualize_result(result_dir, output_type=visualization)

        return metric_results

    def visualize_result(self, result_dir, output_type, **kwargs):
        info = read_table_file(os.path.join(result_dir, "files", "info.csv"))
        info = info.set_index("filename")

        predictions = read_table_file(os.path.join(result_dir, "files", "predictions.csv"))
        original_values = predictions["original"]
        predicted_values = predictions["predicted"]

        if output_type == "pdf":
            import plotly.graph_objs as go
            import plotly.io as pio

            max_axis = float(max(original_values.max(), predicted_values.max()))
            for transformation_name in ["log", "sqrt"]:
                transformation = transformations[transformation_name]

                fig = go.Figure(layout=go.Layout(title='Predicted values vs. original masked values (%s scale)' %
                                                       transformation_name, font=dict(size=12),
                                                 xaxis=go.layout.XAxis(range=[0, transformation(max_axis)]),
                                                 yaxis=go.layout.YAxis(range=[0, transformation(max_axis)])))
                fig.add_scatter(x=transformation(original_values),
                                y=transformation(predicted_values),
                                mode='markers', marker=dict(opacity=0.3))
                pio.write_image(fig, os.path.join(result_dir, "prediction_plot_%s_scale.pdf" % transformation_name),
                                width=800, height=800)
        elif output_type == "html":
            raise NotImplementedError()
        else:
            raise NotImplementedError()


class DownSampledDataReconstructionEvaluator(AbstractEvaluator):
    def __init__(self, uid, data_set_name=None):
        super(DownSampledDataReconstructionEvaluator, self).__init__(uid)

        self.data_set_name = data_set_name
        self.data_set = None

    def prepare(self, **kwargs):
        if self.data_set_name is None:
            raise ValueError("data_set_name can not be None.")

        self.data_set = get_data_set_class(self.data_set_name)()
        self.data_set.prepare()

    def _load_data(self, n_samples):
        assert "data" in self.data_set.keys()

        data = self.data_set.get("data")

        if n_samples is None or n_samples == 0:
            pass
        else:
            subset = np.random.choice(data.shape[1], n_samples, replace=False)
            data = data.iloc[:, subset].copy()

        return data

    def generate_test_bench(self, count_file_path, **kwargs):
        n_samples = kwargs['n_samples']
        read_ratio = kwargs['read_ratio']
        replce = kwargs['replace']
        preserve_columns = kwargs['preserve_columns']

        count_file_path = os.path.abspath(count_file_path)
        data = self._load_data(n_samples)

        # find cumulative distribution (sum)
        data_values = data.astype(int).values
        n_all_reads = np.sum(data_values)
        data_cumsum = np.reshape(np.cumsum(data_values), data_values.shape)

        # Sample from original dataset
        new_reads = np.sort(np.random.choice(n_all_reads, int(read_ratio * n_all_reads), replace=replce))

        low_quality_data = np.zeros_like(data_values)
        read_index = 0
        for x in range(data_values.shape[0]):
            for y in range(data_values.shape[1]):
                while read_index < len(new_reads) and new_reads[read_index] < data_cumsum[x, y]:
                    low_quality_data[x, y] += 1
                    read_index += 1

        # Convert to data frame
        low_quality_data = pd.DataFrame(low_quality_data, index=data.index, columns=data.columns)

        # Shuffle columns
        low_quality_data, original_columns, column_permutation = \
            shuffle_and_rename_columns(low_quality_data, disabled=preserve_columns)

        # Remove zero rows
        data = data[np.sum(low_quality_data, axis=1) > 0].copy()
        low_quality_data = low_quality_data[np.sum(low_quality_data, axis=1) > 0].copy()

        # Save hidden data
        make_sure_dir_exists(settings.STORAGE_DIR)
        hidden_data_file_path = os.path.join(settings.STORAGE_DIR, "%s.hidden.pkl.gz" % self.uid)
        dump_gzip_pickle([data.to_sparse(), read_ratio, original_columns, column_permutation],
                         hidden_data_file_path)
        log("Benchmark hidden data saved to `%s`" % hidden_data_file_path)

        make_sure_dir_exists(os.path.dirname(count_file_path))
        write_csv(low_quality_data, count_file_path)
        log("Count file saved to `%s`" % count_file_path)

    def _load_hidden_state(self):
        hidden_data_file_path = os.path.join(settings.STORAGE_DIR, "%s.hidden.pkl.gz" % self.uid)
        sparse_data, read_ratio, original_columns, column_permutation = load_gzip_pickle(hidden_data_file_path)

        scaled_data = sparse_data.to_dense() * read_ratio

        return scaled_data, original_columns, column_permutation

    def evaluate_result(self, processed_count_file_path, result_dir, visualization, **kwargs):
        transformation = kwargs['transformation']

        make_sure_dir_exists(os.path.join(result_dir, "files"))

        # Load hidden state and data
        scaled_data, original_columns, column_permutation = self._load_hidden_state()

        # Load imputed data
        imputed_data = read_table_file(processed_count_file_path)

        # Restore column names and order
        imputed_data = rearrange_and_rename_columns(imputed_data, original_columns, column_permutation)

        # Replace negative values with zero
        imputed_data = imputed_data.clip(lower=0)

        # Data transformation
        scaled_data = transformations[transformation](scaled_data)
        imputed_data = transformations[transformation](imputed_data)

        # Evaluation
        rmse_distances = []
        mae_distances = []
        euclidean_distances = []
        cosine_distances = []
        correlation_distances = []

        rmse = float(np.sum(np.where(scaled_data.values > 0, 1, 0) * np.square(scaled_data.values - imputed_data.values)) /
                     np.sum(np.where(scaled_data.values > 0, 1, 0))) ** 0.5
        mae = float(np.sum(np.where(scaled_data.values > 0, 1, 0) * np.abs(scaled_data.values - imputed_data.values)) /
                    np.sum(np.where(scaled_data.values > 0, 1, 0)))

        for i in range(scaled_data.shape[1]):
            non_zeros = scaled_data.values[:, i] > 0
            x = scaled_data.values[non_zeros, i]
            y = imputed_data.values[non_zeros, i]
            rmse_distances.append(float(np.sum(np.square(x - y)) / np.sum(non_zeros)) ** 0.5)
            mae_distances.append(float(np.sum(np.abs(x - y)) / np.sum(non_zeros)))
            cosine_distances.append(pdist(np.vstack((x, y)), 'cosine')[0])
            euclidean_distances.append(pdist(np.vstack((x, y)), 'euclidean')[0])
            correlation_distances.append(pdist(np.vstack((x, y)), 'correlation')[0])

        metric_results = {
            'all_mean_absolute_error_on_non_zeros': mae,
            'all_root_mean_squared_error_on_non_zeros': rmse,
            'cell_mean_mean_absolute_error_on_non_zeros': np.mean(mae_distances),
            'cell_mean_root_mean_squared_error_on_non_zeros': np.mean(rmse_distances),
            'cell_mean_euclidean_distance': np.mean(euclidean_distances),
            'cell_mean_cosine_distance': np.mean(cosine_distances),
            'cell_mean_correlation_distance': np.mean(correlation_distances),
        }

        # Save results to a file
        result_path = os.path.join(result_dir, "result.txt")
        with open(result_path, 'w') as file:
            file.write("## METRICS:\n")
            for metric in sorted(metric_results):
                file.write("%s\t%4f\n" % (metric, float(metric_results[metric])))

            file.write("##\n## ADDITIONAL INFO:\n")
            file.write("# CELL\troot_mean_squared_error_on_non_zeros\tmean_absolute_error_on_non_zeros\t"
                       "euclidean_distance_on_non_zeros\tcosine_distance_on_non_zeros\tcorrelation_distance_on_non_zeros:\n")
            for i in range(scaled_data.shape[1]):
                file.write("# %s\t%f\t%f\t%f\t%f\t%f\n" % (scaled_data.columns.values[i],
                                                           rmse_distances[i],
                                                           mae_distances[i],
                                                           euclidean_distances[i],
                                                           cosine_distances[i],
                                                           correlation_distances[i]))

        log("Evaluation results saved to `%s`" % result_path)

        if visualization != "none":
            self.visualize_result(result_dir, output_type=visualization)

        return metric_results

    def visualize_result(self, result_dir, output_type, **kwargs):
        if output_type == "pdf":
            print("Nothing to visualize")
        elif output_type == "html":
            print("Nothing to visualize")
        else:
            raise NotImplementedError()
