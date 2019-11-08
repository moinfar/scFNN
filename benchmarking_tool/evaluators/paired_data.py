import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from evaluators.base import AbstractEvaluator
from general.conf import settings
from utils.base import make_sure_dir_exists, dump_gzip_pickle, log, load_gzip_pickle
from data.data_set import get_data_set_class
from data.io import write_csv, read_table_file
from data.operations import shuffle_and_rename_columns, rearrange_and_rename_columns, normalizations, transformations


class PairedLQHQDataEvaluator(AbstractEvaluator):
    def __init__(self, uid, data_set_name=None):
        super(PairedLQHQDataEvaluator, self).__init__(uid)

        self.data_set_name = data_set_name
        self.data_set_hq = None
        self.data_set_lq = None

    def prepare(self, **kwargs):
        if self.data_set_name is None:
            raise ValueError("data_set_name can not be None.")

        self.data_set_hq = get_data_set_class(self.data_set_name + "-HQ")()
        self.data_set_lq = get_data_set_class(self.data_set_name + "-LQ")()
        self.data_set_hq.prepare()
        self.data_set_lq.prepare()

    def _load_lq_data(self):
        count_matrix = self.data_set_lq.get("data")
        return count_matrix

    def _load_hq_data(self):
        count_matrix = self.data_set_hq.get("data")
        return count_matrix

    def generate_test_bench(self, count_file_path, **kwargs):
        preserve_columns = kwargs['preserve_columns']

        count_file_path = os.path.abspath(count_file_path)

        count_matrix_lq = self._load_lq_data()
        count_matrix_hq = self._load_hq_data()

        # Shuffle columns
        count_matrix_lq, original_columns, column_permutation = \
            shuffle_and_rename_columns(count_matrix_lq, disabled=preserve_columns)

        # Remove zero rows
        count_matrix_hq = count_matrix_hq[np.sum(count_matrix_lq, axis=1) > 0].copy()
        count_matrix_lq = count_matrix_lq[np.sum(count_matrix_lq, axis=1) > 0].copy()

        # Save hidden data
        make_sure_dir_exists(settings.STORAGE_DIR)
        hidden_data_file_path = os.path.join(settings.STORAGE_DIR, "%s.hidden.pkl.gz" % self.uid)
        dump_gzip_pickle([count_matrix_lq.to_sparse(), original_columns, column_permutation,
                          count_matrix_hq.to_sparse()],
                         hidden_data_file_path)
        log("Benchmark hidden data saved to `%s`" % hidden_data_file_path)

        make_sure_dir_exists(os.path.dirname(count_file_path))
        write_csv(count_matrix_lq, count_file_path)
        log("Count file saved to `%s`" % count_file_path)

    def _load_hidden_state(self):
        hidden_data_file_path = os.path.join(settings.STORAGE_DIR, "%s.hidden.pkl.gz" % self.uid)
        sparse_count_matrix_lq, original_columns, column_permutation, sparse_count_matrix_hq = \
            load_gzip_pickle(hidden_data_file_path)

        count_matrix_lq = sparse_count_matrix_lq.to_dense()
        count_matrix_hq = sparse_count_matrix_hq.to_dense()

        del sparse_count_matrix_lq
        del sparse_count_matrix_hq

        return count_matrix_lq, original_columns, column_permutation, count_matrix_hq

    def evaluate_result(self, processed_count_file_path, result_dir, visualization, **kwargs):
        normalization = kwargs['normalization']
        transformation = kwargs['transformation']

        # Load hidden state and data
        count_matrix_lq, original_columns, column_permutation, count_matrix_hq = self._load_hidden_state()

        # Load imputed data
        imputed_data = read_table_file(processed_count_file_path)

        # Restore column names and order
        imputed_data = rearrange_and_rename_columns(imputed_data, original_columns, column_permutation)
        
        # Replace negative values with zero
        imputed_data = imputed_data.clip(lower=0)

        # Data transformations
        imputed_data = transformations[transformation](normalizations[normalization](imputed_data))
        count_matrix_hq = transformations[transformation](normalizations[normalization](count_matrix_hq))

        # Evaluation
        rmse_distances = []
        mae_distances = []
        euclidean_distances = []
        cosine_distances = []
        correlation_distances = []

        for i in range(count_matrix_hq.shape[1]):
            non_zeros = np.logical_and(count_matrix_hq.values[:, i] > 0, count_matrix_lq.values[:, i] == 0)
            hq = count_matrix_hq.values[non_zeros, i]
            lq = count_matrix_lq.values[non_zeros, i]
            y = imputed_data.values[non_zeros, i]
            if np.sum(y) > 0:
                y = y * np.sum(hq) / np.sum(y)
            rmse_distances.append(float(np.mean(np.square(hq - y) ** 0.5)))
            mae_distances.append(float(np.mean(np.abs(hq - y))))
            euclidean_distances.append(pdist(np.vstack((hq, y)), 'euclidean')[0])
            cosine_distances.append(pdist(np.vstack((hq, y)), 'cosine')[0])
            correlation_distances.append(pdist(np.vstack((hq, y)), 'correlation')[0])

        metric_results = {
            'cell_root_mean_squared_error': np.mean(rmse_distances),
            'cell_mean_absolute_error': np.mean(mae_distances),
            'cell_mean_euclidean_distance': np.mean(euclidean_distances),
            'cell_mean_cosine_distance': np.mean(cosine_distances),
            'cell_mean_correlation_distance': np.mean(correlation_distances)
        }

        # Save results to a file
        make_sure_dir_exists(os.path.join(result_dir, "files"))
        result_path = os.path.join(result_dir, "result.txt")
        with open(result_path, 'w') as file:
            file.write("## METRICS:\n")
            for metric in sorted(metric_results):
                file.write("%s\t%4f\n" % (metric, float(metric_results[metric])))

            file.write("##\n## ADDITIONAL INFO:\n")
            file.write("# CELL\troot_mean_squared_error\tmean_absolute_error\tmean_euclidean_distance\t"
                       "mean_cosine_distance\tmean_correlation_distance:\n")
            for i in range(count_matrix_hq.shape[1]):
                file.write("# %s\t%f\t%f\t%f\t%f\t%f\n" % (count_matrix_hq.columns.values[i],
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


class CITESeqEvaluator(AbstractEvaluator):
    def __init__(self, uid, data_set_name=None):
        super(CITESeqEvaluator, self).__init__(uid)

        self.data_set_name = data_set_name
        self.data_set = None
        self.protein_rna_mapping = None

    def prepare(self, **kwargs):
        if self.data_set_name is None:
            raise ValueError("data_set_name can not be None.")

        self.data_set = get_data_set_class(self.data_set_name)()
        self.data_set.prepare()

        assert "RNA" in self.data_set.keys()
        assert "ADT" in self.data_set.keys()

        if self.data_set_name in ["CITE-CBMC", "CITE-PBMC", "CITE-CD8"]:
            self.protein_rna_mapping = {
                "CD2": "CD2",
                "CD3": "CD3E",  # It is about CD8E (See Supplementary Table 2 of Cite-Seq paper)
                "CD4": "CD4",
                "CD8": "CD8A",  # It is CD8A not CD8B (See Supplementary Table 2 of Cite-Seq paper)
                "CD45RA": "PTPRC",
                "CD56": "NCAM1",
                "CD57": "B3GAT1",
                "CD16": "FCGR3A",  # What about FCGR3B? (We note that FCGR3B is expressed in a few samples)
                "CD10": "MME",
                "CD11c": "ITGAX",
                "CD14": "CD14",
                "CD19": "CD19",
                "CD34": "CD34",
                "CCR5": "CCR5",
                "CCR7": "CCR7"
            }
        else:
            raise NotImplementedError()

    def generate_test_bench(self, count_file_path, **kwargs):
        preserve_columns = kwargs['preserve_columns']

        count_file_path = os.path.abspath(count_file_path)

        count_rna = self.data_set.get("RNA")
        count_adt = self.data_set.get("ADT")

        # Shuffle columns
        count_rna, original_columns, column_permutation = \
            shuffle_and_rename_columns(count_rna, disabled=preserve_columns)

        # Remove zero rows
        count_rna = count_rna[np.sum(count_rna, axis=1) > 0].copy()

        # Save hidden data
        make_sure_dir_exists(settings.STORAGE_DIR)
        hidden_data_file_path = os.path.join(settings.STORAGE_DIR, "%s.hidden.pkl.gz" % self.uid)
        dump_gzip_pickle([count_rna.to_sparse(), original_columns, column_permutation,
                          count_adt.to_sparse(), self.protein_rna_mapping],
                         hidden_data_file_path)
        log("Benchmark hidden data saved to `%s`" % hidden_data_file_path)

        make_sure_dir_exists(os.path.dirname(count_file_path))
        write_csv(count_rna, count_file_path)
        log("Count file saved to `%s`" % count_file_path)

    def _load_hidden_state(self):
        hidden_data_file_path = os.path.join(settings.STORAGE_DIR, "%s.hidden.pkl.gz" % self.uid)
        sparse_count_rna, original_columns, column_permutation, sparse_count_adt, protein_rna_mapping = \
            load_gzip_pickle(hidden_data_file_path)

        count_rna = sparse_count_rna.to_dense()
        count_adt = sparse_count_adt.to_dense()

        del sparse_count_rna
        del sparse_count_adt

        return count_rna, original_columns, column_permutation, count_adt, protein_rna_mapping

    def evaluate_result(self, processed_count_file_path, result_dir, visualization, **kwargs):
        make_sure_dir_exists(os.path.join(result_dir, "files"))

        transformation = kwargs['transformation']

        # Load hidden state and data
        _, original_columns, column_permutation, count_adt, protein_rna_mapping = self._load_hidden_state()

        # Load imputed data
        imputed_rna = read_table_file(processed_count_file_path)

        # Restore column names and order
        imputed_rna = rearrange_and_rename_columns(imputed_rna, original_columns, column_permutation)

        # Data transformations
        imputed_rna = transformations[transformation](imputed_rna)
        count_adt = transformations[transformation](count_adt)

        # Use related data
        adt = count_adt.loc[[prot for prot in count_adt.index.values if (protein_rna_mapping[prot] in imputed_rna.index.values)]].copy()
        adt.index = ["prot_" + p for p in adt.index.values]
        rna = imputed_rna.loc[[protein_rna_mapping[prot] for prot in count_adt.index.values if (protein_rna_mapping[prot] in imputed_rna.index.values)]]
        rna.index = ["gene_" + g for g in rna.index.values]

        info = []

        write_csv(adt, os.path.join(result_dir, "files", "adt.csv"))
        info.append({'filename': "adt.csv",
                     'description': 'Protein expressions (adt) after transformation',
                     'plot_description': 'Protein expressions (adt) after transformation',
                     })

        write_csv(rna, os.path.join(result_dir, "files", "rna.csv"))
        info.append({'filename': "rna.csv",
                     'description': 'Gene expressions of genes related to adt data after transformation',
                     'plot_description': 'Gene expressions of genes related to adt data after transformation',
                     })

        n = adt.shape[0]

        # Calculating Spearman correlations
        combined_df = pd.concat((adt, rna)).transpose()
        correlations = combined_df.corr(method="spearman")

        adt_adt_spearmanr = correlations.iloc[:n, :n]
        rna_rna_spearmanr = correlations.iloc[n:, n:]
        adt_rna_spearmanr = correlations.iloc[:n, n:]

        write_csv(correlations, os.path.join(result_dir, "files", "spearman_correlations.csv"))
        info.append({'filename': "spearman_correlations.csv",
                     'description': 'Pairwise Spearman correlations (first n items are '
                                    'adt expressions and second n items are rna expressions)',
                     'plot_description': 'Pairwise Spearman correlations (first n items are '
                                         'adt expressions and second n items are rna expressions)',
                     })

        # Calculating Pearson correlations
        combined_df = pd.concat((adt, rna)).transpose()
        correlations = combined_df.corr(method="pearson")

        adt_adt_pearsonr = correlations.iloc[:n, :n]
        rna_rna_pearsonr = correlations.iloc[n:, n:]
        adt_rna_pearsonr = correlations.iloc[:n, n:]

        write_csv(correlations, os.path.join(result_dir, "files", "pearson_correlations.csv"))
        info.append({'filename': "pearson_correlations.csv",
                     'description': 'Pairwise Pearson correlations (first n items are '
                                    'adt expressions and second n items are rna expressions)',
                     'plot_description': 'Pairwise Pearson correlations (first n items are '
                                         'adt expressions and second n items are rna expressions)',
                     })

        # Evaluation
        metric_results = {
            'rna_protein_mean_spearman_correlatoin': np.mean(adt_rna_spearmanr.values.diagonal()),
            'rna_protein_mean_pearson_correlatoin': np.mean(adt_rna_pearsonr.values.diagonal()),
            'MSE_of_adt_adt_and_rna_rna_spearman_correlations':
                np.mean((adt_adt_spearmanr.values - rna_rna_spearmanr.values) ** 2),
            'MSE_of_adt_adt_and_rna_rna_pearson_correlations':
                np.mean((adt_adt_pearsonr.values - rna_rna_pearsonr.values) ** 2)
        }

        write_csv(pd.DataFrame(info), os.path.join(result_dir, "files", "info.csv"))

        # Save results to a file
        result_path = os.path.join(result_dir, "result.txt")
        with open(result_path, 'w') as file:
            file.write("## METRICS:\n")
            for metric in sorted(metric_results):
                file.write("%s\t%4f\n" % (metric, float(metric_results[metric])))

            file.write("##\n## ADDITIONAL INFO:\n")
            file.write("## Pearson of adt/rna:\n")
            file.write("## " + "\n## ".join(adt_rna_pearsonr.to_string().split("\n")) + "\n")
            file.write('## Spearman of adt/rna:\n')
            file.write("## " + "\n## ".join(adt_rna_spearmanr.to_string().split("\n")) + "\n")
            file.write("## Pearson of adt/adt:\n")
            file.write("## " + "\n## ".join(adt_adt_pearsonr.to_string().split("\n")) + "\n")
            file.write("## Pearson of rna/rna:\n")
            file.write("## " + "\n## ".join(rna_rna_pearsonr.to_string().split("\n")) + "\n")
            file.write('## Spearman of adt/adt:\n')
            file.write("## " + "\n## ".join(adt_adt_spearmanr.to_string().split("\n")) + "\n")
            file.write('## Spearman of rna/rna:\n')
            file.write("## " + "\n## ".join(rna_rna_spearmanr.to_string().split("\n")) + "\n")

        log("Evaluation results saved to `%s`" % result_path)

        if visualization != "none":
            self.visualize_result(result_dir, output_type=visualization)

        return metric_results

    def visualize_result(self, result_dir, output_type, **kwargs):
        info = read_table_file(os.path.join(result_dir, "files", "info.csv"))
        info = info.set_index("filename")

        spearman_correlations = read_table_file(os.path.join(result_dir, "files", "spearman_correlations.csv"))
        pearson_correlations = read_table_file(os.path.join(result_dir, "files", "pearson_correlations.csv"))

        n = spearman_correlations.shape[0] // 2

        adt_adt_spearmanr = spearman_correlations.iloc[:n, :n]
        rna_rna_spearmanr = spearman_correlations.iloc[n:, n:]
        adt_rna_spearmanr = spearman_correlations.iloc[:n, n:]

        adt_adt_pearsonr = pearson_correlations.iloc[:n, :n]
        rna_rna_pearsonr = pearson_correlations.iloc[n:, n:]
        adt_rna_pearsonr = pearson_correlations.iloc[:n, n:]

        if output_type == "pdf":
            import plotly.graph_objs as go
            import plotly.io as pio

            plots = [
                ("Pairwise Spearman correlations between ADT values", adt_adt_spearmanr,
                 "heatmap_adt_adt_spearmanr.pdf"),
                ("Pairwise Spearman correlations between RNA values", rna_rna_spearmanr,
                 "heatmap_rna_rna_spearmanr.pdf"),
                ("Pairwise Spearman correlations between ADT and RNA values",
                 adt_rna_spearmanr, "heatmap_adt_rna_spearmanr.pdf"),
                ("Pairwise Pearson correlations between ADT values", adt_adt_pearsonr, "heatmap_adt_adt_pearsonr.pdf"),
                ("Pairwise Pearson correlations between RNA values", rna_rna_pearsonr, "heatmap_rna_rna_pearsonr.pdf"),
                ("Pairwise Pearson correlations between ADT and RNA values",
                 adt_rna_pearsonr, "heatmap_adt_rna_pearsonr.pdf")
            ]

            for title, data_frame, filename in plots:
                fig = go.Figure(layout=go.Layout(title=title, font=dict(size=9)))

                fig.add_heatmap(z=data_frame.values,
                                x=data_frame.columns.values,
                                y=data_frame.index.values,
                                colorscale='Picnic')  # RdBu is also good

                pio.write_image(fig, os.path.join(result_dir, filename),
                                width=600, height=700)

        elif output_type == "html":
            print("Nothing to visualize")
        else:
            raise NotImplementedError()
