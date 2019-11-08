import os

import numpy as np
import pandas as pd
import umap
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabaz_score, silhouette_score, accuracy_score
from sklearn.metrics.cluster import adjusted_mutual_info_score, v_measure_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from data.data_set import get_data_set_class
from data.io import read_table_file, write_csv
from data.operations import shuffle_and_rename_columns, rearrange_and_rename_columns, normalizations, transformations
from evaluators.base import AbstractEvaluator
from general.conf import settings
from utils.base import make_sure_dir_exists, log, dump_gzip_pickle, load_gzip_pickle


class CellCyclePreservationEvaluator(AbstractEvaluator):
    def __init__(self, uid):
        super(CellCyclePreservationEvaluator, self).__init__(uid)

        self.data_set = None

    def _load_and_combine_data(self):
        data_G1 = self.data_set.get("G1")
        data_G2M = self.data_set.get("G2M")
        data_S = self.data_set.get("S")

        shared_columns = ['EnsemblGeneID', 'EnsemblTranscriptID', 'AssociatedGeneName', 'GeneLength']

        merged_data = pd.merge(data_G1,
                               pd.merge(data_G2M, data_S, on=shared_columns),
                               on=shared_columns)

        merged_data = merged_data.drop(columns=['EnsemblTranscriptID',
                                                'AssociatedGeneName',
                                                'GeneLength'])

        merged_data = merged_data.set_index('EnsemblGeneID')
        merged_data.index.names = ['Symbol']
        merged_data = merged_data.drop(['Ambiguous', 'No_feature', 'Not_aligned',
                                        'Too_low_aQual', 'Aligned'])

        assert merged_data.shape == (38385, 288)

        # remove zero-sum rows
        merged_data = merged_data[merged_data.sum(axis=1) > 0]

        return merged_data

    def prepare(self):
        data_set_class = get_data_set_class("CELL_CYCLE")
        self.data_set = data_set_class()
        self.data_set.prepare()

    def generate_test_bench(self, count_file_path, **kwargs):
        count_file_path = os.path.abspath(count_file_path)
        rm_ercc = kwargs['rm_ercc']
        rm_mt = kwargs['rm_mt']
        rm_lq = kwargs['rm_lq']
        preserve_columns = kwargs['preserve_columns']

        # Load dataset
        data = self._load_and_combine_data()

        # Remove some rows and columns
        if rm_ercc:
            remove_list = [symbol for symbol in data.index.values if symbol.startswith("ERCC-")]
            data = data.drop(remove_list)
        if rm_mt:
            remove_list = [symbol for symbol in data.index.values if symbol.startswith("mt-")]
            data = data.drop(remove_list)
        if rm_lq:
            remove_list = data.columns.values[data.sum(axis=0) < 1e6]
            data = data.drop(columns=remove_list)
        # Remove empty rows
        remove_list = data.index.values[data.sum(axis=1) == 0]
        data = data.drop(remove_list)

        # Shuffle columns
        new_data, original_columns, column_permutation = shuffle_and_rename_columns(data, disabled=preserve_columns)

        # Save hidden data
        make_sure_dir_exists(settings.STORAGE_DIR)
        hidden_data_file_path = os.path.join(settings.STORAGE_DIR, "%s.hidden.pkl.gz" % self.uid)
        dump_gzip_pickle([data.to_sparse(), original_columns, column_permutation], hidden_data_file_path)
        log("Benchmark hidden data saved to `%s`" % hidden_data_file_path)

        make_sure_dir_exists(os.path.dirname(count_file_path))
        write_csv(new_data, count_file_path)
        log("Count file saved to `%s`" % count_file_path)

        return None

    def _load_data_and_imputed_data_for_evaluation(self, processed_count_file):
        hidden_data_file_path = os.path.join(settings.STORAGE_DIR, "%s.hidden.pkl.gz" % self.uid)
        sparse_data, original_columns, column_permutation = load_gzip_pickle(hidden_data_file_path)
        data = sparse_data.to_dense()
        del sparse_data

        imputed_data = read_table_file(processed_count_file)

        # Restoring original column names
        imputed_data = rearrange_and_rename_columns(imputed_data, original_columns, column_permutation)

        # Remove (error correction) ERCC and mitochondrial RNAs
        remove_list = [symbol for symbol in imputed_data.index.values
                       if symbol.startswith("ERCC-") or symbol.startswith("mt-")]

        imputed_data = imputed_data.drop(remove_list)
        data = data.drop(remove_list)

        return data, imputed_data

    @staticmethod
    def _get_related_part(data):
        G1_S_related_genes = ["ENSMUSG00000000028", "ENSMUSG00000001228", "ENSMUSG00000002870", "ENSMUSG00000004642",
                              "ENSMUSG00000005410", "ENSMUSG00000006678", "ENSMUSG00000006715", "ENSMUSG00000017499",
                              "ENSMUSG00000020649", "ENSMUSG00000022360", "ENSMUSG00000022422", "ENSMUSG00000022673",
                              "ENSMUSG00000022945", "ENSMUSG00000023104", "ENSMUSG00000024151", "ENSMUSG00000024742",
                              "ENSMUSG00000025001", "ENSMUSG00000025395", "ENSMUSG00000025747", "ENSMUSG00000026355",
                              "ENSMUSG00000027242", "ENSMUSG00000027323", "ENSMUSG00000027342", "ENSMUSG00000028212",
                              "ENSMUSG00000028282", "ENSMUSG00000028560", "ENSMUSG00000028693", "ENSMUSG00000028884",
                              "ENSMUSG00000029591", "ENSMUSG00000030346", "ENSMUSG00000030528", "ENSMUSG00000030726",
                              "ENSMUSG00000030978", "ENSMUSG00000031629", "ENSMUSG00000031821", "ENSMUSG00000032397",
                              "ENSMUSG00000034329", "ENSMUSG00000037474", "ENSMUSG00000039748", "ENSMUSG00000041712",
                              "ENSMUSG00000042489", "ENSMUSG00000046179", "ENSMUSG00000055612"]
        G2_M_related_genes = ["ENSMUSG00000001403", "ENSMUSG00000004880", "ENSMUSG00000005698", "ENSMUSG00000006398",
                              "ENSMUSG00000009575", "ENSMUSG00000012443", "ENSMUSG00000015749", "ENSMUSG00000017716",
                              "ENSMUSG00000019942", "ENSMUSG00000019961", "ENSMUSG00000020330", "ENSMUSG00000020737",
                              "ENSMUSG00000020808", "ENSMUSG00000020897", "ENSMUSG00000020914", "ENSMUSG00000022385",
                              "ENSMUSG00000022391", "ENSMUSG00000023505", "ENSMUSG00000024056", "ENSMUSG00000024795",
                              "ENSMUSG00000026605", "ENSMUSG00000026622", "ENSMUSG00000026683", "ENSMUSG00000027306",
                              "ENSMUSG00000027379", "ENSMUSG00000027469", "ENSMUSG00000027496", "ENSMUSG00000027699",
                              "ENSMUSG00000028044", "ENSMUSG00000028678", "ENSMUSG00000028873", "ENSMUSG00000029177",
                              "ENSMUSG00000031004", "ENSMUSG00000032218", "ENSMUSG00000032254", "ENSMUSG00000034349",
                              "ENSMUSG00000035293", "ENSMUSG00000036752", "ENSMUSG00000036777", "ENSMUSG00000037313",
                              "ENSMUSG00000037544", "ENSMUSG00000037725", "ENSMUSG00000038252", "ENSMUSG00000038379",
                              "ENSMUSG00000040549", "ENSMUSG00000044201", "ENSMUSG00000044783", "ENSMUSG00000045328",
                              "ENSMUSG00000048327", "ENSMUSG00000048922", "ENSMUSG00000054717", "ENSMUSG00000062248",
                              "ENSMUSG00000068744", "ENSMUSG00000074802"]

        G1_S_related_part_of_data = data.loc[G1_S_related_genes]
        G2_M_related_part_of_data = data.loc[G2_M_related_genes]

        return G1_S_related_part_of_data, G2_M_related_part_of_data

    @staticmethod
    def _get_embeddings(related_part_of_imputed_data):
        emb_pca = PCA(n_components=2). \
            fit_transform(related_part_of_imputed_data.transpose())
        emb_ica = FastICA(n_components=2). \
            fit_transform(related_part_of_imputed_data.transpose())
        emb_tsvd = TruncatedSVD(n_components=2). \
            fit_transform(related_part_of_imputed_data.transpose())
        emb_tsne = TSNE(n_components=2, method='exact'). \
            fit_transform(related_part_of_imputed_data.transpose())
        emb_umap = umap.UMAP(n_components=2, n_neighbors=4, min_dist=0.3, metric='correlation'). \
            fit_transform(related_part_of_imputed_data.transpose())

        embedded_data = {
            "PCA": emb_pca,
            "ICA": emb_ica,
            "Truncated SVD": emb_tsvd,
            "tSNE": emb_tsne,
            "UMAP": emb_umap
        }

        return embedded_data

    @staticmethod
    def _get_classification_results(related_part_of_imputed_data, gold_standard_classes, repeats=10):
        svm_results = []
        knn_results = []
        for algorithm in ["svm", "knn"]:
            for i in range(repeats):
                X_train, X_test, y_train, y_test = train_test_split(related_part_of_imputed_data.transpose().values,
                                                                    gold_standard_classes,
                                                                    test_size=0.25,
                                                                    shuffle=True)
                if algorithm == "svm":
                    classifier = svm.SVC(gamma='scale')
                    classifier.fit(X_train, y_train)
                    svm_results.append(accuracy_score(y_test, classifier.predict(X_test)))
                elif algorithm == "knn":
                    classifier = KNeighborsClassifier(n_neighbors=3)
                    classifier.fit(X_train, y_train)
                    knn_results.append(accuracy_score(y_test, classifier.predict(X_test)))

        return svm_results, knn_results

    def evaluate_result(self, processed_count_file, result_dir, visualization, **kwargs):
        normalization = kwargs['normalization']
        transformation = kwargs['transformation']
        clear_cache = kwargs['clear_cache']

        make_sure_dir_exists(os.path.join(result_dir, "files"))
        info = []

        data, imputed_data = self._load_data_and_imputed_data_for_evaluation(processed_count_file)
        gold_standard_classes = [column_name.split("_")[0] for column_name in data.columns.values]

        # Data transformations
        if np.sum(imputed_data.values < 0) > 0:
            log("Observed some negative values!")
            imputed_data[imputed_data < 0] = 0
        imputed_data = transformations[transformation](normalizations[normalization](imputed_data))

        G1_S_related_part_of_imputed_data, G2_M_related_part_of_imputed_data = self._get_related_part(imputed_data)

        related_part_of_imputed_data = pd.concat([G1_S_related_part_of_imputed_data,
                                                  G2_M_related_part_of_imputed_data])

        write_csv(G1_S_related_part_of_imputed_data,
                  os.path.join(result_dir, "files", "G1_S_related_part_of_imputed_data.csv"))
        info.append({'filename': 'G1_S_related_part_of_imputed_data.csv',
                     'description': 'Vales of genes related to G1/S',
                     'plot_description': 'Heatmap of Genes related to G1/S',
                     })

        write_csv(G2_M_related_part_of_imputed_data,
                  os.path.join(result_dir, "files", "G2_M_related_part_of_imputed_data.csv"))
        info.append({'filename': 'G2_M_related_part_of_imputed_data.csv',
                     'description': 'Vales of genes related to G2/M',
                     'plot_description': 'Heatmap of Genes related to G2/M',
                     })

        svm_results, knn_results = self._get_classification_results(related_part_of_imputed_data,
                                                                    gold_standard_classes)

        embedded_data_file_path = os.path.join(result_dir, "files", "embedded_data.pkl.gz")
        if os.path.exists(embedded_data_file_path) and not clear_cache:
            embedded_data = load_gzip_pickle(embedded_data_file_path)
        else:
            embedded_data = self._get_embeddings(related_part_of_imputed_data)
            dump_gzip_pickle(embedded_data, embedded_data_file_path)

        metric_results = {
            "classification_svm_mean_accuracy": np.mean(svm_results),
            "classification_knn_mean_accuracy": np.mean(knn_results)
        }

        embedded_data["identity"] = related_part_of_imputed_data.transpose()

        for i, embedding_name in enumerate(embedded_data):
            emb = embedded_data[embedding_name]

            k_means = KMeans(n_clusters=3)
            k_means.fit(emb)
            clusters = k_means.predict(emb)

            embedding_slug = embedding_name.replace(" ", "_").lower()

            if embedding_name != "identity":
                embedding_df = pd.DataFrame({"X": emb[:, 0],
                                             "Y": emb[:, 1],
                                             "class": gold_standard_classes,
                                             "k_means_clusters": clusters}, index=data.columns.values)
                write_csv(embedding_df, os.path.join(result_dir, "files", "%s.csv" % embedding_slug))
                info.append({'filename': "%s.csv" % embedding_slug,
                             'description': '%s embedding of cells considering genes related '
                                            'to cell-cycle' % embedding_name,
                             'plot_description': '%s embedding of cells considering genes related '
                                                 'to cell-cycle (K-means clusters are marked '
                                                 'with different shapes)' % embedding_name,
                             })

            metric_results.update({
                'kmeans_on_%s_adjusted_mutual_info_score' % embedding_slug:
                    adjusted_mutual_info_score(gold_standard_classes, clusters, average_method="arithmetic"),
                'kmeans_on_%s_v_measure_score' % embedding_slug:
                    v_measure_score(gold_standard_classes, clusters),
                'embedding_%s_calinski_harabaz_score' % embedding_slug:
                    calinski_harabaz_score(emb, gold_standard_classes),
                'embedding_%s_silhouette_score' % embedding_slug:
                    silhouette_score(emb, gold_standard_classes)
            })

        write_csv(pd.DataFrame(info), os.path.join(result_dir, "files", "info.csv"))

        result_path = os.path.join(result_dir, "result.txt")
        with open(result_path, 'w') as file:
            file.write("## METRICS:\n")
            for metric in sorted(metric_results):
                file.write("%s\t%4f\n" % (metric, metric_results[metric]))

            file.write("##\n## ADDITIONAL INFO:\n")
            file.write("## SVM classifiers accuracies: %s\n" % str(svm_results))
            file.write("## KNN classifiers accuracies: %s\n" % str(knn_results))

        log("Evaluation results saved to `%s`" % result_path)

        if visualization != "none":
            self.visualize_result(result_dir, output_type=visualization)

        return metric_results

    def visualize_result(self, result_dir, output_type, **kwargs):
        info = read_table_file(os.path.join(result_dir, "files", "info.csv"))
        info = info.set_index("filename")

        G1_S_related_part_of_imputed_data = read_table_file(os.path.join(result_dir, "files",
                                                                         "G1_S_related_part_of_imputed_data.csv"))
        G2_M_related_part_of_imputed_data = read_table_file(os.path.join(result_dir, "files",
                                                                         "G2_M_related_part_of_imputed_data.csv"))

        embeddings = ["PCA", "ICA", "Truncated SVD", "tSNE", "UMAP"]
        embedded_dfs = dict()
        for embedding_name in embeddings:
            embedding_slug = embedding_name.replace(" ", "_").lower()
            embedded_dfs[embedding_name] = read_table_file(os.path.join(result_dir,
                                                                         "files", "%s.csv" % embedding_slug))

        if output_type == "pdf":
            import plotly.graph_objs as go
            import plotly.io as pio

            G1_S_heatmap_fig = go.Figure(layout=go.Layout(title='Heatmap of Genes related to G1/S', font=dict(size=5),
                                                          xaxis=dict(title='Marker Genes', tickangle=60)))
            G2_M_heatmap_fig = go.Figure(layout=go.Layout(title='Heatmap of Genes related to G2/M', font=dict(size=5),
                                                          xaxis=dict(title='Marker Genes', tickangle=60)))

            def normalize(df):
                return df.subtract(df.mean(axis=1), axis=0).divide(df.std(axis=1), axis=0)

            G1_S_heatmap_fig.add_heatmap(z=normalize(G1_S_related_part_of_imputed_data).values.T,
                                         x=G1_S_related_part_of_imputed_data.index.values,
                                         y=G1_S_related_part_of_imputed_data.columns.values,
                                         colorscale='Viridis')
            G2_M_heatmap_fig.add_heatmap(z=normalize(G2_M_related_part_of_imputed_data).values.T,
                                         x=G2_M_related_part_of_imputed_data.index.values,
                                         y=G2_M_related_part_of_imputed_data.columns.values,
                                         colorscale='Viridis')

            pio.write_image(G1_S_heatmap_fig, os.path.join(result_dir, "plot_G1_S_related_genes_heatmap.pdf"),
                            width=600, height=700)
            pio.write_image(G2_M_heatmap_fig, os.path.join(result_dir, "plot_G2_M_related_genes_heatmap.pdf"),
                            width=600, height=700)

            embeddings = ["PCA", "ICA", "Truncated SVD", "tSNE", "UMAP"]
            for i, embedding_name in enumerate(embeddings):
                embedding_slug = embedding_name.replace(" ", "_").lower()

                fig = go.Figure(layout=go.Layout(title='%s embedding of cells considering genes related '
                                                       'to cell-cycle (K-means clusters are marked '
                                                       'with different shapes)' % embedding_name,
                                                 font=dict(size=8)))

                embedding_df = embedded_dfs[embedding_name]
                X = embedding_df["X"].values
                Y = embedding_df["Y"].values
                classes = embedding_df["class"].values
                clusters = embedding_df["k_means_clusters"].values

                for j, state in enumerate(["G1", "G2M", "S"]):
                    indices = [k for k, c in enumerate(classes) if c == state]
                    fig.add_scatter(x=X[indices], y=Y[indices], mode='markers',
                                    marker=dict(color=["red", "green", "blue"][j],
                                                symbol=[["circle-open", "diamond", "cross"][c]
                                                        for c in clusters[indices]]
                                                ),
                                    name="%s Phase" % state)

                pio.write_image(fig, os.path.join(result_dir, "plot_%s.pdf" % embedding_slug),
                                width=800, height=600)

        elif output_type == "html":
            raise NotImplementedError()
        else:
            raise NotImplementedError()
