import abc
import csv
import os

import numpy as np
import pandas as pd
import six
from scipy.io import mmread

from scFNN.data.io import read_csv
from scFNN.general.conf import settings
from scFNN.general.utils import make_sure_dir_exists, download_file_if_not_exists, extract_compressed_file, \
    load_class, log


def get_data_set_class(dataset_id):
    return load_class(settings.data_sets[dataset_id])


@six.add_metaclass(abc.ABCMeta)
class DataSet:

    @abc.abstractmethod
    def prepare(self):
        pass

    @abc.abstractmethod
    def keys(self):
        pass

    @abc.abstractmethod
    def get(self, key):
        pass

    @abc.abstractmethod
    def info(self):
        pass


class DataSet_ERP006670(DataSet):
    DATA_SET_URL = "https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-2805/E-MTAB-2805.processed.1.zip"
    DATA_SET_MD5_SUM = "6e9f2611d670e14bb0fe750682843e10"

    def __init__(self):
        self.DATA_SET_URL = DataSet_ERP006670.DATA_SET_URL
        self.DATA_SET_MD5_SUM = DataSet_ERP006670.DATA_SET_MD5_SUM

        self.DATA_SET_FILE_PATH = os.path.join("cell_cycle", os.path.basename(self.DATA_SET_URL))
        self.DATA_SET_FILE_PATH = os.path.join(settings.CACHE_DIR, self.DATA_SET_FILE_PATH)
        self.DATA_SET_DIR = "%s.d" % self.DATA_SET_FILE_PATH

        self.G1_DATA_PATH = os.path.join(self.DATA_SET_DIR, "G1_singlecells_counts.txt")
        self.G2M_DATA_PATH = os.path.join(self.DATA_SET_DIR, "G2M_singlecells_counts.txt")
        self.S_DATA_PATH = os.path.join(self.DATA_SET_DIR, "S_singlecells_counts.txt")

        self.KEYS = ["G1", "G2M", "S", "data"]

    def _download_data_set(self):
        make_sure_dir_exists(os.path.dirname(self.DATA_SET_FILE_PATH))
        download_file_if_not_exists(self.DATA_SET_URL,
                                    self.DATA_SET_FILE_PATH,
                                    self.DATA_SET_MD5_SUM)

    def _extract_data_set(self):
        extract_compressed_file(self.DATA_SET_FILE_PATH, self.DATA_SET_DIR)

    def prepare(self):
        self._download_data_set()
        self._extract_data_set()

    def keys(self):
        return self.KEYS

    def _merge_and_return_data(self):
        data_G1 = self.get("G1")
        data_G2M = self.get("G2M")
        data_S = self.get("S")

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

    def get(self, key):
        assert key in self.keys()

        if key == "data":
            return self._merge_and_return_data()

        key_to_data_path_mapping = {
            "G1": self.G1_DATA_PATH,
            "G2M": self.G2M_DATA_PATH,
            "S": self.S_DATA_PATH
        }

        data = pd.read_csv(key_to_data_path_mapping[key], sep="\t")
        return data

    def info(self):
        return dict(
            publication_link="https://www.nature.com/articles/nbt.3102",
            url=self.DATA_SET_URL,
            umi=False,
            isolation="FACS",
            technology="Fluidigm C1"
        )


class DataSet_10xPBMC4k(DataSet):
    DATA_SET_URL = "http://cf.10xgenomics.com/samples/cell-exp/2.1.0/pbmc4k/" \
                   "pbmc4k_filtered_gene_bc_matrices.tar.gz"
    DATA_SET_MD5_SUM = "f61f4deca423ef0fa82d63fdfa0497f7"

    def __init__(self):
        self.DATA_SET_URL = DataSet_10xPBMC4k.DATA_SET_URL
        self.DATA_SET_MD5_SUM = DataSet_10xPBMC4k.DATA_SET_MD5_SUM

        self.DATA_SET_FILE_PATH = os.path.join("PBMC4k", os.path.basename(self.DATA_SET_URL))
        self.DATA_SET_FILE_PATH = os.path.join(settings.CACHE_DIR, self.DATA_SET_FILE_PATH)
        self.DATA_SET_DIR = "%s.d" % self.DATA_SET_FILE_PATH

        genome = "GRCh38"
        files_dir = os.path.join(self.DATA_SET_DIR, "filtered_gene_bc_matrices", genome)
        self.BARCODE_DATA_PATH = os.path.join(files_dir, "barcodes.tsv")
        self.GENE_DATA_PATH = os.path.join(files_dir, "genes.tsv")
        self.MATRIX_DATA_PATH = os.path.join(files_dir, "matrix.mtx")

        self.KEYS = [genome, "data"]

    def _download_data_set(self):
        make_sure_dir_exists(os.path.dirname(self.DATA_SET_FILE_PATH))
        download_file_if_not_exists(self.DATA_SET_URL,
                                    self.DATA_SET_FILE_PATH,
                                    self.DATA_SET_MD5_SUM)

    def _extract_data_set(self):
        extract_compressed_file(self.DATA_SET_FILE_PATH, self.DATA_SET_DIR)

    def prepare(self):
        self._download_data_set()
        self._extract_data_set()

    def keys(self):
        return self.KEYS

    def get(self, key):
        assert key in self.keys()

        barcode_data = [row for row in csv.reader(open(self.BARCODE_DATA_PATH), delimiter="\t")]
        barcodes = [row[0] for row in barcode_data]

        gene_data = [row for row in csv.reader(open(self.GENE_DATA_PATH), delimiter="\t")]
        gene_ids = [row[0] for row in gene_data]
        # gene_names = [row[1] for row in gene_data]

        matrix = mmread(self.MATRIX_DATA_PATH).todense()

        data = pd.DataFrame(matrix, index=gene_ids, columns=barcodes)
        data.index.name = 'Symbol'

        return data

    def info(self):
        return dict(
            publication_link="https://www.nature.com/articles/ncomms14049",
            url=self.DATA_SET_URL,
            umi=True,
            isolation="droplet-based",
            technology="10x"
        )


class DataSet_GSE60361(DataSet):
    # DATA_SET_URL = "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE60nnn/" \
    #                "GSE60361/suppl/GSE60361_C1-3005-Expression.txt.gz"
    # DATA_SET_MD5_SUM = "fbf6f0ec39d54d8aac7233c56d0c9e30"
    DATA_SET_URL = "https://storage.googleapis.com/linnarsson-lab-www-blobs/" \
                   "blobs/cortex/expression_mRNA_17-Aug-2014.txt"
    DATA_SET_MD5_SUM = "6bb4c2a9ade87b16909d39004021849e"

    def __init__(self):
        self.DATA_SET_URL = DataSet_GSE60361.DATA_SET_URL
        self.DATA_SET_MD5_SUM = DataSet_GSE60361.DATA_SET_MD5_SUM

        self.DATA_SET_FILE_PATH = os.path.join("GSE60361", os.path.basename(self.DATA_SET_URL))
        self.DATA_SET_FILE_PATH = os.path.join(settings.CACHE_DIR, self.DATA_SET_FILE_PATH)

        self.KEYS = ["mRNA", "details", "data"]

    def _download_data_set(self):
        make_sure_dir_exists(os.path.dirname(self.DATA_SET_FILE_PATH))
        download_file_if_not_exists(self.DATA_SET_URL,
                                    self.DATA_SET_FILE_PATH,
                                    self.DATA_SET_MD5_SUM)

    def prepare(self):
        self._download_data_set()

    def keys(self):
        return self.KEYS

    def get(self, key):
        assert key in self.keys()

        if key == "mRNA" or key == "data":
            full_data = read_csv(self.DATA_SET_FILE_PATH, header=None)
            data = full_data.iloc[11:-1, 1:3006]
            data.columns = full_data.iloc[7, 1:3006]
            data.index.name = ""
            data.columns.name = ""
            data = data.astype("int64")
            return data
        elif key == "details":
            full_data = read_csv(self.DATA_SET_FILE_PATH, header=None)
            data = full_data.iloc[:10, 1:3006]
            data.columns = full_data.iloc[7, 1:3006]
            data.index = full_data.iloc[:10, 0].values
            data.index.name = ""
            data.columns.name = ""
            return data

    def info(self):
        return dict(
            publication_link="http://science.sciencemag.org/content/early/2015/02/18/science.aaa1934",
            url=self.DATA_SET_URL,
            umi=True,
            isolation="FACS",
            technology="Fluidigm C1"
        )


class DataSet_SRP041736(DataSet):
    DATA_SET_URL = "http://duffel.rail.bio/recount/v2/SRP041736/counts_gene.tsv.gz"
    DATA_SET_MD5_SUM = "535271b8cd81a93eb210254b766ebcbb"

    def __init__(self):
        self.DATA_SET_URL = DataSet_SRP041736.DATA_SET_URL
        self.DATA_SET_MD5_SUM = DataSet_SRP041736.DATA_SET_MD5_SUM

        self.DATA_SET_FILE_PATH = os.path.join(settings.CACHE_DIR,
                                               "SRP041736",
                                               os.path.basename(self.DATA_SET_URL))

        self.EXPERIMENT_INFO_FILE_PATH = os.path.join(settings.STATIC_FILES_DIR,
                                                      "data", "SRP041736_info.txt")

        self.KEYS = ["details", "HQ-details", "LQ-details", "data", "HQ-data", "LQ-data"]

    def _download_data_set(self):
        make_sure_dir_exists(os.path.dirname(self.DATA_SET_FILE_PATH))
        download_file_if_not_exists(self.DATA_SET_URL,
                                    self.DATA_SET_FILE_PATH,
                                    self.DATA_SET_MD5_SUM)

    def prepare(self):
        self._download_data_set()
        assert os.path.exists(self.EXPERIMENT_INFO_FILE_PATH)

    def keys(self):
        return self.KEYS

    def get(self, key):
        assert key in self.keys()

        info = pd.read_csv(self.EXPERIMENT_INFO_FILE_PATH, engine="python", sep="\t", comment="#")
        info = info[np.logical_and(info["Experiment"] != "SRX534506", info["Experiment"] != "SRX534553")]
        info = info.sort_values(by=["Experiment", "MBases"])
        info.index = info["Run"].values
        info["class"] = [sample_name.split("_")[0] for sample_name in info["Sample_Name"]]
        info = info.transpose()

        if key == "details":
            return info

        if key == "HQ-details":
            return info.iloc[:, range(1, 692, 2)]

        if key == "LQ-details":
            return info.iloc[:, range(0, 692, 2)]

        data = pd.read_csv(self.DATA_SET_FILE_PATH, engine="python", sep=None, index_col=-1)
        data = data.astype("int64")
        data["pure_gene_id"] = [gene_name.split(".")[0] for gene_name in list(data.index.values)]
        data = data.groupby(["pure_gene_id"]).sum()
        data.index.name = "gene_id"
        data = data[info.loc["Run"].values]

        if key == "data":
            return data

        if key == "HQ-data":
            return data.iloc[:, range(1, 692, 2)]

        if key == "LQ-data":
            return data.iloc[:, range(0, 692, 2)]

    def info(self):
        return dict(
            publication_link="https://www.nature.com/articles/nbt.2967",
            url=self.DATA_SET_URL,
            umi=False,
            isolation="FACS",
            technology="Fluidigm C1"
        )


class DataSet_SRP041736_HQ(DataSet):
    def __init__(self):
        self.ds = DataSet_SRP041736()
        self.KEYS = ["details", "data"]

    def prepare(self):
        self.ds.prepare()

    def keys(self):
        return self.KEYS

    def get(self, key):
        if key == "details":
            return self.ds.get("HQ-details")

        if key == "data":
            return self.ds.get("HQ-data")

    def info(self):
        return self.ds.info()


class DataSet_SRP041736_LQ(DataSet):
    def __init__(self):
        self.ds = DataSet_SRP041736()
        self.KEYS = ["details", "data"]

    def prepare(self):
        self.ds.prepare()

    def keys(self):
        return self.KEYS

    def get(self, key):
        if key == "details":
            return self.ds.get("LQ-details")

        if key == "data":
            return self.ds.get("LQ-data")

    def info(self):
        return self.ds.info()


class DataSet_GSE100866(DataSet):
    PBMC_RNA_DATA_URL = "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE100nnn/GSE100866/suppl/" \
                        "GSE100866_PBMC_vs_flow_10X-RNA_umi.csv.gz"
    PBMC_RNA_DATA_MD5_SUM = "5ce36806a4b17fc8385ae612eef4f8e8"
    PBMC_ADT_DATA_URL = "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE100nnn/GSE100866/suppl/" \
                        "GSE100866_PBMC_vs_flow_10X-ADT_umi.csv.gz"
    PBMC_ADT_DATA_MD5_SUM = "515146739962eaa7e114b5838cfeeecf"
    PBMC_TRANSFORMED_ADT_DATA_URL = "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE100nnn/GSE100866/suppl/" \
                                    "GSE100866_PBMC_vs_flow_10X-ADT_clr-transformed.csv.gz"
    PBMC_TRANSFORMED_ADT_DATA_MD5_SUM = "b371af3d9e06a0a33e7bc52941220343"
    CBMC_RNA_DATA_URL = "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE100nnn/GSE100866/suppl/" \
                        "GSE100866_CBMC_8K_13AB_10X-RNA_umi.csv.gz"
    CBMC_RNA_DATA_MD5_SUM = "b8d8332e5b56f689427d8c0580b0fc38"
    CBMC_ADT_DATA_URL = "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE100nnn/GSE100866/suppl/" \
                        "GSE100866_CBMC_8K_13AB_10X-ADT_umi.csv.gz"
    CBMC_ADT_DATA_MD5_SUM = "d90fed7d120317281e63826931a0e070"
    CBMC_TRANSFORMED_ADT_DATA_URL = "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE100nnn/GSE100866/suppl/" \
                                    "GSE100866_CBMC_8K_13AB_10X-ADT_clr-transformed.csv.gz"
    CBMC_TRANSFORMED_ADT_DATA_MD5_SUM = "5e53cab00dd0b073028098d52e51e565"
    CD8_RNA_DATA_URL = "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE100nnn/GSE100866/suppl/" \
                       "GSE100866_CD8_merged-RNA_umi.csv.gz"
    CD8_RNA_DATA_MD5_SUM = "bb055e35f989a19f6d0f3fbc803e65d7"
    CD8_ADT_DATA_URL = "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE100nnn/GSE100866/suppl/" \
                       "GSE100866_CD8_merged-ADT_umi.csv.gz"
    CD8_ADT_DATA_MD5_SUM = "81f47f6a8fdede82ed338199df2c792b"
    CD8_TRANSFORMED_ADT_DATA_URL = "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE100nnn/GSE100866/suppl/" \
                                   "GSE100866_CD8_merged-ADT_clr-transformed.csv.gz"
    CD8_TRANSFORMED_ADT_DATA_MD5_SUM = "a9c084f4c13760fc364087e76cfee16e"

    def __init__(self):

        self.DATA_SET_DIR_NAME = os.path.join(settings.CACHE_DIR, "GSE100866")

        self.PBMC_RNA_DATA_FILE_PATH = os.path.join(self.DATA_SET_DIR_NAME, self.PBMC_RNA_DATA_URL.split("/")[-1])
        self.PBMC_ADT_DATA_FILE_PATH = os.path.join(self.DATA_SET_DIR_NAME, self.PBMC_ADT_DATA_URL.split("/")[-1])
        self.PBMC_TRANSFORMED_ADT_DATA_FILE_PATH = os.path.join(self.DATA_SET_DIR_NAME,
                                                                self.PBMC_TRANSFORMED_ADT_DATA_URL.split("/")[-1])
        self.CBMC_RNA_DATA_FILE_PATH = os.path.join(self.DATA_SET_DIR_NAME, self.CBMC_RNA_DATA_URL.split("/")[-1])
        self.CBMC_ADT_DATA_FILE_PATH = os.path.join(self.DATA_SET_DIR_NAME, self.CBMC_ADT_DATA_URL.split("/")[-1])
        self.CBMC_TRANSFORMED_ADT_DATA_FILE_PATH = os.path.join(self.DATA_SET_DIR_NAME,
                                                                self.CBMC_TRANSFORMED_ADT_DATA_URL.split("/")[-1])
        self.CD8_RNA_DATA_FILE_PATH = os.path.join(self.DATA_SET_DIR_NAME, self.CD8_RNA_DATA_URL.split("/")[-1])
        self.CD8_ADT_DATA_FILE_PATH = os.path.join(self.DATA_SET_DIR_NAME, self.CD8_ADT_DATA_URL.split("/")[-1])
        self.CD8_TRANSFORMED_ADT_DATA_FILE_PATH = os.path.join(self.DATA_SET_DIR_NAME,
                                                               self.CD8_TRANSFORMED_ADT_DATA_URL.split("/")[-1])

        self.KEYS = ["PBMC-RNA", "CBMC-RNA", "CD8-RNA",
                     "PBMC-ADT", "CBMC-ADT", "CD8-ADT",
                     "PBMC-ADT-clr", "CBMC-ADT-clr", "CD8-ADT-clr"]

    def _download_data_set(self):
        make_sure_dir_exists(self.DATA_SET_DIR_NAME)
        download_file_if_not_exists(self.PBMC_RNA_DATA_URL, self.PBMC_RNA_DATA_FILE_PATH, self.PBMC_RNA_DATA_MD5_SUM)
        download_file_if_not_exists(self.PBMC_ADT_DATA_URL, self.PBMC_ADT_DATA_FILE_PATH, self.PBMC_ADT_DATA_MD5_SUM)
        download_file_if_not_exists(self.PBMC_TRANSFORMED_ADT_DATA_URL, self.PBMC_TRANSFORMED_ADT_DATA_FILE_PATH,
                                    self.PBMC_TRANSFORMED_ADT_DATA_MD5_SUM)
        download_file_if_not_exists(self.CBMC_RNA_DATA_URL, self.CBMC_RNA_DATA_FILE_PATH, self.CBMC_RNA_DATA_MD5_SUM)
        download_file_if_not_exists(self.CBMC_ADT_DATA_URL, self.CBMC_ADT_DATA_FILE_PATH, self.CBMC_ADT_DATA_MD5_SUM)
        download_file_if_not_exists(self.CBMC_TRANSFORMED_ADT_DATA_URL, self.CBMC_TRANSFORMED_ADT_DATA_FILE_PATH,
                                    self.CBMC_TRANSFORMED_ADT_DATA_MD5_SUM)
        download_file_if_not_exists(self.CD8_RNA_DATA_URL, self.CD8_RNA_DATA_FILE_PATH, self.CD8_RNA_DATA_MD5_SUM)
        download_file_if_not_exists(self.CD8_ADT_DATA_URL, self.CD8_ADT_DATA_FILE_PATH, self.CD8_ADT_DATA_MD5_SUM)
        download_file_if_not_exists(self.CD8_TRANSFORMED_ADT_DATA_URL, self.CD8_TRANSFORMED_ADT_DATA_FILE_PATH,
                                    self.CD8_TRANSFORMED_ADT_DATA_MD5_SUM)

    def _extract_PBMC_human_cells(self):
        human_cells_file_path = os.path.join(self.DATA_SET_DIR_NAME, "PBMC-human-cells.csv")

        if os.path.exists(human_cells_file_path):
            log("PBMC human cells are already extracted.")
            return

        data = pd.read_csv(self.PBMC_RNA_DATA_FILE_PATH, index_col=0)

        human_section = data.loc[[gene for gene in data.index.values if gene.startswith("HUMAN")]]
        mouse_section = data.loc[[gene for gene in data.index.values if gene.startswith("MOUSE")]]

        pd.DataFrame(data.columns.values[human_section.sum(axis=0) > 20 * mouse_section.sum(axis=0)],
                     columns=["human"]).to_csv(human_cells_file_path, index=None)

        log("PBMC human cells extracted.")

    def _extract_CBMC_human_cells(self):
        human_cells_file_path = os.path.join(self.DATA_SET_DIR_NAME, "CBMC-human-cells.csv")

        if os.path.exists(human_cells_file_path):
            log("CBMC human cells are already extracted.")
            return

        data = pd.read_csv(self.CBMC_RNA_DATA_FILE_PATH, index_col=0)

        human_section = data.loc[[gene for gene in data.index.values if gene.startswith("HUMAN")]]
        mouse_section = data.loc[[gene for gene in data.index.values if gene.startswith("MOUSE")]]

        # 20x is enough (See Supplementary Figure 2 of Cite-Seq paper.)
        pd.DataFrame(data.columns.values[human_section.sum(axis=0) > 20 * mouse_section.sum(axis=0)],
                     columns=["human"]).to_csv(human_cells_file_path , index=None)

        log("CBMC human cells extracted.")

    def prepare(self):
        self._download_data_set()
        self._extract_CBMC_human_cells()
        self._extract_PBMC_human_cells()

    def keys(self):
        return self.KEYS

    def get(self, key):
        assert key in self.keys()

        key_to_filename_mapping = {
            "PBMC-RNA": self.PBMC_RNA_DATA_FILE_PATH,
            "CBMC-RNA": self.CBMC_RNA_DATA_FILE_PATH,
            "CD8-RNA": self.CD8_RNA_DATA_FILE_PATH,
            "PBMC-ADT": self.PBMC_ADT_DATA_FILE_PATH,
            "CBMC-ADT": self.CBMC_ADT_DATA_FILE_PATH,
            "CD8-ADT": self.CD8_ADT_DATA_FILE_PATH,
            "PBMC-ADT-clr": self.PBMC_TRANSFORMED_ADT_DATA_URL,
            "CBMC-ADT-clr": self.CBMC_TRANSFORMED_ADT_DATA_URL,
            "CD8-ADT-clr": self.CD8_TRANSFORMED_ADT_DATA_URL
        }

        data = pd.read_csv(key_to_filename_mapping[key], index_col=0)

        experiment = key.split("-")[0]

        if experiment == "CBMC":
            human_cells_file_path = os.path.join(self.DATA_SET_DIR_NAME, "CBMC-human-cells.csv")
            human_cells = pd.read_csv(human_cells_file_path)["human"]
            data = data[human_cells]
            if key.endswith("RNA"):
                human_genes = [gene for gene in data.index.values if gene.startswith("HUMAN")]
                data = data.loc[human_genes]
                data.index = [gene.split("_", maxsplit=1)[1] for gene in data.index.values]
            return data
        elif experiment == "PBMC":
            human_cells_file_path = os.path.join(self.DATA_SET_DIR_NAME, "PBMC-human-cells.csv")
            human_cells = pd.read_csv(human_cells_file_path)["human"]
            data = data[human_cells]
            if key.endswith("RNA"):
                human_genes = [gene for gene in data.index.values if gene.startswith("HUMAN")]
                data = data.loc[human_genes]
                data.index = [gene.split("_", maxsplit=1)[1] for gene in data.index.values]
            return data
        elif experiment == "CD8":
            return data

    def info(self):
        return dict(
            publication_link="https://www.nature.com/articles/nmeth.4380",
            url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE100866",
            umi=True,
            isolation="droplet-based",
            technology="mixed"
        )


class DataSet_GSE100866_CBMC(DataSet):
    def __init__(self):
        self.ds = DataSet_GSE100866()
        self.KEYS = ["RNA", "ADT", "ADT-clr", "data"]

    def prepare(self):
        self.ds.prepare()

    def keys(self):
        return self.KEYS

    def get(self, key):
        if key == "data":
            key = "RNA"
        return self.ds.get("CBMC-%s" % key)

    def info(self):
        return dict(
            publication_link="https://www.nature.com/articles/nmeth.4380",
            url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE100866",
            umi=True,
            isolation="droplet-based",
            technology="10x"
        )


class DataSet_GSE100866_PBMC(DataSet):
    def __init__(self):
        self.ds = DataSet_GSE100866()
        self.KEYS = ["RNA", "ADT", "ADT-clr", "data"]

    def prepare(self):
        self.ds.prepare()

    def keys(self):
        return self.KEYS

    def get(self, key):
        if key == "data":
            key = "RNA"
        return self.ds.get("PBMC-%s" % key)

    def info(self):
        return dict(
            publication_link="https://www.nature.com/articles/nmeth.4380",
            url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE100866",
            umi=True,
            isolation="droplet-based",
            technology="Drop-seq"
        )


class DataSet_GSE100866_CD8(DataSet):
    def __init__(self):
        self.ds = DataSet_GSE100866()
        self.KEYS = ["RNA", "ADT", "ADT-clr", "data"]

    def prepare(self):
        self.ds.prepare()

    def keys(self):
        return self.KEYS

    def get(self, key):
        if key == "data":
            key = "RNA"
        return self.ds.get("CD8-%s" % key)

    def info(self):
        return dict(
            publication_link="https://www.nature.com/articles/nmeth.4380",
            url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE100866",
            umi=True,
            isolation="droplet-based",
            technology="Drop-seq"
        )


class DataSet_GSE84133(DataSet):
    DATA_SET_URL = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE84133&format=file"
    DATA_SET_MD5_SUM = "e9171075e0c45ab1aa1c2f7374e006d3"

    def __init__(self):
        self.DATA_SET_URL = self.DATA_SET_URL
        self.DATA_SET_MD5_SUM = self.DATA_SET_MD5_SUM

        self.DATA_SET_FILE_PATH = os.path.join(settings.CACHE_DIR, "GSE84133", "GSE84133_RAW.tar")
        self.DATA_SET_DIR = "%s.d" % self.DATA_SET_FILE_PATH

        self.human1_DATA_PATH = os.path.join(self.DATA_SET_DIR, "GSM2230757_human1_umifm_counts.csv.gz")
        self.human2_DATA_PATH = os.path.join(self.DATA_SET_DIR, "GSM2230758_human2_umifm_counts.csv.gz")
        self.human3_DATA_PATH = os.path.join(self.DATA_SET_DIR, "GSM2230759_human3_umifm_counts.csv.gz")
        self.human4_DATA_PATH = os.path.join(self.DATA_SET_DIR, "GSM2230760_human4_umifm_counts.csv.gz")
        self.mouse1_DATA_PATH = os.path.join(self.DATA_SET_DIR, "GSM2230761_mouse1_umifm_counts.csv.gz")
        self.mouse2_DATA_PATH = os.path.join(self.DATA_SET_DIR, "GSM2230762_mouse2_umifm_counts.csv.gz")

        self.KEYS = ["human1", "human2", "human3", "human4",
                     "mouse1", "mouse2",
                     "human1-details", "human2-details", "human3-details", "human4-details",
                     "mouse1-details", "mouse2-details"]

    def _download_data_set(self):
        make_sure_dir_exists(os.path.dirname(self.DATA_SET_FILE_PATH))
        download_file_if_not_exists(self.DATA_SET_URL,
                                    self.DATA_SET_FILE_PATH,
                                    self.DATA_SET_MD5_SUM)

    def _extract_data_set(self):
        extract_compressed_file(self.DATA_SET_FILE_PATH, self.DATA_SET_DIR)

    def prepare(self):
        self._download_data_set()
        self._extract_data_set()

    def keys(self):
        return self.KEYS

    def get(self, key):
        assert key in self.keys()

        key_to_file_mapping = {
            "human1": self.human1_DATA_PATH,
            "human2": self.human2_DATA_PATH,
            "human3": self.human3_DATA_PATH,
            "human4": self.human4_DATA_PATH,
            "mouse1": self.mouse1_DATA_PATH,
            "mouse2": self.mouse2_DATA_PATH
        }

        data = read_csv(key_to_file_mapping[key.split("-")[0]])
        data = data.transpose()

        if key.endswith("-details"):
            data = data.loc[["barcode", "assigned_cluster"]]
        else:
            data = data.drop(index=["barcode", "assigned_cluster"])
            data = data.astype("int64")

        return data

    def info(self):
        return dict(
            publication_link="https://www.sciencedirect.com/science/article/pii/S2405471216302666?via%3Dihub",
            url=self.DATA_SET_URL,
            umi=False,
            isolation="droplet-based",
            technology="inDrop"
        )


class DataSet_GSE84133_Human(DataSet):
    def __init__(self):
        self.ds = DataSet_GSE84133()
        self.KEYS = ["data", "details"]

    def prepare(self):
        self.ds.prepare()

    def keys(self):
        return self.KEYS

    def get(self, key):
        if key == "data":
            human1 = self.ds.get("human1")
            human2 = self.ds.get("human2")
            human3 = self.ds.get("human3")
            human4 = self.ds.get("human4")

            assert np.all(human1.index.values==human2.index.values) and \
                   np.all(human1.index.values==human2.index.values) and \
                   np.all(human1.index.values==human3.index.values)

            return pd.concat([human1, human2, human3, human4], axis=1)
        elif key == "details":
            human1 = self.ds.get("human1-details")
            human2 = self.ds.get("human2-details")
            human3 = self.ds.get("human3-details")
            human4 = self.ds.get("human4-details")

            return pd.concat([human1, human2, human3, human4], axis=1)
        else:
            raise NotImplementedError()

    def info(self):
        return self.ds.info()


class DataSet_GSE84133_Mouse(DataSet):
    def __init__(self):
        self.ds = DataSet_GSE84133()
        self.KEYS = ["data", "details"]

    def prepare(self):
        self.ds.prepare()

    def keys(self):
        return self.KEYS

    def get(self, key):
        if key == "data":
            mouse1 = self.ds.get("mouse1")
            mouse2 = self.ds.get("mouse2")

            assert np.all(mouse1.index.values == mouse2.index.values)

            return pd.concat([mouse1, mouse2], axis=1)
        elif key == "details":
            mouse1 = self.ds.get("mouse1-details")
            mouse2 = self.ds.get("mouse2-details")

            return pd.concat([mouse1, mouse2], axis=1)
        else:
            raise NotImplementedError()

    def info(self):
        return self.ds.info()


class DataSet_GTEx(DataSet):
    # DATA_SET_URL = "https://storage.googleapis.com/gtex_analysis_v7/rna_seq_data" \
    #                "/GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_reads.gct.gz"
    # DATA_SET_MD5_SUM = "8e7fce1e7e93749a8c48d301e9d848f5"
    DATA_SET_URL = "https://storage.googleapis.com/gtex_analysis_v7/rna_seq_data" \
                   "/GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_tpm.gct.gz"
    DATA_SET_MD5_SUM = "e027506deca6a62f50e316561c6cb8cb"

    ANNOTATION_URL = "https://storage.googleapis.com/gtex_analysis_v7/" \
                     "annotations/GTEx_v7_Annotations_SampleAttributesDS.txt"
    ANNOTATION_MD5_SUM = "80f2b89d45843abb0a0dc39d5054530c"

    def __init__(self):
        self.DATA_SET_URL = DataSet_GTEx.DATA_SET_URL
        self.DATA_SET_MD5_SUM = DataSet_GTEx.DATA_SET_MD5_SUM

        self.DATA_SET_FILE_PATH = os.path.join(settings.CACHE_DIR,
                                               "GTEx",
                                               os.path.basename(self.DATA_SET_URL))

        self.ANNOTATION_URL = DataSet_GTEx.ANNOTATION_URL
        self.ANNOTATION_MD5_SUM = DataSet_GTEx.ANNOTATION_MD5_SUM

        self.ANNOTATION_FILE_PATH = os.path.join(settings.CACHE_DIR,
                                                 "GTEx",
                                                 os.path.basename(self.ANNOTATION_URL))

        self.KEYS = ["details", "data"]

    def _download_data_set(self):
        make_sure_dir_exists(os.path.dirname(self.DATA_SET_FILE_PATH))
        download_file_if_not_exists(self.DATA_SET_URL,
                                    self.DATA_SET_FILE_PATH,
                                    self.DATA_SET_MD5_SUM)
        download_file_if_not_exists(self.ANNOTATION_URL,
                                    self.ANNOTATION_FILE_PATH,
                                    self.ANNOTATION_MD5_SUM)

    def prepare(self):
        self._download_data_set()

    def keys(self):
        return self.KEYS

    def get(self, key):
        assert key in self.keys()

        info = pd.read_csv(self.ANNOTATION_FILE_PATH, engine="python", sep="\t", index_col=0)
        info = info.transpose()

        if key == "details":
            return info

        data = pd.read_csv(self.DATA_SET_FILE_PATH, engine="python", sep=None, index_col=0, skiprows=2)
        data.drop("Description", axis=1, inplace=True)

        # Remove version from Entrez gene id
        data.index = [gene_id.split(".")[0] for gene_id in data.index.values]
        data.index.name = "gene_id"

        if key == "data":
            return data

    def info(self):
        return dict(
            publication_link="https://www.liebertpub.com/doi/full/10.1089/bio.2015.0032",
            url=self.DATA_SET_URL,
            umi=False,
            isolation="None",
            technology="RNA-seq"
        )


class DataSet_FANTOM5_HUMAN(DataSet):
    DATA_SET_URL = "http://fantom.gsc.riken.jp/5/datafiles/latest/extra/CAGE_peaks/" \
                   "hg19.cage_peak_phase1and2combined_tpm_ann.osc.txt.gz"
    DATA_SET_MD5_SUM = "0b288555ef51d1f1f9f04a2536d51a1d"

    ANNOTATION_URL = "http://fantom.gsc.riken.jp/5/datafiles/latest/extra/CAGE_peaks/" \
                     "hg19.cage_peak_phase1and2combined_ann.txt.gz"
    ANNOTATION_MD5_SUM = "9c2d2cbfebfff849e18308319d7d691f"

    def __init__(self):
        self.DATA_SET_URL = DataSet_GTEx.DATA_SET_URL
        self.DATA_SET_MD5_SUM = DataSet_GTEx.DATA_SET_MD5_SUM

        self.DATA_SET_FILE_PATH = os.path.join(settings.CACHE_DIR,
                                               "FANTOM5",
                                               os.path.basename(self.DATA_SET_URL))

        self.ANNOTATION_URL = DataSet_GTEx.ANNOTATION_URL
        self.ANNOTATION_MD5_SUM = DataSet_GTEx.ANNOTATION_MD5_SUM

        self.ANNOTATION_FILE_PATH = os.path.join(settings.CACHE_DIR,
                                                 "FANTOM5",
                                                 os.path.basename(self.ANNOTATION_URL))

        self.KEYS = ["data"]

    def _download_data_set(self):
        make_sure_dir_exists(os.path.dirname(self.DATA_SET_FILE_PATH))
        download_file_if_not_exists(self.DATA_SET_URL,
                                    self.DATA_SET_FILE_PATH,
                                    self.DATA_SET_MD5_SUM)
        download_file_if_not_exists(self.ANNOTATION_URL,
                                    self.ANNOTATION_FILE_PATH,
                                    self.ANNOTATION_MD5_SUM)

    def prepare(self):
        self._download_data_set()

    def keys(self):
        return self.KEYS

    def get(self, key):
        assert key in self.keys()

        info = pd.read_csv(self.ANNOTATION_FILE_PATH, engine="python", sep="\t", index_col=0, comment="#")
        info = info.transpose()

        if key == "details":
            return info

        data = pd.read_csv(self.DATA_SET_FILE_PATH, engine="python", sep=None, index_col=0, skiprows=2)
        data.drop("Description", axis=1, inplace=True)

        # Remove version from Entrez gene id
        data.index = [gene_id.split(".")[0] for gene_id in data.index.values]
        data.index.name = "gene_id"

        if key == "data":
            return data

    def info(self):
        return dict(
            publication_link="https://www.liebertpub.com/doi/full/10.1089/bio.2015.0032",
            url=self.DATA_SET_URL,
            umi=False,
            isolation="None",
            technology="RNA-seq"
        )
