import os

# Addresses
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES_DIR = os.path.join(os.path.join(BASE_DIR, "files"))
STATIC_FILES_DIR = os.path.join(os.path.join(BASE_DIR, "static"))
CACHE_DIR = os.path.join(os.path.join(FILES_DIR, "cache"))


# Neural net settings
EPSILON = 1e-7


# Available data sets
data_sets = {
    'ERP006670': "DataSet_ERP006670",
    'CELL_CYCLE': "DataSet_ERP006670",
    'GSE60361': "DataSet_GSE60361",
    'CORTEX_3005': "DataSet_GSE60361",
    'SRP041736': "DataSet_SRP041736",
    'POLLEN': "DataSet_SRP041736",
    'SRP041736-HQ': "DataSet_SRP041736_HQ",
    'POLLEN-HQ': "DataSet_SRP041736_HQ",
    'SRP041736-LQ': "DataSet_SRP041736_LQ",
    'POLLEN-LQ': "DataSet_SRP041736_LQ",
    'GSE100866': "DataSet_GSE100866",
    'CITE_Seq': "DataSet_GSE100866",
    'CITE-CBMC': "DataSet_GSE100866_CBMC",
    'CITE-PBMC': "DataSet_GSE100866_PBMC",
    'CITE-CD8': "DataSet_GSE100866_CD8",
    'GSE84133': "DataSet_GSE84133",
    'BARON': "DataSet_GSE84133",
    'GSE84133-HUMAN': "DataSet_GSE84133_Human",
    'BARON-HUMAN': "DataSet_GSE84133_Human",
    'GSE84133-MOUSE': "DataSet_GSE84133_Mouse",
    'BARON-MOUSE': "DataSet_GSE84133_Mouse",
    '10xPBMC4k': "DataSet_10xPBMC4k",
    'GTEx': "DataSet_GTEx"
}

for key in data_sets:
    data_sets[key] = "scFNN.data.data_set." + data_sets[key]
