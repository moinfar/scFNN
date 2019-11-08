import os


DEBUG = True

# Addresses
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES_DIR = os.path.join(os.path.join(BASE_DIR, "files"))
STATIC_FILES_DIR = os.path.join(os.path.join(BASE_DIR, "static"))

CACHE_DIR = os.path.join(os.path.join(FILES_DIR, "cache"))
STORAGE_DIR = os.path.join(os.path.join(FILES_DIR, "storage"))

# Available data sets
data_sets = {
    'ERP006670': "data.data_set.DataSet_ERP006670",
    'CELL_CYCLE': "data.data_set.DataSet_ERP006670",
    'GSE60361': "data.data_set.DataSet_GSE60361",
    'CORTEX_3005': "data.data_set.DataSet_GSE60361",
    'SRP041736': "data.data_set.DataSet_SRP041736",
    'POLLEN': "data.data_set.DataSet_SRP041736",
    'SRP041736-HQ': "data.data_set.DataSet_SRP041736_HQ",
    'POLLEN-HQ': "data.data_set.DataSet_SRP041736_HQ",
    'SRP041736-LQ': "data.data_set.DataSet_SRP041736_LQ",
    'POLLEN-LQ': "data.data_set.DataSet_SRP041736_LQ",
    'GSE100866': "data.data_set.DataSet_GSE100866",
    'CITE_Seq': "data.data_set.DataSet_GSE100866",
    'CITE-CBMC': "data.data_set.DataSet_GSE100866_CBMC",
    'CITE-PBMC': "data.data_set.DataSet_GSE100866_PBMC",
    'CITE-CD8': "data.data_set.DataSet_GSE100866_CD8",
    'GSE84133': "data.data_set.DataSet_GSE84133",
    'BARON': "data.data_set.DataSet_GSE84133",
    'GSE84133-HUMAN': "data.data_set.DataSet_GSE84133_Human",
    'BARON-HUMAN': "data.data_set.DataSet_GSE84133_Human",
    'GSE84133-MOUSE': "data.data_set.DataSet_GSE84133_Mouse",
    'BARON-MOUSE': "data.data_set.DataSet_GSE84133_Mouse",
    '10xPBMC4k': "data.data_set.DataSet_10xPBMC4k"
}
