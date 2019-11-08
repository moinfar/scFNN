import numpy as np
from torch.utils.data import Dataset

from scFNN.data.data_set import get_data_set_class
from scFNN.data.io import read_table_file


class DataFrameDataSet(Dataset):
    def __init__(self, data_frame, dtype=np.float32, ref=None, transpose=True):
        super(DataFrameDataSet, self).__init__()
        self.data_frame = data_frame.astype(dtype)
        if transpose:
            self.data_frame = self.data_frame.transpose()

        if ref is not None:
            drop_columns = np.setdiff1d(self.data_frame.columns.values, ref)
            print("Dropping {} genes not in reference genes.".format(len(drop_columns)))
            self.data_frame.drop(columns=drop_columns, inplace=True)

    def __getitem__(self, index):
        slice = self.data_frame.iloc[[index]]
        return tuple([slice])

    def __len__(self):
        return self.data_frame.shape[0]

    @property
    def n_genes(self):
        return self.data_frame.shape[1]

    @property
    def genes(self):
        return self.data_frame.columns.values

    @property
    def cells(self):
        return self.data_frame.index.values


class CsvFileDataSet(DataFrameDataSet):
    def __init__(self, filename, ref=None, transpose=True):
        self.data_frame = read_table_file(filename)
        super(CsvFileDataSet, self).__init__(self.data_frame, ref=ref, transpose=transpose)


class PredefinedDataSet(DataFrameDataSet):
    def __init__(self, data_set_name, ref=None):
        self.data_set = get_data_set_class(data_set_name)()
        self.data_set.prepare()
        self.data_frame = self.data_set.get("data")

        super(PredefinedDataSet, self).__init__(data_frame=self.data_frame, ref=ref, transpose=True)
