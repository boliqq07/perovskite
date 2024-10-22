import os
import os.path as osp
from shutil import rmtree
from typing import List, Callable, Dict, Union, Tuple

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.data import InMemoryDataset


class SimpleDataset(Dataset):
    """Data list with shuffle.
    (For small data <= 500.)
    Load data from Memory, and stored in Memory for use.

    Examples
    -----------

    >>> #Not with Dataset
    >>> # with data.edge_index shape (2,num_edge)
    >>> dataset = [Data.from_dict(di) for di in data]
    >>> # SparseTensor: with data.edge_index shape (num_node,num_node)
    >>> import torch_geometric.transforms as T
    >>> dataset2 = [T.ToSparseTensor(Data.from_dict(di))) for di in data]

    >>> # with SimpleDataset
    >>> # with data.edge_index shape (2,num_edge)
    >>> dataset = SimpleDataset(data)
    >>> # SparseTensor: with data.edge_index shape (num_node,num_node)
    >>> import torch_geometric.transforms as T
    >>> dataset2 = SimpleDataset(data,pre_transform=T.ToSparseTensor())

    The result is list. If needed, it could be convertd BatchData by:
    >>> from torch_geometric.data.batch import Batch
    >>> data = Batch.from_data_list(dataset2)

    """

    def __init__(self, data: Union[List[Data], List[Dict]], pre_filter=None, pre_transform=None,
                 transform: Callable = None, ):
        """

        Args:
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            pre_filter (callable, optional): A function that takes in an
                :obj:`torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset. (default: :obj:`None`)

        """
        if isinstance(data[0], dict):
            self.data = [Data.from_dict(di) for di in data]
        else:
            self.data = data

        self.__indices__ = None
        super(SimpleDataset, self).__init__(root=".",
                                            transform=transform,
                                            pre_transform=pre_transform,
                                            pre_filter=pre_filter)
        self.process()

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        r"""The name of the files in the :obj:`self.raw_dir` folder that must
        be present in order to skip downloading."""
        return ""

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        r"""The name of the files in the :obj:`self.processed_dir` folder that
        must be present in order to skip processing."""
        return ""

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]

    @property
    def num_features(self):
        r"""Returns the number of features per node in the dataset."""
        return self[0].num_features

    @property
    def num_node_features(self):
        r"""alias for num_features."""
        return self[0].num_features

    @property
    def num_edge_features(self):
        r"""Returns the number of features per edge in the dataset."""
        return self[0].num_edge_features

    def process(self):
        # Read data into huge `Data` list.
        data_list = self.data

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.data = data_list

    def to_mtbatch_data(self):
        from hpc2ml.data.batchdata import MtBatchData
        return MtBatchData.from_data_list(self.data)


class InMemoryDatasetGeo(InMemoryDataset):
    """InMemoryDataset for materials.
    (For small data <= 10000.)
    Load data from local disk, and stored in Memory for use.

    Examples
    -----------
    >>> # with data.edge_index shape (2,num_edge)
    >>> dataset = InMemoryDatasetGeo(root=".")
    >>> # SparseTensor: with data.edge_index shape (num_node,num_node)
    >>> import torch_geometric.transforms as T
    >>> dataset2 = InMemoryDatasetGeo(root=".", pre_transform=T.ToSparseTensor())

    """

    def __init__(self, root=".", pre_transform=None, pre_filter=None, re_process_init=True, transform=None, load_mode="o"):
        """

        Args:
            load_mode (str): load from the independent data in different files: "i",
                load from the batch data in one overall file: "o".
            re_process_init (bool): process raw data or not. if there is no raw data but processed data is offered.
                this parameter could be False. Because the default
            root (string, optional): Root directory where the dataset should be
                saved. (optional: :obj:`None`)
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            pre_filter (callable, optional): A function that takes in an
                :obj:`torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset. (default: :obj:`None`)

        """

        if load_mode in ["o", "O", "overall", "Overall"]:
            self._load = self._load_overall
            # "overall"
        elif load_mode in ["I", "R", "i", "r", "Respective", "respective"]:
            # "Respectively"
            self._load = self._load_respective
        else:
            raise NotImplementedError("load_mode = 'o' or 'i'")

        super(InMemoryDatasetGeo, self).__init__(root, transform, pre_transform, pre_filter=pre_filter)

        if re_process_init:
            self.re_process()
        else:
            self.data, self.slices = torch.load(osp.join(self.processed_paths[0]))

    def _load_overall(self):
        assert len(self.raw_paths) == 1, "There is more than one .pt file,and not sure which one to import."
        return torch.load(self.raw_paths[0])

    def _load_respective(self):
        return [torch.load(i) for i in self.raw_paths]

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)  # must with 'raw' dir.

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        print("Convert ./raw files to ./processed. >>>")
        data_list = self._load()

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])

    def re_process(self):
        """Re-process for skip reset in 56 line in ``InMemoryDataset``
        `self.data, self.slices = None, None` ."""
        try:
            rmtree(self.processed_dir)
        except FileNotFoundError:
            pass
        os.makedirs(self.processed_dir)
        self.process()

        print('Done.')

    def to_mtbatch_data(self):
        from hpc2ml.data.batchdata import MtBatchData
        return MtBatchData.from_data(self.data, self.slices)


class DatasetGeo(Dataset):
    """For very very huge data out of memory.
    load data from local disk each epoth, and stored in local disk.

    It is not suggest if needed, due it reread data form local file.

    Examples
    -----------
    >>> # with data.edge_index shape (2,num_edge)
    >>> dataset = DatasetGeo(root=".")
    >>> # SparseTensor: with data.edge_index shape (num_node,num_node)
    >>> import torch_geometric.transforms as T
    >>> dataset2 = DatasetGeo(root=".", pre_transform=T.ToSparseTensor())

    """

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, re_process_init=True):
        """
        "Just accept mode 'i' for ``DatasetGEO``."

        Args:
            re_process_init (bool): process raw data or not. if there is no raw data but processed data is offered.
                this parameter could be False.
            root (string, optional): Root directory where the dataset should be
                saved. (optional: :obj:`None`)
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            pre_filter (callable, optional): A function that takes in an
                :obj:`torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset. (default: :obj:`None`)


        """
        super().__init__(root, transform, pre_transform, pre_filter)
        self.re_process_init = re_process_init

        super(DatasetGeo, self).__init__(root, transform, pre_transform, pre_filter=pre_filter)

        if re_process_init:
            self.re_process()
        else:
            pass

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @staticmethod
    def _files_exist(path):
        if os.path.isdir(path):
            files = os.listdir(path)
            return len(files) > 0
        else:
            return False

    @property
    def processed_file_names(self):
        if self._files_exist(self.processed_dir) and not self.re_process_init:
            return os.listdir(self.processed_dir)
        else:
            return ["data_{}.pt".format(i) for i in range(len(self.raw_file_names))]

    def re_process(self):
        """For temporary debug"""
        rmtree(self.processed_dir)  # remove the old
        os.makedirs(self.processed_dir)
        self.process()

        print('Done.')

    def process(self):
        print("Convert ./raw files to ./processed. >>>")
        i = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            try:
                data = torch.load(raw_path)
            except AttributeError as e:
                print(e)
                raise AttributeError("Just accept mode 'i' for ``DatasetGEO``, which load one at a time.",
                                     "The {} may be a batch of data. "
                                     "That is , if your raw data are save in one file overall, with 'o' mode."
                                     "please turn to ``InMemoryDatasetGeo``".format(raw_path))

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
            i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data
