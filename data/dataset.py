import glob
import os
import re
import yaml

import numpy as np
# import data.dataset
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from data_aug import augment


class fNIRS2MW(Dataset):
    """
    Class description: fNIRS2MW dataset loader
    """

    def __init__(self,
                 dataset_name="fNIRS2MW",
                 experiment_type="visual",
                 # non bpf, window size 150ts, stride 3ts
                 slide_window_option=10,
                 config_path="config_dataset.yml",
                 subject_list=None,
                 mode='xy',
                 labels=None):
        assert labels is not None, "label can not be empty"
        self.labels = labels
        self.label_keys = self.labels.keys()
        self.label_list = []
        self.df = None  # read the csv file into pandas dataframe
        assert mode in ['x', 'xy', 'contrastive']
        self.mode = mode
        self.chunks_tensor = []
        self._1st_aug_chunks_tensor = []
        self._2nd_aug_chunks_tensor = []
        assert subject_list is not None, "subject list should not be None."
        self.subject_list = subject_list
        self.dir = None
        self.dataset_config = []
        self.window_size = 150
        self.window_stride = 3

        self.__load_config_dataset__(config_path)
        self.experiment_type = self.dataset_config['fNIRS2MW']['experiment_type'][0]
        self.__getdir__(dataset_name, self.experiment_type, slide_window_option)
        self.__get_window_size_stride__(dataset_name, slide_window_option)
        self.__initialize_dataset__(subject_list)

    def __getitem__(self, index):
        """
        Get the instance or tuple from the dataset at given index
        :param index:
        :return:  item : instance or tuple
        """
        if self.mode == 'x':
            return torch.tensor(self.chunks_tensor[index])
        elif self.mode == 'xy':
            return torch.tensor(self.chunks_tensor[index]), torch.tensor(self.label_list[index])
        elif self.mode == 'contrastive':
            return torch.tensor(augment(self.chunks_tensor[index]).copy()), torch.tensor(
                augment(self.chunks_tensor[index]).copy())

    def __load_config_dataset__(self, config_path):
        """
        __load_config_dataset__ description
        Parameters
        ----------
        config_path : str
            Name of .yml file for configuring the dataset
        """
        with open(os.path.dirname(os.path.realpath(__file__)) + "/" + config_path, "r") as stream:
            try:
                self.dataset_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    def __getdir__(self, dataset_name, experiment_type, slide_window_option):
        """
        :param dataset_name:
        :param experiment_type:
        :param slide_window_option:
        :return:
        """
        global_path = self.dataset_config[dataset_name]['global_path']
        slide_window_path = self.dataset_config[dataset_name]['slide_window_path']
        slide_window_options = self.dataset_config[dataset_name]['slide_window_options'][slide_window_option]

        slide_window_data_path = global_path + experiment_type + slide_window_path
        self.dir = slide_window_data_path + slide_window_options + "/*.csv"

    def __get_window_size_stride__(self, dataset_name, slide_window_option):
        """
        __get_window_size_stride__ description
            Update two parameter:
            "self.window_size"
            "self.window_stride"
        :param slide_window_option:
        :return: None
        """
        slide_window_options = self.dataset_config[dataset_name]['slide_window_options'][slide_window_option]
        if slide_window_options == "bpf_whole_data":
            self.window_size = 0
            self.window_stride = 0
        elif slide_window_options == "non_overlapping_task_142ts":
            self.window_size = 142
        elif slide_window_options.startswith("bpf"):
            result = re.search('bpf_size_(.*)ts_stride_3ts', slide_window_options)
            self.window_size = int(result.group(1))
        else:
            result = re.search('size_(.*)ts_stride_3ts', slide_window_options)
            self.window_size = int(result.group(1))

    def __initialize_dataset__(self, subject_list):
        """
            __initialize_dataset__ description
            initialize the dataset for three modes and:
            "x":
            "xy":
            "contrastive": return two augmented data

        :param subject_list: list
        :return:
        """
        data_list = []
        _1st_aug_data_list = []
        _2nd_aug_data_list = []
        label_list = []

        for filename in glob.glob(self.dir):
            # read the subject id
            subject_id = os.path.basename(filename).strip("sub_.csv")
            # read the subject id and check if it is in the subject_list
            if int(subject_id) not in subject_list:
                continue
            else:
                # read the csv
                df = pd.read_csv(filename)
                # group and aggregate by chunk id here
                grouped = df.groupby(['chunk', 'label'])
                grouped_list = []
                _1st_aug_grouped_list = []
                _2nd_aug_grouped_list = []
                labels = []

                for tuple, group in grouped:
                    data = group.drop(columns=['chunk', 'label']).to_numpy(dtype=np.float32)
                    grouped_list.append(data)
                    labels.append(tuple[1])
                data_list.extend(grouped_list)
                label_list.extend(labels)

        # select data with assigned labels
        if self.mode == "xy":
            data_list = [data for data, label in zip(data_list, label_list) if label in self.label_keys]
            label_list = [label for label in label_list if label in self.label_keys]

        self.chunks_tensor = np.array(data_list)
        self.label_list = np.array(label_list)

        for k, v in self.labels.items():
            self.label_list[self.label_list == k] = v

    def __len__(self):
        return len(self.chunks_tensor)


def naive_split(dataset, split_ratio):
    """
    Naive split description
    Parameters
    ----------
    dataset : Dataset
        The dataset
    split_ratio : float
        The percent of data
    Returns
    -------
    train_set, test_set : tuple
        A tuple of the train set and test set
    """
    data_length = len(dataset)

    train_length = int(split_ratio * data_length)
    test_length = data_length - train_length

    train_set, test_set = regular_split(dataset, [train_length, test_length])

    return train_set, test_set


# Taken from python 3.5 docs
def _accumulate(iterable, fn=lambda x, y: x + y):
    'Return running totals'
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total


def regular_split(dataset, lengths):
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = range(sum(lengths))
    return [Subset(dataset, indices[offset - length: offset]) for offset, length in zip(_accumulate(lengths), lengths)]
