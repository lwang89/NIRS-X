"""The entrance of training, """
import os
from argparse import Namespace
from copy import deepcopy
import pandas as pd
import torch
import yaml
from data.dataset import fNIRS2MW, naive_split
from utils import draw_graph, generate_filepath
import pathlib
from base_trainer import BaseTrainer
from torch.utils.data import DataLoader


class BaseExpRunner:

    def __init__(self, config_path, slide_window_option=4):
        self.config_dict = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
        self.dataset_path = self.config_dict["dataset_path"][0]
        self.data_path = pathlib.Path().absolute() / "data" / self.dataset_path
        self.data_mode = self.config_dict["data_mode"][0]
        self.eligible_subject_ids = self.initialize_eligible_subjects_list()
        self.labels = self.initialize_labels()
        self.mode = self.config_dict["mode"][0]
        self.paradigm = self.config_dict["paradigm"][0]
        self.model_name = self.config_dict["model"][0]

        # Specify slide window option
        self.slide_window_option = slide_window_option

        # Specify data split ratios for subject specific paradigm
        self.subject_specific_train_test_split_ratio = 0.5
        self.subject_specific_train_val_split_ratio = 0.5

        # Specify data split ratios for generic and generic subject specific paradigm
        self.generic_bucket_size = 4
        self.generic_buckets_train_val_split_ratio = 0.75
        self.generic_subject_train_test_split_ratio = 0.5
        self.generic_subject_train_val_split_ratio = 0.5

    def save_results(self, results):
        # Convert namespaces to dictionaries
        results = results.applymap(lambda model_info: model_info.__dict__)

        # 1. Take the column of model_info dictionaries out and convert it into a list
        # 2. Build a new dataframe with subject ids as indices and model_info keys as columns
        results_list = pd.DataFrame.from_records(results.iloc[:, 0].to_list(), index=results.index)

        # Generate the filepath to save the training graphs under
        # results/graphs/model_name/classification_type/paradigm_name
        graph_filepath = generate_filepath(self.model_name, self.labels, self.paradigm, self.data_mode)
        path = pathlib.Path().resolve()
        path = str(path) + "/results/data/" + graph_filepath

        # Create the directories if they don't exist
        if not os.path.exists(path):
            os.makedirs(path)

        # Save training and testing results and model metadata to the generated filepath
        results_list.to_csv(path + ".csv")

        # Draw a training graph for each subject
        results_list.apply(lambda row: draw_graph(graph_filepath, "sub" + str(row.name),
                                                  row['train_loss'], row["val_loss"], row["val_accu"]), axis=1)

    def exp_train(self, config, train_data, val_data, subject_id_list, load_model=False, model_filename=None,
                  model_filepath=None):
        """
        description:
        1. load and train with all hyperparameter configuration settings;
        2. save top k models' information according to validation accuracy
        :param config:
        :param model_filepath:
        :param model_filename:
        :param load_model:
        :param val_data:
        :param train_data:
        :param k:
        :param subject_id_list:
        :return: the list of top k models' information ([validation_accuracy, model_filepath, config, subject_id])
        """
        # Train a model under the specified hyperparameter configuration
        print("Training under config {}".format(config))

        if load_model:
            # For generic subject-specific paradigm
            trainer = BaseTrainer(config, train_data, val_data, subject_id_list, paradigm="generic subject-specific")
            trainer.network.load_state_dict(torch.load(model_filepath + model_filename))

        else:
            # For subject specific and generic paradigms
            trainer = BaseTrainer(config, train_data, val_data, subject_id_list)

        # Fit the model to data, get the filepath and filename to the trained model and its training results
        model_filename, model_filepath, train_loss, val_loss, val_acc = trainer.train()

        # Record the trained model's metadata and training results
        model_info = Namespace(model_filename=model_filename,
                               model_filepath=model_filepath,
                               config=config,
                               subject_id_list=subject_id_list,
                               train_loss=train_loss,
                               val_loss=val_loss,
                               val_accu=val_acc)

        return model_info

    def exp_test(self, model_info, results, test_data_list):
        """
        description:
        1. load the top k models and test data
        2. save results along with model information ([test_accuracy, validation_accuracy, model_filepath, config])
        :param top_k_results:
        :param top_k:
        :param k:
        :param test_data_list:
        :return: None
        """
        # Load the trained model
        model = BaseTrainer(config=model_info.config)
        model.network.load_state_dict(torch.load(model_info.model_filepath + model_info.model_filename))

        test_accs = []
        # Run testing and save testing accuracy for each subject in the subject id list
        for subject_id_idx in range(len(model_info.subject_id_list)):
            # Get test data for the specified subject
            subject_id = model_info.subject_id_list[subject_id_idx]
            subject_test_data = test_data_list[subject_id_idx]

            # Get testing accuracy and model metadata
            test_acc = model.test(subject_test_data)
            test_accs.append(test_acc)
            print("Test accuracy for subject {} is: {}".format(subject_id, test_acc))

            # Save subject testing results and model metadata
            subject_results = deepcopy(model_info)
            del subject_results.subject_id_list
            subject_results.test_acc = test_acc
            results.loc[subject_id] = subject_results

        return test_accs, results

    def initialize_eligible_subjects_list(self):
        eligible_subjects_list = []
        with open(self.data_path, "r") as _file:
            try:
                _data = yaml.safe_load(_file)
                eligible_subjects_list = _data['fNIRS2MW']['eligible_subject_list'][self.data_mode]
            except yaml.YAMLError as _exc:
                print(_exc)
        return eligible_subjects_list

    def initialize_labels(self):
        """
        # transform the labels into ordinals like: [0, 2] ----> [0, 1]
        #                                          [0, 1, 3] ---> [0, 1, 2]
        #                                          [0, 1 ,2 ,3] ---> [0, 1, 2, 3]
        :param
        :return:
        """
        # extract all labels from the configs_dict
        labels = self.config_dict["labels"][0]
        labels = {k: v for v, k in enumerate(labels)}

        return labels

    def split_subject_specific_data(self, subject_id, mode="xy"):
        subject_whole_data = fNIRS2MW(slide_window_option=self.slide_window_option, config_path=self.dataset_path,
                                      subject_list=[subject_id], labels=self.labels, mode=mode)

        subject_whole_train_data, subject_test_data = naive_split(subject_whole_data,
                                                                  self.subject_specific_train_test_split_ratio)
        subject_train_data, subject_val_data = naive_split(subject_whole_train_data,
                                                           self.subject_specific_train_val_split_ratio)

        return subject_train_data, subject_val_data, subject_test_data
