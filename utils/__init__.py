import os
import random
import shutil
import torch
import yaml
import matplotlib.pyplot as plt
import pathlib
from argparse import Namespace
from copy import deepcopy
from datetime import datetime
from data.dataset import fNIRS2MW, naive_split


# Copied from https://github.com/sthalles/SimCLR/blob/master/utils.py
def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k

    Parameters
    ----------
    output : list
        A list of the predictions for each data point
    target : list
        A list of labels for each data point
    topk : tuple, default=(1,)
        A tuple of all the k(s) to compute the accuracy over

    Returns
    -------
    res : list
        A list of the accuracy over all the k(s)
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
            
        return res


def draw_graph(graph_filepath, graph_filename, train_loss, val_loss, val_acc):
    plt.figure(graph_filename)
    plt.subplot(2, 1, 1)
    plt.plot(train_loss, '-o')
    plt.plot(val_loss, '-o')
    plt.legend(['train', 'val'], loc='upper left')
    plt.xlabel('iteration')
    plt.ylabel('loss')

    plt.subplot(2, 1, 2)
    plt.plot(val_acc, '-o')
    plt.legend(['val'], loc='upper left')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')

    path = pathlib.Path().resolve()
    path = str(path) + "/results/graphs/" + graph_filepath
    
    # Create the directories if they don't exist
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(path + str(graph_filename) + ".jpg")
    plt.close()


def draw_contra_graph(graph_filepath, graph_filename, contra_train_loss):
    plt.figure(graph_filename)
    plt.plot(contra_train_loss, '-o')
    plt.xlabel('iteration')
    plt.ylabel('loss')

    path = pathlib.Path().resolve()
    path = str(path) + "/results/graphs/" + graph_filepath
    
    # Create the directories if they don't exist
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(path + str(graph_filename) + "contrastive.jpg")
    plt.close()


def print_path():
    import pathlib
    path = pathlib.Path().resolve()
    path = str(path) + "/graphs/"
    print(path)


def _grid_generator(cfgs_dict):
    keys = cfgs_dict.keys()
    result = {}

    if cfgs_dict == {}:
        yield {}

    else:
        # Create a copy to remove keys
        configs_copy = deepcopy(cfgs_dict)

        # Get the "first" key
        param = list(keys)[0]
        del configs_copy[param]

        first_key_values = cfgs_dict[param]
        for value in first_key_values:
            result[param] = value

            for nested_config in _grid_generator(configs_copy):
                result.update(nested_config)
                yield deepcopy(result)


def create_grid(configs_dict):
    config_list = [Namespace(**config) for config in _grid_generator(configs_dict)]
    
    return config_list


def generate_filepath(model_name, labels, paradigm, dataset):
    # Define experiment settings' file names
    class_dict = {
        2: "bi",
        4: "multi"
    }
    paradigm_dict = {
        "subject-specific": "subject_specific",
        "generic": "generic",
        "generic subject-specific": "generic_subject_specific",
        "contrastive": "contrastive"
    }
    dataset_dict = {
        "audio": "audio",
        "visual": "visual",
        "mix": "mix"
    }

    # Concatenate class labels into a string
    labels_name = "".join(str(label) for label in labels)

    # Concatenate datetime and experiment settings into a string
    logging_filepath = (f"/{model_name}/{class_dict[len(labels)]}_{labels_name}/"
                        f"{dataset_dict[dataset]}/{paradigm_dict[paradigm]}/" +
                        datetime.now().strftime("%y%m%d%H%M") + "/")

    return logging_filepath


def generate_buckets(eligible_subjects, bucket_size):
    """

    :param eligible_subjects:
    :param bucket_size:
    :return:
    """
    buckets = []
    num_buckets = int(len(eligible_subjects) / bucket_size)

    # Divide all subjects into buckets
    subjects = deepcopy(eligible_subjects)  
    random.shuffle(subjects)
    for i in range(num_buckets):
        bucket = subjects[i * bucket_size:(i + 1) * bucket_size]
        buckets.append(bucket)
        
    return buckets


def split_generic_training_data(train_buckets, dataset_path, labels, slide_window_option, train_val_ratio):
    """
    load all data from the train_buckets;
    split them into two groups (train/val) by train_val_ratio
    :param train_buckets:
    :param dataset_path:
    :param labels:
    :param slide_window_option:
    :param train_val_ratio:
    :return:
    """
    # Load train subjects list and then shuffle
    whole_train_subjects_list = [subject for train_bucket in train_buckets for subject in train_bucket]
    random.shuffle(whole_train_subjects_list)
    num_subjects = len(whole_train_subjects_list)

    # Create validation and train subjects list from the whole train subject list
    train_subjects_list = whole_train_subjects_list[:int(train_val_ratio * num_subjects)]
    val_subjects_list = whole_train_subjects_list[int(train_val_ratio * num_subjects):num_subjects]

    # Load validation and train data
    buckets_train_data = fNIRS2MW(slide_window_option=slide_window_option, config_path=dataset_path,
                                  subject_list=train_subjects_list, labels=labels)
    buckets_val_data = fNIRS2MW(slide_window_option=slide_window_option, config_path=dataset_path,
                                subject_list=val_subjects_list, labels=labels)

    return buckets_train_data, buckets_val_data


def split_generic_test_data(test_bucket, dataset_path, labels, slide_window_option, train_test_ratio, train_val_ratio):
    """

    :param test_bucket:
    :param dataset_path:
    :param labels:
    :param slide_window_option:
    :param train_test_ratio:
    :param train_val_ratio:
    :return:
    """
    bucket_train_data_list = []
    bucket_val_data_list = []
    bucket_test_data_list = []

    # For each subject in the test bucket, load the subject's data
    for subject_id in test_bucket:
        subject_whole_data = fNIRS2MW(slide_window_option=slide_window_option, config_path=dataset_path,
                                      subject_list=[subject_id], labels=labels)
        
        # Split the whole data into 3 pieces: train/val/test
        whole_subject_train_data, subject_test_data = naive_split(subject_whole_data, train_test_ratio)
        subject_train_data, subject_val_data = naive_split(whole_subject_train_data, train_val_ratio)

        # Append them into assigned list
        bucket_train_data_list.append(subject_train_data)
        bucket_val_data_list.append(subject_val_data)
        bucket_test_data_list.append(subject_test_data)

    return bucket_train_data_list, bucket_val_data_list, bucket_test_data_list
