import numpy
import torch
from experiment_runner import ExpRunner
from copy import deepcopy

import logging
import os
import random
import time
from argparse import Namespace
from copy import deepcopy
from datetime import datetime

import pandas as pd
from data.dataset import fNIRS2MW
from hyperparameter_optimizer import contra_grid_search
from contrastive_trainer import contra_Trainer


class ContraExpRunner(ExpRunner):

    def run(self):
        # Save training and testing results and model metadata trained under contrastive learning paradigm
        contra_results = pd.DataFrame(index=self.eligible_subject_ids, columns=["results"])
        contra_hpo_test_accs = []
        contra_test_accs = []

        # For each target subject, train on all other subjects' unlabeled data, validate on the subject's val data,
        # test on the subject's test data
        for subject_id in self.eligible_subject_ids:
            
            # Get the list of subject ids used to prepare pretext data
            pretext_subject_id_list = deepcopy(self.eligible_subject_ids)
            pretext_subject_id_list.remove(subject_id)

            # Load group unlabeled training data with contrastive mode
            pretext_data = fNIRS2MW(slide_window_option=self.slide_window_option, config_path=self.dataset_path,
                                    subject_list=pretext_subject_id_list, mode="contrastive", labels=self.labels)

            # Load labeled target subject's train, val, test data
            subject_train_data, subject_val_data, subject_test_data = self.split_subject_specific_data(subject_id)

            # Optimize hyperparameter configuration on contrastive task and unlabeled group data
            best_config, best_config_test_acc = contra_grid_search(self.config_dict, pretext_data, subject_train_data, subject_val_data, subject_test_data, self.labels)
            contra_hpo_test_accs.append(best_config_test_acc)

            # Train a new contrastive encoder on group unlabeled data and get the training results and model metadata
            contra_model_info = self.exp_pretext_and_downstream(best_config, pretext_data, subject_train_data,
                                                                subject_val_data, subject_test_data, subject_id, self.labels)

            subject_contra_results = deepcopy(contra_model_info)
            del subject_contra_results.subject_id_list
            contra_results.loc[subject_id] = subject_contra_results
            contra_test_accs.append(subject_contra_results.downstream_test_acc)
            print("Test accuracy for subject {} is: {}".format(subject_id, subject_contra_results.downstream_test_acc))

        contra_test_avg_acc = sum(contra_test_accs) / len(contra_test_accs)
        print("Average Test accuracy for contrastive paradigm is: {}\n".format(contra_test_avg_acc))

        self.save_results(contra_results)

    def exp_pretext_and_downstream(self, config, pretext_data, train_data, val_data, test_data, subject_id, labels):
        print("Pretext under config {}".format(config))
        contra_trainer = contra_Trainer(config, pretext_set=pretext_data, train_set=train_data, val_set=val_data,
                                        test_set=test_data, labels=labels, subject_id_list=[subject_id])

        # Pretrain the encoder with the contrastive pretext task and get the training loss and accuracy records
        contra_train_loss = contra_trainer.pretext()

        # Freeze the encoder and train a linear classifier on top with the downstream classification task
        downstream_train_loss, downstream_val_loss, downstream_val_acc, downstream_test_loss, downstream_test_acc = contra_trainer.linear_probe()

        # Record the trained model's metadata and training results
        model_info = Namespace(config=config,
                               subject_id_list=[subject_id],
                               contra_train_loss=contra_train_loss,
                               train_loss=downstream_train_loss,
                               val_loss=downstream_val_loss,
                               val_accu=downstream_val_acc,
                               downstream_test_loss=downstream_test_loss,
                               downstream_test_acc=downstream_test_acc, )

        return model_info


if __name__ == '__main__':
    ser = ContraExpRunner("config_nirsformer.yml")
    ser.run()
