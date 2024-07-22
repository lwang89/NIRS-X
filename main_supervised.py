from experiment_runner import ExpRunner
from copy import deepcopy

import pandas as pd
from utils import split_generic_test_data, split_generic_training_data, generate_buckets
from hyperparameter_optimizer import grid_search
from torch.utils.data import DataLoader


class SupervisedExpRunner(ExpRunner):
    def run(self):
        """

        :return:
        """
        # Split datasets and train models based on the paradigm we choose
        if self.paradigm == "subject-specific":
            # Save the top k models' test results and related information trained on each specific subject
            ss_results = pd.DataFrame(index=self.eligible_subject_ids, columns=["results"])
            ss_test_accs = []

            for subject_id in self.eligible_subject_ids:
                # Load individual subject data and split it into train, validation and test data
                subject_train_data, subject_val_data, subject_test_data = self.split_subject_specific_data(subject_id)

                # Optimize hyperparameter configuration on subject data
                best_config = grid_search(self.config_dict, subject_train_data, subject_val_data, self.paradigm)

                # Train a new model on subject data and get the training results and model metadata
                ss_model_info = self.exp_train(best_config, subject_train_data, subject_val_data, [subject_id])

                # Test the trained model and record the testing result
                ss_test_acc, ss_results = self.exp_test(ss_model_info, ss_results, [subject_test_data])
                ss_test_accs.append(ss_test_acc[0])

            ss_test_avg_acc = sum(ss_test_accs) / len(ss_test_accs)
            print(f"Average testing accuracy of subject-specific models is {ss_test_avg_acc}.")
            self.save_results(ss_results)

        else:
            # Save training and testing results and model metadata trained on each group under generic paradigm
            g_results = pd.DataFrame(index=self.eligible_subject_ids, columns=["results"])
            g_total_test_accs = []

            # Save training and testing results and model metadata pretrained on each group and fine-tuned on each subject under
            # generic subject-specific paradigm
            gss_results = pd.DataFrame(index=self.eligible_subject_ids, columns=["results"])
            gss_total_test_accs = []

            # Divide all subjects into different buckets of 4(by default) subjects each
            buckets = generate_buckets(self.eligible_subject_ids, self.generic_bucket_size)

            for bucket in buckets:
                print(f"Start to train under generic paradigm for target bucket {bucket}\n")

                # Use each bucket of subjects for testing and combine the remaining buckets for training and validation
                test_bucket = bucket
                train_buckets = deepcopy(buckets)
                train_buckets.remove(test_bucket)

                # Load the group training and validation data for each train_val/test split
                buckets_train_data, buckets_val_data = split_generic_training_data(train_buckets, self.dataset_path,
                                                                                   self.labels,
                                                                                   self.slide_window_option,
                                                                                   self.generic_buckets_train_val_split_ratio)

                # Get the list of test data for each subject in the test bucket
                bucket_train_data_list, bucket_val_data_list, bucket_test_data_list = split_generic_test_data(
                    test_bucket, self.dataset_path, self.labels, self.slide_window_option,
                    self.generic_subject_train_test_split_ratio, self.generic_subject_train_val_split_ratio)

                # Optimize hyperparameter configuration on group data
                best_config = grid_search(self.config_dict, buckets_train_data, buckets_val_data, self.paradigm)

                # Train a new model on group data and get the training results and model metadata
                g_model_info = self.exp_train(best_config, buckets_train_data, buckets_val_data, test_bucket)

                # Test the trained model and record the testing result
                g_test_accs, g_results = self.exp_test(g_model_info, g_results, bucket_test_data_list)
                g_total_test_accs += g_test_accs
                g_test_avg_acc = sum(g_test_accs) / len(g_test_accs)
                print("Test accuracy for subject {} is: {}\n".format(test_bucket, g_test_avg_acc))

                # For each target in the bucket, load the pretrained model and fine-tune it on the subject data
                print(f"Start to fine-tune each group model on target bucket {test_bucket}\n")
                for subject_id_idx in range(self.generic_bucket_size):
                    # Get the target subject id
                    target_subject_id = test_bucket[subject_id_idx]

                    # Load the training, validation and testing data of the target
                    target_train_data = bucket_train_data_list[subject_id_idx]
                    target_val_data = bucket_val_data_list[subject_id_idx]
                    target_test_data = bucket_test_data_list[subject_id_idx]

                    # Optimize hyperparameter configuration on subject data
                    best_config = grid_search(self.config_dict, target_train_data, target_val_data, self.paradigm)

                    # Load the pretrained group model and fine-tune it with the target train data
                    gss_model_info = self.exp_train(best_config, target_train_data, target_val_data,
                                                    [target_subject_id],
                                                    load_model=True,
                                                    model_filename=g_model_info.model_filename,
                                                    model_filepath=g_model_info.model_filepath)

                    # For each of the top k*v models trained on group and fine-tuned on the target, save the test
                    # results and other related model information
                    gss_test_accs, gss_results = self.exp_test(gss_model_info, gss_results, [target_test_data])
                    gss_total_test_accs += gss_test_accs

            g_total_test_avg_acc = sum(g_total_test_accs) / len(g_total_test_accs)
            print("\nAverage Test accuracy for generic paradigm is: {}".format(g_total_test_avg_acc))
            gss_total_test_avg_acc = sum(gss_total_test_accs) / len(gss_total_test_accs)
            print("Average Test accuracy for generic subject specific paradigm is: {}\n".format(gss_total_test_avg_acc))

            # Save the training and testing results and model metadata under the generic paradigm
            self.paradigm = "generic"
            self.save_results(g_results)

            # Update the paradigm name to be generic subject-specific in the saving filepath
            self.paradigm = "generic subject-specific"
            self.save_results(gss_results)


if __name__ == '__main__':
    ser = SupervisedExpRunner("config_nirsformer.yml")
    ser.run()
