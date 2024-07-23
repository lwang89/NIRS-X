import logging
import os

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import get_model
from utils.early_stopper import EarlyStopper
from utils import save_config_file, accuracy, generate_filepath


class BaseTrainer:
    """The base class for training models with data."""

    def __init__(self, config, train_set=None, val_set=None, subject_id_list=None, paradigm=None, save_model=True):
        # Prepare model
        self.config = config
        self.network = get_model(self.config)
        self.criterion = nn.CrossEntropyLoss()

        # Choose an optimizer and learning rate scheduler
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.lr, weight_decay=0.01)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)

        # Prepare training and validation data
        self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size,
                                       shuffle=True) if train_set is not None else None
        self.val_loader = DataLoader(val_set, batch_size=self.config.batch_size,
                                     shuffle=False) if val_set is not None else None
        self.subject_id_list = subject_id_list

        # Transfer model to GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.network.to(self.device)
            self.criterion = self.criterion.to(self.device)
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        # Keep the string name of the current hyperparameter configuration
        self.config_name = "lr" + str(self.config.lr) + "_" + \
                           "dropout" + str(self.config.dropout) + "_" + \
                           "patience" + str(self.config.patience)

        # Initialize the training log writer
        self.writer = self.initialize_writer(paradigm)
        self.save_model = save_model

    def train(self):
        # Initialize the early stopper
        early_stopper = EarlyStopper(patience=self.config.patience)

        # Record the loss
        train_loss = []
        val_loss = []
        val_acc = []

        # Reset writer for training a new model
        logging.info(f"\nStart training for {self.config.n_epoch} epochs.")

        # Fit model to the training data
        for epoch in range(self.config.n_epoch):
            # Get and record the training loss, validation loss and accuracy of the current epoch
            epoch_train_loss, epoch_val_loss, epoch_val_acc = self.train_epoch(self.network, self.optimizer, epoch)
            train_loss.append(epoch_train_loss)
            val_loss.append(epoch_val_loss)
            val_acc.append(epoch_val_acc)

            # Update the tdqm progress bar with the training loss and validation loss of the current epoch
            print(f'Epoch {epoch + 1}: \tTrain Loss: {epoch_train_loss:.8f}\t\tVal Loss: {epoch_val_loss:.8f}')

            # Check if early stops
            if early_stopper.early_stop(self.network.state_dict(), epoch, epoch_val_loss):
                break

        # Mark the end of training the model
        logging.info("Training has finished.")

        if self.save_model:
            # Save best model;
            model_filename = self.config_name + '.pth'
            model_filepath = self.writer.log_dir
            best_model_state_dict = early_stopper.best_model_state_dict if early_stopper.best_model_state_dict is not None else self.network.state_dict()
            torch.save(best_model_state_dict, model_filepath + model_filename)

            # Save best model hyperparameter configuration to file
            save_config_file(self.writer.log_dir, self.config)

            # Log model metadata saving filepath and reach the end of log
            logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
            self.writer.close()

            return model_filename, model_filepath, train_loss, val_loss, val_acc

        return train_loss, val_loss, val_acc

    def train_epoch(self, network, optimizer, epoch):
        network.train()
        total_loss = 0
        total_accuracy = 0

        for data in self.train_loader:
            # Transfer training data to GPU
            chunks_tensor, labels_tensor = data
            chunks_tensor = chunks_tensor.to(self.device)
            labels_tensor = labels_tensor.to(self.device)

            # Generate results from the model, then compute loss
            preds = network(chunks_tensor)
            loss = self.criterion(preds, labels_tensor)
            acc = accuracy(preds, labels_tensor)[0].cpu()

            # Clean gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Add batch loss to total loss
            total_loss += loss.item()
            total_accuracy += acc

        # Compute average loss and accuracy
        epoch_train_loss = total_loss / len(self.train_loader)
        epoch_train_acc = total_accuracy / len(self.train_loader)

        # Validate the model of the current epoch
        epoch_val_loss, epoch_val_acc = self.evaluate(network, self.val_loader)

        # Log training and validation results of the current epoch
        logging.debug(f"\nEpoch: {epoch + 1}\tTraining Loss: {epoch_train_loss}\tTraining Accuracy: {epoch_train_acc}")
        logging.debug(f"\nEpoch: {epoch + 1}\tValidation Loss: {epoch_val_loss}\tValidation Accuracy: {epoch_val_acc}")

        return epoch_train_loss, epoch_val_loss, epoch_val_acc

    def evaluate(self, network, loader):
        """
        Evaluate the accuracy with data for selected model
        :param loader: validation or test data
        :return: accuracy
        """
        network.eval()

        # Variables for computing accuracy
        correct = 0
        total = 0
        total_loss = 0

        for data in loader:
            chunks_tensor, labels_tensor = data
            chunks_tensor = chunks_tensor.to(self.device)
            labels_tensor = labels_tensor.to(self.device)

            # Predict the votes
            preds = network(chunks_tensor)

            # Accumulate loss
            total_loss += self.criterion(preds, labels_tensor).item()

            # Change votes to labels, to calculate the accuracy
            preds = preds.argmax(-1)
            correct += (preds == labels_tensor).sum()
            total += preds.shape[0]

        acc = (correct / total).item() * 100

        return total_loss / len(loader), acc

    def test(self, test_data):
        # Load test data
        test_loader = DataLoader(test_data, batch_size=self.config.batch_size, shuffle=False)
        _, test_accuracy = self.evaluate(self.network, test_loader)

        return test_accuracy

    def initialize_writer(self, paradigm):
        # If the trainer is initialized for hpo or testing, no need to initialize training log writer
        if self.subject_id_list is None:
            return

        # Generate the filepath to save logs
        if paradigm is None:
            logging_filepath = generate_filepath(self.config.model, self.config.labels, self.config.paradigm,
                                                 self.config.data_mode)
        else:
            logging_filepath = generate_filepath(self.config.model, self.config.labels, paradigm, self.config.data_mode)

        # Get the string name of all the subjects in subject_id_list
        subject_name = ""
        for subject_id in self.subject_id_list:
            subject_name += f"_{subject_id}"

        # Write training log to ./runs/ directory by default
        writer = SummaryWriter(
            log_dir=os.path.dirname(os.path.realpath(__file__)) + "/runs/" + logging_filepath + subject_name[1:] + "/")
        logging.basicConfig(filename=os.path.join(writer.log_dir, 'training.log'), level=logging.DEBUG)

        return writer
