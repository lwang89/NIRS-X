import logging
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import trange
from data.dataset import fNIRS2MW, naive_split
from early_stopper import EarlyStopper
from models import get_model
from utils import save_config_file, accuracy, draw_graph
from models.base_models import linear_classifier
from contra_loss.contra_loss import SimSiam
from trainer import Trainer
import warnings
warnings.simplefilter("ignore")


class contra_Trainer(Trainer):
    def __init__(self, config=None, pretext_set=None, train_set=None, val_set=None, test_set=None, labels=None, subject_id_list=None):
        super().__init__(config=config, train_set=train_set, val_set=val_set, paradigm="contrastive", subject_id_list=subject_id_list)

        # Initialize two encoders
        self.labels = labels

        # Transfer criterion to GPU, then use cudnn
        if torch.cuda.is_available():
            self.cl_criterion = SimSiam(device=self.device, T=self.config.T).to(self.device)

        # Prepare pretext data
        self.pretext_loader = DataLoader(pretext_set, batch_size=self.config.batch_size, shuffle=True, num_workers=4)
        self.test_loader = DataLoader(test_set, batch_size=self.config.batch_size, shuffle=False)

    def pretext(self):
        self.network.train()

        # Record the loss
        train_loss = []

        # Fit model to the pretext data
        for epoch in range(self.config.n_epoch):
            total_loss = 0

            for data in self.pretext_loader:
                # Transfer training data to GPU
                aug1_tensor, aug2_tensor = data
                aug1_tensor, aug2_tensor = aug1_tensor.to(self.device, dtype=torch.float), aug2_tensor.to(self.device, dtype=torch.float)
                emb_aug1, pred1 = self.network(aug1_tensor)
                emb_aug2, pred2 = self.network(aug2_tensor)

                # Computer contrastive loss
                loss = self.cl_criterion(emb_aug1, emb_aug2, pred1, pred2)

                # Clean gradient
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Add batch loss to total loss
                total_loss += loss.item()

            # Compute average training loss of the current epoch
            epoch_train_loss = total_loss / len(self.pretext_loader)
            print(f'Epoch {epoch + 1}: \tTrain Loss: {epoch_train_loss:.8f}')
            train_loss.append(epoch_train_loss)

        return train_loss

    def linear_probe(self):
        # 1. Load the pretrained model parameters the subjects'/subject's train data
        # 2. Use target subject's train data to train the linear classifier then validate on target's val data
        # 3. Obtain testing results on target's test data
        self.network.eval()

        # Create a new linear classification model
        network = linear_classifier(self.network, num_classes=len(self.labels))
        if torch.cuda.is_available():
            network.to(self.device)

        # Freeze encoder, update linear layer parameters
        network.encoder.requires_grad = False
        parameters = [param for param in network.parameters() if param.requires_grad is True]
        optimizer = torch.optim.Adam(parameters, self.config.linear_lr)

        # Initialize the early stopper
        early_stopper = EarlyStopper()

        # Record the loss
        train_loss = []
        val_loss = []
        val_acc = []

        # Get and record the training loss, validation loss and accuracy of the current epoch
        for epoch in range(0, self.config.linear_epoch):
            epoch_train_loss, epoch_val_loss, epoch_val_acc = self.train_epoch(network, optimizer, epoch)
            train_loss.append(epoch_train_loss)
            val_loss.append(epoch_val_loss)
            val_acc.append(epoch_val_acc)

            # Check if early stops
            if early_stopper.early_stop(network.state_dict(), epoch, epoch_val_loss):
                break

        best_model_state_dict = early_stopper.best_model_state_dict if early_stopper.best_model_state_dict is not None else self.network.state_dict()
        network.load_state_dict(best_model_state_dict)
        test_loss, test_acc = self.evaluate(network, self.test_loader)

        return train_loss, val_loss, val_acc, test_loss, test_acc
