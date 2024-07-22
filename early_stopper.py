class EarlyStopper:
    """Copied Keras.EarlyStopping with modification."""

    def __init__(self, patience=10, min_delta=1e-4, restore_best_weights=True, start_from_epoch=0):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.start_from_epoch = start_from_epoch
        self.min_val_loss = float("inf")
        self.trigger_counter = 0
        self.stopped_epoch = 0
        self.best_epoch = 0
        self.best_model_state_dict = None

    def early_stop(self, model_state_dict, epoch, epoch_val_loss):

        # If still in initial warm-up stage
        if epoch < self.start_from_epoch:
            return False

        # Set the initial network weights as the best
        if self.restore_best_weights and self.best_model_state_dict is None:
            self.best_model_state_dict = model_state_dict
            self.best_epoch = epoch

        self.trigger_counter += 1
        
        # Remember the model with the least validation loss
        if epoch_val_loss < self.min_val_loss:
            self.min_val_loss = epoch_val_loss
            self.best_epoch = epoch
            self.best_model_state_dict = model_state_dict

        # Restart counter if the validation performance improves by a threshold
        if (epoch_val_loss - self.min_delta) < self.min_val_loss:
            self.trigger_counter = 0
            return False

        # Keep track of how many times the early stopper is continuously triggered
        if self.trigger_counter >= self.patience:
            self.stopped_epoch = epoch
            return True

        return False
