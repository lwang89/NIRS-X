'''Modify from d2l.torch'''
import time
import inspect
from scipy import stats
from base_trainer import BaseTrainer
from base_nirsiam_trainer import BaseNIRSiamTrainer
from argparse import Namespace
from copy import deepcopy
from utils import create_grid


class HyperParameters:
    """The base class of hyperparameters"""

    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes"""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k: v for k, v in local_vars.items()
                        if k not in set(ignore + ['self']) and not k.startswith('_')}

        for k, v in self.hparams.items():
            setattr(self, k, v)


class HPOSearcher(HyperParameters):
    """The random searcher for sampling configurations for new trials"""

    def __init__(self, config_space: dict, initial_config=None):
        self.save_hyperparameters()

    def sample_configuration(self) -> dict:
        """Sample a new candidate configuration uniformly at random"""
        if self.initial_config is not None:
            result = self.initial_config
            self.initial_config = None
        else:
            result = {
                name: domain.rvs()
                for name, domain in self.config_space.items()
            }
        return result


class GridSearcher(HyperParameters):
    """The random searcher for sampling configurations for new trials"""

    def __init__(self, config_list, initial_config=None):
        self.save_hyperparameters()

    def sample_configuration(self) -> dict:
        """Sample a new candidate configuration uniformly at random"""
        if self.initial_config is not None:
            result = self.initial_config
            self.initial_config = None
        else:
            result = next(self.config_list)
        return result


class HPOScheduler(HyperParameters):
    """The basic scheduler that schedules a new configuration every time new resources become available"""

    def __init__(self, searcher):
        self.save_hyperparameters()

    def suggest(self) -> dict:
        return self.searcher.sample_configuration()


class HPOTuner(HyperParameters):
    """The component that runs the scheduler/searcher and does some book-keeping of the results"""

    def __init__(self, scheduler: HPOScheduler, objective: callable):
        self.save_hyperparameters()

        # Bookkeeping results for plotting
        self.best_config = None
        self.best_config_error = float("inf")
        self.best_config_trajectory = []
        self.cumulative_runtime = []
        self.current_runtime = 0
        self.records = []
        self.num_train_epoch = None

    def run(self, config_dict, train_data, val_data, paradigm, num_epochs=20, num_trials=3):
        # Convert value lists to numbers
        config_dict = {key: value[0] for key, value in config_dict.items()}
        self.num_train_epoch = config_dict["n_epoch"]
        config_dict["n_epoch"] = num_epochs

        for i in range(num_trials):
            '''Execute HPO trials sequentially'''
            start_time = time.time()
            config_sample = self.scheduler.suggest()
            print(f"\nTrial {i + 1}: config = {config_sample}")

            # Reset configuration dictionary for training
            config = deepcopy(config_dict)
            for key, value in config_sample.items():
                config[key] = value

            # Convert dict to namespace
            config = Namespace(**config)

            error = self.objective(config, train_data, val_data, paradigm)
            error = float(error)
            runtime = time.time() - start_time
            self.bookkeeping(config, error, runtime)
            print(f"error = {error}, runtime = {runtime}")

    def bookkeeping(self, config: dict, error: float, runtime: float):
        self.records.append({"config": config, "error": error, "runtime": runtime})

        # Check if the last hyperparameter configuration performs better than the current best one
        if self.best_config is None or self.best_config_error > error:
            self.best_config = config
            self.best_config_error = error
            config.n_epoch = self.num_train_epoch

        # Add current best observed performance to the optimization trajectory
        self.best_config_trajectory.append(self.best_config_error)

        # Update runtime
        self.current_runtime += runtime
        self.cumulative_runtime.append(self.current_runtime)


class GridSearchTuner(HyperParameters):
    """The component that runs the scheduler/searcher and does some book-keeping of the results"""

    def __init__(self, scheduler: HPOScheduler, objective: callable):
        self.save_hyperparameters()

        # Bookeeping results for plotting
        self.best_config = None
        self.best_config_error = float("inf")
        self.best_config_trajectory = []
        self.cumulative_runtime = []
        self.current_runtime = 0
        self.records = []
        self.num_train_epoch = None
        self.num_linear_train_epoch = None
        self.min_val_accs = []
        self.test_accs = []

    def run(self, train_data, val_data, pretext_data=None, test_data=None, paradigm=None, labels=None, num_epochs=20,
            num_trials=10, num_linear_epochs=None):

        for i in range(num_trials):
            '''Execute HPO trials sequentially'''
            start_time = time.time()
            config = self.scheduler.suggest()

            # Update training epoch number in config
            if self.num_train_epoch is None:
                self.num_train_epoch = config.n_epoch
            if num_linear_epochs is not None and self.num_linear_train_epoch is None:
                self.num_linear_train_epoch = num_linear_epochs
            config.n_epoch = num_epochs
            config.linear_epoch = num_linear_epochs
            print(f"\nTrial {i + 1}: config = {config}")

            # Check if is supervised or contrastive learning
            if pretext_data is None:
                error = self.objective(config, train_data, val_data, paradigm)
            else:
                error, min_val_acc, test_acc = self.objective(config, pretext_data, train_data, val_data, test_data, labels)
                self.min_val_accs.append(min_val_acc)
                self.test_accs.append(test_acc)
            error = float(error)
            runtime = time.time() - start_time
            self.bookkeeping(config, error, runtime)
            print(f"error = {error}, runtime = {runtime}")

    def bookkeeping(self, config: dict, error: float, runtime: float):
        self.records.append({"config": config, "error": error, "runtime": runtime})

        # Check if the last hyperparameter configuration performs better than the current best one
        if self.best_config is None or self.best_config_error > error:
            self.best_config = config
            self.best_config_error = error
            self.best_config.n_epoch = self.num_train_epoch
            self.best_config.linear_epoch = self.num_linear_train_epoch

        # Add current best observed performance to the optimization trajectory
        self.best_config_trajectory.append(self.best_config_error)

        # Update runtime
        self.current_runtime += runtime
        self.cumulative_runtime.append(self.current_runtime)


def hpo_supervised_objective(config, train_data, val_data, paradigm):
    trainer = BaseTrainer(config, train_data, val_data, paradigm=paradigm, save_model=False)
    _, _, val_acc = trainer.train()
    return 1 - sum(val_acc) / len(val_acc) / 100


def hpo_contrastive_objective(config, pretext_data, train_data, val_data, test_data, labels):
    contra_trainer = BaseNIRSiamTrainer(config, pretext_data, train_data, val_data, test_data, labels, subject_id_list=[-1])
    _ = contra_trainer.pretext()
    _, val_loss, val_acc, _, test_acc = contra_trainer.linear_probe()

    return 1 - sum(val_acc) / len(val_acc) / 100, min(val_loss), test_acc


# random search
def hpo(config_dict, train_data, val_data, paradigm):
    config_space = {
        "lr": stats.loguniform(1e-5, 1e-3),
        "batch_size": stats.randint(32, 256),
        "dropout": stats.uniform(0, 0.75),
    }

    searcher = HPOSearcher(config_space)
    scheduler = HPOScheduler(searcher=searcher)
    tuner = HPOTuner(scheduler=scheduler, objective=hpo_supervised_objective)
    tuner.run(config_dict, train_data, val_data, paradigm)

    return tuner.best_config, best_test_acc


# grid search for supervised learning
def grid_search(config_dict, train_data, val_data, paradigm):
    config_list = create_grid(config_dict)
    searcher = GridSearcher(iter(config_list))
    scheduler = HPOScheduler(searcher=searcher)
    tuner = GridSearchTuner(scheduler=scheduler, objective=hpo_supervised_objective)
    tuner.run(train_data, val_data, paradigm=paradigm, num_epochs=15, num_trials=len(config_list))

    return tuner.best_config


# grid search for contrastive learning
def contra_grid_search(config_dict, pretext_data, train_data, val_data, test_data, labels):
    config_list = create_grid(config_dict)
    searcher = GridSearcher(iter(config_list))
    scheduler = HPOScheduler(searcher=searcher)
    tuner = GridSearchTuner(scheduler=scheduler, objective=hpo_contrastive_objective)
    tuner.run(train_data, val_data, pretext_data, test_data, labels=labels, num_epochs=15, num_trials=len(config_list), num_linear_epochs=100)

    best_test_acc = tuner.test_accs[tuner.min_val_accs.index(min(tuner.min_val_accs))]
    return tuner.best_config, best_test_acc
