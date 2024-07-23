# code_for_UIST2024
Details will be available after the related paper is published.

## Folder Structure
- contra_loss: 
	- contra_loss.py: calculate the contrastive loss, inlucde implementation of the classic SimCLR and SimSiam loss.
- data
	- __init__.py
	- config_dataset.yml: configure dataset settings
	- dataset.py: define the dataset loader
- data_aug
	- __init__.py: define the augmentation methods
- models
	- __init__.py: return the selected model
	- DeepConvNet.py
	- EEGNet.py
	- NIRSformer.py
	- NIRSiam.py
	- base_models.py
	- config_deepconvnet.yml
	- config_eegnet.yml
	- config_nirsformer.yml
	- config_simcnn.yml
- utils
	- __init__.py
	- early_stopper.py
	- hyperparameter_optimizer.py
- base_experiment_runner.py: define the base experiment runner for initializing experiment settings, splitting data, loading data, and saving results
- base_nirsiam_trainer.py: define the base trainer for NIRSiam adaptive learning phase 1
- base_trainer.py: define the base trainer for supervised learning
- main_nirsiam.py: entrance for running NIRSiam adaptive learning experiments
- main_supervised.py: entrance for running supervised experiments
	

## Datasets
Modify the codes to fit the algorithms to your dataset or other available open-access datsets. The default datasets used in this repo are: fNIRS2MW visual n-back(link) and fNIRS2MW audio n-back (link). 

More details in these datasets' description can be found from the website.

###

## How to run the experiments

### supervised

1. Download the data and add its directory to data/config_dataset.yml, global_path.

2. Update the parameter of SupervisedExpRunner() in main_supervised.py to be the yaml file of the encoder model you choose, from a list of: 
	```
	a. config_deepconvnet.yml;
	b. config_eegnet.yml;
	c. config_nirsformer.yml;
	d. config_simcnn.yml;
	e. any other config Yaml file of your own model.
	```

3. Prepare the model's yaml file to be under supervised settings, including:
	```
	a. model: "modelname";
	b. paradigm:  "subject-specific" or "generic"; 
	c. mode: "xy";
	```

4. Uncomment the use of linear classifier in model's forward method.

5. Run command: 
	```
	python main_supervised.py
	```

### NIRSiam

1. Download the data and add its directory to data/config_dataset.yml, global_path.

2. Update the parameter of NIRSiamExpRunner() in main_nirsiam.py to be the yaml file of the encoder model you choose, from a list of: 
	```
	a. config_deepconvnet.yml;
	b. config_eegnet.yml;
	c. config_nirsformer.yml;
	d. config_simcnn.yml;
	e. any other config Yaml file of your own model.
	```

3. Prepare the model's yaml file to be under contrastive settings, including: 
	```
	a. model:  "contra_ + modelname" ; 
	b. paradigm: "contrastive";
	c. mode: "contrastive".
	```

4. Comment the use of linear classifier in model's forward function.

5. Run command: 
	```
	python main_nirsiam.py
	```

## Citation
TODO



