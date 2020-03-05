from tensorflow.keras.optimizers import *
from tensorflow.keras.metrics import *
from os.path import join
import os

from constants.AI_params import *

# ----------------------------- UM -----------------------------------
_run_name = F'sobel_from_mnist'  # Name of the model, for training and classification
_output_folder = '/home/olmozavala/Dropbox/TutorialsByMe/TensorFlow/Examples/TestBedCNNs/OUTPUT'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Decide which GPU to use to execute the code

def append_model_params(cur_config):
    model_config = {
        ModelParams.MODEL: AiModels.DENSE_CNN,
        ModelParams.DROPOUT: False,
        ModelParams.BATCH_NORMALIZATION: False,
        # ModelParams.INPUT_SIZE: [28, 28],
        ModelParams.INPUT_SIZE: [1200, 1920],   # Just for the visualizer
        ModelParams.START_NUM_FILTERS: 1,
        ModelParams.FILTER_SIZE: 3,
        ModelParams.NUMBER_DENSE_LAYERS: 1, # In this case are 'DENSE' CNN
    }
    return {**cur_config, **model_config}


def get_training_config():
    cur_config = {
        TrainingParams.input_folder: '/home/olmozavala/Dropbox/TestData/MNIST/training',
        TrainingParams.output_folder: F"{join(_output_folder,'Training', _run_name)}",
        TrainingParams.cases: 'all', # This can be also a numpy array
        TrainingParams.validation_percentage: .1,
        TrainingParams.test_percentage: .1,
        TrainingParams.evaluation_metrics: [mean_squared_error],  # Metrics to show in tensor flow in the training
        TrainingParams.loss_function: mean_squared_error,  # Loss function to use for the learning
        TrainingParams.optimizer: Adam(),  # Default values lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
        TrainingParams.epochs: 1000,
        TrainingParams.config_name: _run_name,
    }
    return append_model_params(cur_config)


def get_model_viz_config():
    cur_config = {
        ClassificationParams.model_weights_file: join(_output_folder,'Training/sobel_from_mnist/models/sobel_from_mnist_2020_03_05_20_54-61-0.00000.hdf5'),
    }
    return append_model_params(cur_config)

