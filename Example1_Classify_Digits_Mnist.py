from datetime import datetime
from os.path import join

from config.Classification_For_MNIST_Example import get_training_2d

from inout_common.io_common import create_folder

from constants.AI_params import *
import AI_proj.trainingutils as utilsNN
from models.modelSelector import select_2d_model

from tensorflow.keras.utils import plot_model
import tensorflow as tf

if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Reshaping the array to 4-dims so that it can work with the tensorflow.keras API
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print('Number of images in x_train', x_train.shape[0])
    print('Number of images in x_test', x_test.shape[0])

    config = get_training_2d()

    # -------- Reading configuration ---------
    output_folder = config[TrainingParams.output_folder]
    eval_metrics = config[TrainingParams.evaluation_metrics]
    loss_func = config[TrainingParams.loss_function]
    val_perc = config[TrainingParams.validation_percentage]
    epochs = config[TrainingParams.epochs]
    model_name_user = config[TrainingParams.config_name]
    optimizer = config[TrainingParams.optimizer]

    nn_input_size = config[ModelParams.INPUT_SIZE]
    model_type = config[ModelParams.MODEL]

    # -------- Setting up everything ---------
    weights_folder = join(output_folder, 'models')
    logs_folder = join(output_folder, 'logs')
    create_folder(weights_folder)
    create_folder(logs_folder)

    print("Selecting and generating the model....")
    now = datetime.utcnow().strftime("%Y_%m_%d_%H_%M")
    model_name = F'{model_name_user}_{now}'

    # # ******************* Selecting the model **********************
    model = select_2d_model(config)
    plot_model(model, to_file=join(output_folder,F'{model_name}.png'))


    print("Compiling model ...")
    model.compile(optimizer=optimizer, loss=loss_func, metrics=eval_metrics)

    [logger, save_callback, stop_callback] = utilsNN.get_all_callbacks(model_name=model_name,
                                                                       early_stopping_func=F'val_{eval_metrics[0].__name__}',
                                                                       weights_folder=weights_folder,
                                                                       logs_folder=logs_folder)

    model.fit(x=x_train, y=y_train, epochs=epochs,
              validation_split=val_perc,
              callbacks=[logger, save_callback, stop_callback])

    model.evaluate(x_test, y_test)
