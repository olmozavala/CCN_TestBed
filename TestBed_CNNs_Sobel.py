from datetime import datetime
from os.path import join

from config.Config_CNN_Sobel import get_training_config

from inout_common.io_common import create_folder

from constants.AI_params import *
import trainingutils as utilsNN
from models.modelSelector import select_2d_model
from AI_proj.data_generation.GeneratorsSobel import data_gen_sobel

from tensorflow.keras.utils import plot_model
import tensorflow as tf

if __name__ == '__main__':
    config = get_training_config()

    # -------- Reading configuration ---------
    input_folder = config[TrainingParams.input_folder]
    output_folder = config[TrainingParams.output_folder]
    eval_metrics = config[TrainingParams.evaluation_metrics]
    loss_func = config[TrainingParams.loss_function]
    val_perc = config[TrainingParams.validation_percentage]
    test_perc = config[TrainingParams.test_percentage]
    epochs = config[TrainingParams.epochs]
    model_name_user = config[TrainingParams.config_name]
    optimizer = config[TrainingParams.optimizer]

    nn_input_size = config[ModelParams.INPUT_SIZE]
    model_type = config[ModelParams.MODEL]

    tot_examples = 60000

    # ================= Making folders and defining the splits ==========
    split_info_folder = join(output_folder, 'Splits')
    parameters_folder = join(output_folder, 'Parameters')
    weights_folder = join(output_folder, 'models')
    logs_folder = join(output_folder, 'logs')
    create_folder(split_info_folder)
    create_folder(parameters_folder)
    create_folder(weights_folder)
    create_folder(logs_folder)

    # ================ Split definition =================
    [train_ids, val_ids, test_ids] = utilsNN.split_train_validation_and_test(tot_examples,
                                                                             val_percentage=val_perc,
                                                                             test_percentage=test_perc)

    print(F"Train examples (total:{len(train_ids)}) :{train_ids}")
    print(F"Validation examples (total:{len(val_ids)}) :{val_ids}:")
    print(F"Test examples (total:{len(test_ids)}) :{test_ids}")

    # ======== Setting up everything =========
    print("Selecting and generating the model....")
    now = datetime.utcnow().strftime("%Y_%m_%d_%H_%M")
    model_name = F'{model_name_user}_{now}'

    # # ******************* Selecting the model **********************
    model = select_2d_model(config)
    plot_model(model, to_file=join(output_folder,F'{model_name}.png'), show_shapes=True)


    print("Compiling model ...")
    model.compile(optimizer=optimizer, loss=loss_func, metrics=eval_metrics)

    [logger, save_callback, stop_callback] = utilsNN.get_all_callbacks(model_name=model_name,
                                                                       early_stopping_func=F'val_{eval_metrics[0].__name__}',
                                                                       weights_folder=weights_folder,
                                                                       logs_folder=logs_folder)

    train_gen = data_gen_sobel(input_folder, train_ids)
    val_gen = data_gen_sobel(input_folder, val_ids)

    model.fit_generator(train_gen, steps_per_epoch=min(100, len(train_ids)),
                        validation_data=val_gen,
                        validation_steps=min(20, len(val_ids)),
                        use_multiprocessing=False,
                        workers=1,
                        # validation_freq=10, # How often to compute the validation loss
                        epochs=epochs, callbacks=[logger, save_callback, stop_callback])

