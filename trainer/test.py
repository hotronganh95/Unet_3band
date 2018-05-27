import os
import model
import argparse
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import h5py

from datetime import datetime
from tensorflow.python.lib.io import file_io
from io import BytesIO
from keras.callbacks import (ModelCheckpoint, TensorBoard, CSVLogger, History, EarlyStopping, LambdaCallback)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def initialise_hyper_params(args_parser):
    """
    Define the arguments with the default values,
    parses the arguments passed to the task,
    and set the HYPER_PARAMS global variable

    Args:
        args_parser
    """

    # data files arguments
    args_parser.add_argument(
        '--num-channel',
        help='Image channel (3 or 8 band)',
        required=True
    )


    args_parser.add_argument(
        '--xtrain-files',
        help='GCS or local paths to x train',
        required=True
    )

    args_parser.add_argument(
        '--ytrain-files',
        help='GCS or local paths to x train',
        required=True
    )

    args_parser.add_argument(
        '--xval-files',
        help='GCS or local paths to x val',
        required=True
    )

    args_parser.add_argument(
        '--yval-files',
        help='GCS or local paths to y val',
        required=True
    )


    args_parser.add_argument(
        '--num-epochs',
        help="""\
        Maximum number of training data epochs on which to train.
        .\
        """,
        default=20,
        type=int,
    )

    # Saved model arguments
    args_parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )

    args_parser.add_argument(
        '--prefix',
        help='prefix for save files',
        required=True
    )

    args_parser.add_argument(
        '--batch-size',
        help='Batch size',
        default = 32,
        type=int
    )

    return args_parser.parse_args()

def run_experiment():
    print("* experiment configurations")
    print("===========================")
    print("Epoch count: {}".format(HYPER_PARAMS.num_epochs))
    print("Image channel: {}".format(HYPER_PARAMS.num_channel))

    data_set = file_io.File_IO('gs:///building_footprint/data/20180310/xtrain_test.npy')

    print("X train shape: {}".format(xtrain.shape))
    print("Y train shape: {}".format(ytrain.shape))
    print("===========================")

    FMT_VALMODEL_PATH ="{}_val_weights.h5"
    FMT_VALMODEL_LAST_PATH = "{}_val_weights_last.h5"
    FMT_VALMODEL_HIST = "{}_val_hist.csv"
    PREFIX = HYPER_PARAMS.prefix + "_test"
    INPUT_CHANNEL =  HYPER_PARAMS.num_channel

    unet = model.get_unet(INPUT_CHANNEL)

    # train and evaluate
    model_checkpoint = ModelCheckpoint(
        FMT_VALMODEL_PATH.format(PREFIX + "_{epoch:02d}"),
        monitor='val_jaccard_coef_int',
        save_best_only=False)

    model_earlystop = EarlyStopping(
        monitor='val_jaccard_coef_int',
        patience=10,
        verbose=0,
        mode='max')

    model_history = History()

    model_board = TensorBoard(
        log_dir=os.path.join(HYPER_PARAMS.job_dir, 'logs'),
        histogram_freq=0,
        write_graph=True,
        embeddings_freq=0)

    save_checkpoint_gcs = LambdaCallback(
        on_epoch_end=lambda epoch, logs: copy_file_to_gcs(HYPER_PARAMS.job_dir, FMT_VALMODEL_PATH.format(PREFIX + '_' + str(format(epoch + 1, '02d')))))

    unet.fit(
        xtrain, ytrain,
        nb_epoch=HYPER_PARAMS.num_epochs,
        batch_size = HYPER_PARAMS.batch_size,
        shuffle=True,
        verbose=1,
        validation_data=(xval, yval),
        callbacks=[model_checkpoint, model_earlystop, model_history, model_board, save_checkpoint_gcs])

    pd.DataFrame(model_history.history).to_csv(FMT_VALMODEL_HIST.format(PREFIX), index=False)
    copy_file_to_gcs(HYPER_PARAMS.job_dir, FMT_VALMODEL_HIST.format(PREFIX))

    unet.save_weights(FMT_VALMODEL_LAST_PATH.format(PREFIX))
    copy_file_to_gcs(HYPER_PARAMS.job_dir, FMT_VALMODEL_LAST_PATH.format(PREFIX))

def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='r') as input_f:
        with file_io.FileIO(os.path.join(job_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())


def main():

    print('')
    print('Hyper-parameters:')
    print(HYPER_PARAMS)
    print('')

    time_start = datetime.utcnow()
    print("")
    print("Experiment started at {}".format(time_start.strftime("%H:%M:%S")))
    print(".......................................")

    run_experiment()

    time_end = datetime.utcnow()
    print(".......................................")
    print("Experiment finished at {}".format(time_end.strftime("%H:%M:%S")))
    print("")
    time_elapsed = time_end - time_start
    print("Experiment elapsed time: {} seconds".format(time_elapsed.total_seconds()))
    print("")


args_parser = argparse.ArgumentParser()
HYPER_PARAMS = initialise_hyper_params(args_parser)


if __name__ == '__main__':
    main()
