import os

import numpy as np
import tensorflow as tf

INPUT_TENSOR_NAME = 'inputs'
INPUT_SHAPE = 16


def estimator_fn(run_config, params):
    print('Running estimator_fn')
    feature_columns = [tf.feature_column.numeric_column(INPUT_TENSOR_NAME, shape=[INPUT_SHAPE])]
    return tf.estimator.LinearClassifier(feature_columns=feature_columns,
                                          n_classes=3,
                                          config=run_config)

def serving_input_fn(params):
    print('Running serving_input_fn')
    feature_spec = {INPUT_TENSOR_NAME: tf.FixedLenFeature(dtype=tf.float32, shape=[INPUT_SHAPE])}
    return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)()


def train_input_fn(training_dir, params):
    """Returns input function that would feed the model during training"""
    print('Running train_input_fn')
    return _generate_input_fn(training_dir, 'train.csv')


def eval_input_fn(training_dir, params):
    """Returns input function that would feed the model during evaluation"""
    print('Running eval_input_fn')
    return _generate_input_fn(training_dir, 'test.csv')


def _generate_input_fn(training_dir, training_filename):
    data_file = os.path.join(training_dir, training_filename)

    dataset = np.genfromtxt(data_file,delimiter=',',skip_header=1)
    y_data = dataset[:,0].astype(np.int)
    x_data = dataset[:,1:].astype(np.float32)

    return tf.estimator.inputs.numpy_input_fn(
        x={INPUT_TENSOR_NAME: x_data},
        y=y_data,
        num_epochs=None,
        shuffle=True)()




#
