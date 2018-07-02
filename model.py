"""https://github.com/aws/sagemaker-python-sdk/tree/3fb5516d8e707bc0f9e93e45f4bfbc4bc3b4f576/src/sagemaker/tensorflow"""

import os

import numpy as np
import tensorflow as tf

INPUT_TENSOR_NAME = 'inputs'
INPUT_SHAPE = 16


def estimator_fn(run_config, params):
    print('Running estimator_fn')
    feature_columns = [tf.feature_column.numeric_column(INPUT_TENSOR_NAME, shape=[INPUT_SHAPE])]
    return tf.estimator.LinearClassifier(feature_columns=feature_columns,
                                          n_classes=2,
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





# # This doesn't work, likely needs to build from checkpoint
# >>> INPUT_TENSOR_NAME = 'inputs'
# >>> INPUT_SHAPE = 16
# >>> feature_columns = [tf.feature_column.numeric_column(INPUT_TENSOR_NAME, shape=[INPUT_SHAPE])]
# >>> tfi = tf.estimator.inputs.numpy_input_fn(x={INPUT_TENSOR_NAME: x_data},shuffle=False)
# >>> mdir = '/Users/mmorit202/repos/wibsie_ml_lambda3/export2/Servo/1530457035/'
# >>> tfe = tf.estimator.LinearClassifier(model_dir=mdir, feature_columns=feature_columns)
# >>> p = tfe.predict(input_fn=tfi)
# >>> for pi in p: print(pi)
#
# # This is actually intended to load savedmodel
# >>> from tensorflow.contrib import predictor as pred
# >>> pred_fn = pred.from_saved_model(mdir)
# >>> preds = pred_fn({"inputs": x_data})
#
# # Example creating list of expected format
# examples = [
#     tf.train.Example(
#       features=tf.train.Features(
#         feature={"inputs": tf.train.Feature(
#           float_list=tf.train.FloatList(value=x_lf))}))]
#
# # Gives information about the model and what is expects
# predictor_fn.__dict__
#
#
# # This actually works!!!!!!
# mdir = '/Users/mmorit202/repos/wibsie_ml_lambda3/export4/Servo/1530460455'
# from tensorflow.contrib import predictor
# predictor_fn = predictor.from_saved_model(mdir)
#
# x_lf = [3.0, 23.731674382716047, 0.0, 0.0, 0.0, 82.98, 0.14, 0.76, 0.0, 0.0, 79.62, 8.46, 5.21, 0.0, 2.5, 0.8]
# e = tf.train.Example(
#             features=tf.train.Features(
#             feature={"inputs": tf.train.Feature(
#             float_list=tf.train.FloatList(value=x_lf))}))
#
# s = e.SerializeToString()
# p = predictor_fn({"inputs": [s]})
#
# # Output
# >>> p
# {'classes': array([[b'0', b'1', b'2']], dtype=object), 'scores': array([[9.9999654e-01, 3.5017094e-06, 0.0000000e+00]], dtype=float32)}



#
