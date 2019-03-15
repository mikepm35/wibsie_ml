import os

try:
    import numpy as np
    import tensorflow as tf
except:
    print('Failed to import numpy, tensorflow in model.py')


FEATURE_COLS = [
    'user_id',
    # 'total_clo',
    # 'precip_intensity',
    # 'precip_type',
    # 'activity_met',
    # 'wind_speed',
    # 'temperature',
    # 'apparent_temperature',
    # 'humidity',
    # 'cloud_cover'
    'h_intprod',
    'ed_watvapdiff',
    'esw_sweat',
    'ere_resp',
    'l_dryresp',
    'r_radiation',
    'c_convection'
    # 'pmv',
    # 'ppd'
]

CROSSED_COLS = [
    # ['activity_met', 'total_clo'],
    # ['total_clo', 'wind_speed'],
    # ['humidity', 'activity_met']
]

LABEL_COL = 'comfort_level_result'


def get_feature_columns():
    my_numeric_columns = []

    # base columns
    for col in FEATURE_COLS:
        my_numeric_columns.append(tf.feature_column.numeric_column(col))

    # derived columns
    for col in CROSSED_COLS:
        my_numeric_columns.append(tf.feature_column.crossed_column(
            col, hash_bucket_size=1000))

    return my_numeric_columns


def easy_input_function(data_dict, label_key, num_epochs, shuffle, batch_size):
    """Creates input dataset for model; at larger scale should use io streaming
    from file."""

    label_array = np.array(data_dict[label_key], dtype=int)

    data_array = {}
    for key in data_dict:
        if key != label_key:
            data_array[key] = np.array(data_dict[key], dtype=float)

    ds = tf.data.Dataset.from_tensor_slices((data_array,label_array))

    if shuffle:
        ds = ds.shuffle(10000)

    ds = ds.batch(batch_size).repeat(num_epochs)

    return ds




#
