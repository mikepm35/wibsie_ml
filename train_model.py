try:
    import unzip_requirements
except ImportError:
    pass

import os
import datetime
import io
import decimal

import pandas as pd
import numpy as np
import boto3
from boto3.dynamodb.conditions import Key, Attr
import sagemaker as sm
import sagemaker.amazon.common as smac


LINEAR_CONTAINERS = {
    'us-west-2': '174872318107.dkr.ecr.us-west-2.amazonaws.com/linear-learner:latest',
    'us-east-1': '382416733822.dkr.ecr.us-east-1.amazonaws.com/linear-learner:latest',
    'us-east-2': '404615174143.dkr.ecr.us-east-2.amazonaws.com/linear-learner:latest',
    'eu-west-1': '438346466558.dkr.ecr.eu-west-1.amazonaws.com/linear-learner:latest',
    'ap-northeast-1': '351501993468.dkr.ecr.ap-northeast-1.amazonaws.com/linear-learner:latest'}


def train(event, context):
    """Uploads fresh data from dynamodb and runs training.
    Can run globally or for a user, and expects user_id in event."""

    # Read in relevant environment / role variables
    sm_role = sm.get_execution_role()
    bucket = os.environ['SAGE_BUCKET']
    bucket_prefix = os.environ['SAGE_BUCKET_PREFIX']

    return {"message": "Train function executed successfully",
            "event": event}


def upload_data(event, context):
    """Creates a new s3 resource in the format for model training.
    Can be run globally or for a specific user, and expects user_id in event.
    Returns the resource id it created."""

    # Read in relevant environment variables, and allow for local run
    try:
        bucket = os.environ['SAGE_BUCKET']
        bucket_prefix = os.environ['SAGE_BUCKET_PREFIX']
        region = os.environ['REGION']
        stage = os.environ['STAGE']
    except:
        print('Using environment variable placeholders')
        bucket = 'wibsie-ml3-local-sagemaker-test'
        bucket_prefix = 'sagemaker/trainingfiles'
        region = 'us-east-1'
        stage = 'dev'

    user_id = event['user_id']
    now_epoch = get_epoch_ms()

    dynamodb = boto3.resource('dynamodb',
                                region_name=region)

    # Read in all table data (numbers are type Decimal) and organize by keys
    ##Users
    table_users = dynamodb.Table('wibsie-users-'+stage)

    if user_id == 'global':
        response = table_users.scan()
        data_users = response['Items']

        while 'LastEvaluatedKey' in response:
            response = table_users.scan(
                            ExclusiveStartKey=response['LastEvaluatedKey'])
            data_users += response['Items']

    else:
        response = table_users.query(
                        KeyConditionExpression=Key('id').eq(user_id))
        data_users = response['Items']

    datakey_users = {}
    for u in data_users:
        datakey_users[u['id']] = u

    ##Experiences
    table_experiences = dynamodb.Table('wibsie-experiences-'+stage)
    if user_id == 'global':
        response = table_experiences.scan()
        data_experiences = response['Items']

        while 'LastEvaluatedKey' in response:
            response = table_experiences.scan(
                            ExclusiveStartKey=response['LastEvaluatedKey'])
            data_experiences += response['Items']

    else:
        response = table_experiences.query(
                        KeyConditionExpression=Key('user_id').eq(user_id))
        data_experiences = response['Items']

    # Return if no experiences found
    if len(data_experiences) < 10:
        return {"message": "Too few experiences found: {}".format(len(data_experiences)),
                "train_file": "",
                "test_file": "",
                "event": event}

    ##Locations - TODO: Filter based on experiences
    table_locations = dynamodb.Table('wibsie-locations-'+stage)
    response = table_locations.scan()
    data_locations = response['Items']

    while 'LastEvaluatedKey' in response:
        response = table_locations.scan(
                        ExclusiveStartKey=response['LastEvaluatedKey'])
        data_locations += response['Items']

    datakey_locations = {}
    for l in data_locations:
        datakey_locations[l['zip']] = l

    ##Weather reports - TODO: Filter based on experiences
    table_weatherreports = dynamodb.Table('wibsie-weatherreports-'+stage)
    response = table_weatherreports.scan()
    data_weatherreports = response['Items']

    while 'LastEvaluatedKey' in response:
        response = table_weatherreports.scan(
                        ExclusiveStartKey=response['LastEvaluatedKey'])
        data_weatherreports += response['Items']

    datakey_weatherreports = {}
    for w in data_weatherreports:
        key = w['zip'] + str(w['expires'])
        datakey_weatherreports[key] = w

    # Build a join around experiences
    results = []
    for e in data_experiences:
        # Join to other data
        user_row = datakey_users.get(e['user_id'])
        if not user_row:
            raise Exception('Did not find user row for: ', e['user_id'])

        location_row = datakey_locations.get(e['zip'])
        if not location_row:
            raise Exception('Did not find location row for: ', e['zip'])

        weather_row = datakey_weatherreports.get(e['zip']+str(e['weather_expiration']))
        if not weather_row:
            raise Exception('Did not find weather row for: ', e['zip']+str(e['weather_expiration']))

        if 'precipType' not in weather_row:
            weather_row['precipType'] = None

        # Convert activity to met
        if e['activity'] == 'standing':
            activity_met = 1.1
        elif e['activity'] == 'walking':
            activity_met = 2.5
        elif e['activity'] == 'exercising':
            activity_met = 6.0
        else:
            raise Exception('Unrecognized activity: ', e['activity'])

        # Convert clothing to clo
        if e['upper_clothing'] == 'no_shirt':
            upper_clo = 0.0
        elif e['upper_clothing'] == 'short_sleeves':
            upper_clo = 0.2
        elif e['upper_clothing'] == 'long_sleeves':
            upper_clo = 0.4
        elif e['upper_clothing'] == 'jacket':
            upper_clo = 0.6
        else:
            raise Exception('Unrecognized upper clothing: ', e['upper_clothing'])

        if e['lower_clothing'] == 'shorts':
            lower_clo = 0.2
        elif e['lower_clothing'] == 'pants':
            lower_clo = 0.4
        else:
            raise Exception('Unrecognized lower clothing: ', e['lower_clothing'])

        total_clo = upper_clo + lower_clo

        # Get age
        age = datetime.datetime.now().year - int(user_row['birth_year'])

        # Add row
        results.append({'age': age,
                        'bmi': float(user_row['bmi']),
                        'gender': user_row['gender'],
                        'lifestyle': user_row['lifestyle'],
                        'loc_type': location_row['loc_type'],
                        'apparent_temperature': float(weather_row['apparentTemperature']),
                        'cloud_cover': float(weather_row['cloudCover']),
                        'humidity': float(weather_row['humidity']),
                        'precip_intensity': float(weather_row['precipIntensity']),
                        'precip_probability': float(weather_row['precipProbability']),
                        'temperature': float(weather_row['temperature']),
                        'wind_gust': float(weather_row['windGust']),
                        'wind_speed': float(weather_row['windSpeed']),
                        'precip_type': weather_row['precipType'],
                        'activity_met': activity_met,
                        'total_clo': total_clo,
                        'comfort_level_result': e['comfort_level_result']})

    label_column = 'comfort_level_result'
    feature_columns = [l for l in results[0].keys() if l != label_column]

    data = pd.DataFrame(results)
    return data

    # Fill all Nones in precip_type
    data['precip_type'] = data['precip_type'].fillna(value='')

    # Split into 80% train and 20% test
    rand_split = np.random.rand(len(data))
    train_list = rand_split < 0.8
    test_list = rand_split >= 0.8

    data_train = data[train_list]
    data_test = data[test_list]

    # Take 1 column as label, evaluate for true/false via a condition,
    # use +0 to convert bool to 0/1, and as_matrix to convert to numpy array
    label_true = 'comfortable'

    train_y = ((data_train.loc[:,label_column] == label_true) +0).as_matrix()
    train_x = data_train.loc[:,feature_columns].as_matrix()

    #return train_x, train_y

    test_y = ((data_test.loc[:,label_column] == label_true) +0).as_matrix()
    test_x = data_test.loc[:,feature_columns].as_matrix()

    # s3 upload training file
    train_file = user_id + '_train_' + str(now_epoch) + '.data'

    f = io.BytesIO()
    smac.write_numpy_to_dense_tensor(f, train_x, train_y)
    f.seek(0)

    boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(bucket_prefix, 'train', train_file)).upload_fileobj(f)

    # s3 upload test file
    test_file = user_id + '_test_' + str(now_epoch) + '.data'

    f = io.BytesIO()
    smac.write_numpy_to_dense_tensor(f, test_x, test_y)
    f.seek(0)

    boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(bucket_prefix, 'test', train_file)).upload_fileobj(f)

    return {"message": "No experiences found",
            "train_file": train_file,
            "test_file": test_file,
            "event": event}


def get_epoch_ms():
    """Helper function to current epoch int in ms"""
    now = datetime.datetime.utcnow()
    epoch = datetime.datetime.utcfromtimestamp(0)
    return int((now-epoch).total_seconds() * 1000.0)


#
