try:
    import unzip_requirements
except ImportError:
    pass

import os
import datetime
import io

import pandas as pd
import numpy as np
import boto3
from boto3.dynamodb.conditions import Key, Attr
# import sagemaker.amazon.common as smac

from common import model_helper


def upload_data(event, context):
    """Creates a new s3 resource in the format for model training.
    Can be run globally or for a specific user, and expects user_id in event.
    Returns the resource id it created."""

    # Read in relevant environment variables, and allow for local run
    if event.get('runlocal'):
        print('Running local and using environment variable placeholders')
        bucket_prefix = 'sagemaker'
        region = 'us-east-1'
        stage = 'dev'
        bucket = 'wibsie-ml3-sagebucket-' + stage
        file_path = ''
    else:
        bucket = os.environ['SAGE_BUCKET']
        bucket_prefix = os.environ['SAGE_BUCKET_PREFIX']
        region = os.environ['REGION']
        stage = os.environ['STAGE']
        file_path = '/tmp/'

    user_id = event['user_id']
    now_epoch = get_epoch_ms()

    dynamodb = boto3.resource('dynamodb', region_name=region)

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
    feature_columns = ['age', 'bmi', 'gender', 'lifestyle', 'loc_type',
                        'apparent_temperature', 'cloud_cover', 'humidity',
                        'precip_intensity', 'precip_probability', 'temperature',
                        'wind_gust', 'wind_speed', 'precip_type', 'activity_met',
                        'total_clo']
    label_column = 'comfort_level_result'
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

        # Get float representation and convert to dict for pandas
        float_list = model_helper.table_to_floats(data_user=user_row,
                                                data_weatherreport=weather_row,
                                                data_experience=e,
                                                data_location=location_row)

        result_dict = {}
        for i in range(0,len(feature_columns)):
            result_dict[feature_columns[i]] = float_list[i]

        result_dict['comfort_level_result'] = model_helper.hash_comfort_level_result(e['comfort_level_result'])

        results.append(result_dict)

    # Make result first column
    columns = [label_column] + feature_columns

    data = pd.DataFrame(results)
    data = data[columns]

    # Fill all Nones
    # data['precip_type'] = data['precip_type'].fillna(value='')
    data['comfort_level_result'] = data['comfort_level_result'].fillna(value=-1) # may not be needed

    # Remove all rows without a label
    data = data[data['comfort_level_result'] >= 0]

    # Split into 80% train and 10% validation and 10% test
    rand_split = np.random.rand(len(data))
    train_list = rand_split < 0.7
    # val_list = (rand_split >= 0.8) & (rand_split < 0.9)
    test_list = rand_split >= 0.7

    data_train = data[train_list]
    # data_val = data[val_list]
    data_test = data[test_list]

    # s3 upload training file
    train_file = 'train.csv'

    data_train.to_csv(path_or_buf=file_path+train_file, index=False)

    train_s3path = os.path.join(bucket_prefix,user_id,'trainingfiles',str(now_epoch),train_file)

    boto3.Session().resource('s3').Bucket(bucket).Object(train_s3path).upload_file(file_path+train_file)

    # s3 upload test file
    test_file = 'test.csv'

    data_test.to_csv(path_or_buf=file_path+test_file, index=False)

    test_s3path = os.path.join(bucket_prefix,user_id,'trainingfiles',str(now_epoch),test_file)

    boto3.Session().resource('s3').Bucket(bucket).Object(test_s3path).upload_file(file_path+test_file)

    # Update user table with latest data and optionally create model key
    if not datakey_users[user_id].get('model'):
        response = table_users.update_item(
                        Key={'id': user_id},
                        UpdateExpression="""set model=:model""",
                        ExpressionAttributeValues={
                            ':model': {'train_created': now_epoch}
                        },
                        ReturnValues="UPDATED_NEW")

    else:
        response = table_users.update_item(
                        Key={'id': user_id},
                        UpdateExpression="""set model.train_created=:train_created""",
                        ExpressionAttributeValues={
                            ':train_created': now_epoch
                        },
                        ReturnValues="UPDATED_NEW")

    print('Update user succeeded')

    return {"message": "Experiences uploaded",
            "train_file": train_file,
            "test_file": test_file,
            "event": event}


def get_epoch_ms():
    """Helper function to current epoch int in ms"""
    now = datetime.datetime.utcnow()
    epoch = datetime.datetime.utcfromtimestamp(0)
    return int((now-epoch).total_seconds() * 1000.0)


#
