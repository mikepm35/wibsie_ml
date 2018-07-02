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

import model_helper


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

        # Fetch continuous representations
        activity_met = model_helper.activity_to_met(e['activity'])

        upper_clo = model_helper.upper_clothing_to_clo(e['upper_clothing'])

        lower_clo = model_helper.lower_clothing_to_clo(e['lower_clothing'])

        total_clo = upper_clo + lower_clo

        # Get age
        age = model_helper.birth_year_to_age(int(user_row['birth_year']))

        # Add row
        results.append({'age': float(age),
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
                        'activity_met': float(activity_met),
                        'total_clo': float(total_clo),
                        'comfort_level_result': e['comfort_level_result']})

    label_column = 'comfort_level_result'
    feature_columns = [l for l in results[0].keys() if l != label_column]
    columns = [label_column] + feature_columns

    data = pd.DataFrame(results)
    data = data[columns]

    # Fill all Nones
    data['precip_type'] = data['precip_type'].fillna(value='')
    data['comfort_level_result'] = data['comfort_level_result'].fillna(value='none')

    # Remove all rows without a label
    data = data[data['comfort_level_result'] != 'none']

    # Convert categorical data to floats
    data['gender'] = data['gender'].apply(model_helper.hash_gender)
    data['lifestyle'] = data['lifestyle'].apply(model_helper.hash_lifestyle)
    data['loc_type'] = data['loc_type'].apply(model_helper.hash_loc_type)
    data['precip_type'] = data['precip_type'].apply(model_helper.hash_precip_type)
    # data['gender'] = data['gender'].astype('category').cat.codes
    # data['lifestyle'] = data['lifestyle'].astype('category').cat.codes
    # data['loc_type'] = data['loc_type'].astype('category').cat.codes
    # data['precip_type'] = data['precip_type'].astype('category').cat.codes

    # Convert data to buckets and then float
    data['age'] = data['age'].apply(model_helper.hash_age)
    # age_buckets = [0,18, 25, 30, 35, 40, 45, 50, 55, 60, 65,150]
    # data['age'] = pd.cut(data.age, age_buckets, right=True).astype('category').cat.codes

    # Convert label column to integer (comfortable=1)
    data['comfort_level_result'] = data['comfort_level_result'].apply(model_helper.hash_comfort_level_result)
    # data['comfort_level_result'] = ((data.comfort_level_result == 'comfortable') +0)

    # Split into 80% train and 10% validation and 10% test
    rand_split = np.random.rand(len(data))
    train_list = rand_split < 0.9
    # val_list = (rand_split >= 0.8) & (rand_split < 0.9)
    test_list = rand_split >= 0.9

    data_train = data[train_list]
    # data_val = data[val_list]
    data_test = data[test_list]

    # # Convert to matricies (for protobuf used by sagemaker)
    # train_y = data_train.iloc[:,0].as_matrix()
    # train_X = data_train.iloc[:,1:].as_matrix()
    #
    # val_y = data_val.iloc[:,0].as_matrix()
    # val_X = data_val.iloc[:,1:].as_matrix()
    #
    # test_y = data_test.iloc[:,0].as_matrix()
    # test_X = data_test.iloc[:,1:].as_matrix()

    # s3 upload training file
    train_file = 'train.csv'

    data_train.to_csv(path_or_buf=file_path+train_file, index=False)

    train_s3path = os.path.join(bucket_prefix,user_id,'trainingfiles',str(now_epoch),train_file)

    boto3.Session().resource('s3').Bucket(bucket).Object(train_s3path).upload_file(train_file)

    # s3 upload test file
    test_file = 'test.csv'

    data_test.to_csv(path_or_buf=file_path+test_file, index=False)

    test_s3path = os.path.join(bucket_prefix,user_id,'trainingfiles',str(now_epoch),test_file)

    boto3.Session().resource('s3').Bucket(bucket).Object(test_s3path).upload_file(test_file)

    # # s3 upload training file (protobuf used by sagemaker)
    # train_file = user_id + '_train_' + str(now_epoch) + '.data'
    #
    # f = io.BytesIO()
    # smac.write_numpy_to_dense_tensor(f, train_X.astype('float32'), train_y.astype('float32'))
    # f.seek(0)
    #
    # boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(bucket_prefix, 'train', train_file)).upload_fileobj(f)
    #
    # # s3 upload validation file
    # val_file = user_id + '_validation_' + str(now_epoch) + '.data'
    #
    # f = io.BytesIO()
    # smac.write_numpy_to_dense_tensor(f, val_X.astype('float32'), val_y.astype('float32'))
    # f.seek(0)
    #
    # boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(bucket_prefix, 'validation', val_file)).upload_fileobj(f)
    #
    # # s3 upload test file
    # test_file = user_id + '_test_' + str(now_epoch) + '.data'
    #
    # f = io.BytesIO()
    # smac.write_numpy_to_dense_tensor(f, test_X.astype('float32'), test_y.astype('float32'))
    # f.seek(0)
    #
    # boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(bucket_prefix, 'test', test_file)).upload_fileobj(f)

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
