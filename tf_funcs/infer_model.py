try:
    import unzip_requirements
except ImportError:
    pass

import os
import datetime
import io
from decimal import Decimal
import time
import json
import tarfile
import shutil

import boto3
from boto3.dynamodb.conditions import Key, Attr
import botocore
import tensorflow as tf
from tensorflow.contrib import predictor

from common import model_helper


def infer(event, context):
    """Deploy previously uploaded model locally and make a prediction"""

    # Read in relevant environment variables, and allow for local run
    if event.get('runlocal'):
        print('Running local and using environment variable placeholders')
        bucket = 'wibsie-ml3-sagebucket-dev'
        bucket_prefix = 'sagemaker'
        region = 'us-east-1'
        stage = 'dev'
        role = 'arn:aws:iam::530583866435:role/service-role/AmazonSageMaker-ExecutionRole-20180616T150039'
        file_path = ''
    else:
        print('Running using importing environments')
        bucket = os.environ['SAGE_BUCKET']
        bucket_prefix = os.environ['SAGE_BUCKET_PREFIX']
        region = os.environ['REGION']
        stage = os.environ['STAGE']
        service = os.environ['SERVICE']
        function_prefix = os.environ['FUNCTION_PREFIX']
        role = os.environ['SAGE_ROLE']
        file_path = '/tmp/'
        #print('SM execution role: ', sm.get_execution_role()) #not needed

    now_epoch = round(time.time()*1000)
    user_id = event['user_id']
    user_bucket = os.path.join(bucket,bucket_prefix,user_id)
    experience_created = int(event['experience_created'])

    dynamodb = boto3.resource('dynamodb', region_name=region)

    # Retrieve user info
    table_users = dynamodb.Table('wibsie-users-'+stage)

    response = table_users.query(
                    KeyConditionExpression=Key('id').eq(user_id))
    data_users = response['Items']

    if len(data_users) == 0:
        return {"message": "No user found",
                "event": event}

    else:
        data_user = data_users[0]

    # Check user model details
    model_valid = False
    model_keys_expected = ['model_created', 'model_job_name',
                            'model_created_prev', 'model_job_name_prev']

    if data_user.get('model'):
        model_keys = data_user['model'].keys()

        for k in model_keys_expected:
            if k not in model_keys:
                break

        # Convert created decimal to int
        if type(data_user['model']['model_created']) == Decimal:
            data_user['model']['model_created'] = int(data_user['model']['model_created'])

        if type(data_user['model']['model_created_prev']) == Decimal:
            data_user['model']['model_created_prev'] = int(data_user['model']['model_created_prev'])

        model_valid = True

    if not model_valid:
        return {"message": "Valid model details not found",
                "event": event}

    #TODO: Clean-up all this duplicate code
    data_user['model']['model_available'] = False

    suf_list = ['']
    if data_user['model']['model_created_prev'] != 'none':
        suf_list.append('_prev')

    for suf in suf_list:
        print('Attempting model suffix: ', suf_list.index(suf))
        model_artifacts_location = os.path.join(bucket_prefix,user_id,'models',str(data_user['model']['model_created'+suf]),data_user['model']['model_job_name'+suf],'output')
        model_prefix = 'model_' + user_id + '_' + str(data_user['model']['model_created'+suf])
        local_file = model_prefix + '.tar.gz'
        local_file_path = file_path + local_file
        extract_path = file_path + model_prefix

        # Only download and extract if data doesn't already exist
        if not os.path.exists(extract_path):
            print('Downloading and extracting data: ', model_artifacts_location, local_file_path, extract_path)
            try:
                boto3.Session().resource('s3').Bucket(bucket).download_file(model_artifacts_location+'/model.tar.gz',local_file_path)
                tarfile.open(local_file_path, 'r').extractall(extract_path)
                data_user['model']['model_available'] = True
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    print("Primary model zip file does not exist.")
                else:
                    raise
        else:
            data_user['model']['model_available'] = True

        if data_user['model']['model_available']:
            print('Using model suffix: ', suf_list.index(suf))
            data_user['model']['model_created_available'] = data_user['model']['model_created'+suf]
            data_user['model']['model_job_name_available'] = data_user['model']['model_job_name'+suf]
            data_user['model']['model_extract_path_available'] = extract_path
            break

    # # Attempt primary model
    # model_artifacts_location = os.path.join(bucket_prefix,user_id,'models',str(data_user['model']['model_created']),data_user['model']['model_job_name'],'output')
    # model_prefix = 'model_' + user_id + '_' + str(data_user['model']['model_created'])
    # local_file = model_prefix + '.tar.gz'
    # local_file_path = file_path + local_file
    # extract_path = file_path + model_prefix
    #
    # # Only download and extract if data doesn't already exist
    # if not os.path.exists(extract_path):
    #     print('Downloading and extracting data: ', model_artifacts_location, local_file_path, extract_path)
    #     try:
    #         boto3.Session().resource('s3').Bucket(bucket).download_file(model_artifacts_location+'/model.tar.gz',local_file_path)
    #         tarfile.open(local_file, 'r').extractall(extract_path)
    #         data_user['model']['model_available'] = True
    #         data_user['model']['model_created_available'] = data_user['model']['model_created']
    #         data_user['model']['model_job_name_available'] = data_user['model']['model_job_name']
    #         data_user['model']['model_extract_path_available'] = extract_path
    #     except botocore.exceptions.ClientError as e:
    #         if e.response['Error']['Code'] == "404":
    #             print("Primary model zip file does not exist.")
    #         else:
    #             raise
    # else:
    #     data_user['model']['model_available'] = True
    #     data_user['model']['model_created_available'] = data_user['model']['model_created']
    #     data_user['model']['model_job_name_available'] = data_user['model']['model_job_name']
    #     data_user['model']['model_extract_path_available'] = extract_path
    #
    # # Get backup model if needed
    # if data_user['model']['model_created_prev'] != 'none' and not data_user['model']['model_available']:
    #     model_artifacts_location = os.path.join(bucket_prefix,user_id,'models',str(data_user['model']['model_created_prev']),data_user['model']['model_job_name_prev'],'output')
    #     model_prefix = 'model_' + user_id + '_' + str(data_user['model']['model_created_prev'])
    #     local_file = model_prefix + '.tar.gz'
    #     local_file_path = file_path + local_file
    #     extract_path = file_path + model_prefix
    #
    #     # Only download and extract if data doesn't already exist
    #     if not os.path.exists(extract_path):
    #         print('Downloading and extracting data: ', model_artifacts_location, local_file_path, extract_path)
    #         try:
    #             boto3.Session().resource('s3').Bucket(bucket).download_file(model_artifacts_location+'/model.tar.gz',local_file_path)
    #             tarfile.open(local_file, 'r').extractall(extract_path)
    #             data_user['model']['model_available'] = True
    #             data_user['model']['model_created_available'] = data_user['model']['model_created_prev']
    #             data_user['model']['model_job_name_available'] = data_user['model']['model_job_name_prev']
    #             data_user['model']['model_extract_path_available'] = extract_path
    #         except botocore.exceptions.ClientError as e:
    #             if e.response['Error']['Code'] == "404":
    #                 print("Previous model zip file does not exist.")
    #             else:
    #                 raise
    #     else:
    #         data_user['model']['model_available'] = True
    #         data_user['model']['model_created_available'] = data_user['model']['model_created_prev']
    #         data_user['model']['model_job_name_available'] = data_user['model']['model_job_name_prev']
    #         data_user['model']['model_extract_path_available'] = extract_path

    if not data_user['model']['model_available']:
        return {"message": "No model could be resolved",
                "event": event}

    # Get long path to extracted pb file
    data_user['model']['model_pb_path_available'] = None
    for root, dirs, files in os.walk(data_user['model']['model_extract_path_available']):
        for file in files:
            if file.endswith('.pb'):
                data_user['model']['model_pb_path_available'] = root
                break

    if not data_user['model']['model_pb_path_available']:
        return {"message": "Model pb path could not be resolved",
                "event": event}

    # Retrieve experience data
    table_experiences = dynamodb.Table('wibsie-experiences-'+stage)

    response = table_experiences.query(
                    KeyConditionExpression=Key('created').eq(experience_created) & Key('user_id').eq(user_id)
                    )
    data_experiences = response['Items']

    if len(data_experiences) == 0:
        return {"message": "No experience found",
                "event": event}
    else:
        data_experience = data_experiences[0]

    # Get weather data
    table_weatherreports = dynamodb.Table('wibsie-weatherreports-'+stage)

    response = table_weatherreports.query(
                    KeyConditionExpression=Key('expires').eq(int(data_experience['weather_expiration'])) & Key('zip').eq(data_experience['zip'])
                    )
    data_weatherreports = response['Items']

    if len(data_weatherreports) == 0:
        return {"message": "No weather report found",
                "event": event}
    else:
        data_weatherreport = data_weatherreports[0]

    # Get location data
    table_locations = dynamodb.Table('wibsie-locations-'+stage)

    response = table_locations.query(
                    KeyConditionExpression=Key('zip').eq(data_experience['zip'])
                    )
    data_locations = response['Items']

    if len(data_locations) == 0:
        return {"message": "No location data found",
                "event": event}
    else:
        data_location = data_locations[0]

    # Create input for model
    model_input = model_helper.table_to_floats(data_user, data_weatherreport,
                                                data_experience, data_location)

    # Setup model and create prediction
    predictor_fn = predictor.from_saved_model(data_user['model']['model_pb_path_available'])

    predictor_input = tf.train.Example(
                                features=tf.train.Features(
                                feature={"inputs": tf.train.Feature(
                                float_list=tf.train.FloatList(value=model_input))}))

    predictor_string = predictor_input.SerializeToString()

    prediction = predictor_fn({"inputs": [predictor_string]})
    print('Prediction result: ', prediction)

    # Convert prediction ndarray to dict
    prediction_json = prediction_to_dict(prediction)

    return {"message": "Model inferrence executed successfully",
            "prediction": prediction_json,
            "event": event}


def prediction_to_dict(prediction):
    """Takes a prediction ndarray and converts to a list of dictionary results.
        Classes are converted back to string representations.
        result = [{<class1>: <score1>, <class2>: <score2>, ...}, ...]"""

    result = []

    try:
        for rind in range(len(prediction['classes'])):
            item = {}
            for cind in range(len(prediction['classes'][rind])):
                cls_str = model_helper.key_comfort_level_result(int(prediction['classes'][rind][cind]))
                item[cls_str] = float(prediction['scores'][rind][cind])

            result.append(item)

        return result

    except:
        print('prediction_to_dict failed to convert prediction')
        return 'ERROR'









#
