try:
    import unzip_requirements
except ImportError:
    pass

import os
import sys
import datetime
import io
import decimal
import time
import json
import functools
import tarfile
import csv

import boto3
from boto3.dynamodb.conditions import Key, Attr
import numpy as np
import tensorflow as tf

from common import model


#tf.enable_eager_execution() # enables real-time output


def _csv_to_dict(fullfilepath, valid_keys):
    """Expects fully qualified path and filename, returns a dictionary where
    each key is a column as a list. valid_keys is a list of strings that
    are used dto restrict what is read into the data_dict."""

    data_dict = {}
    with open(fullfilepath) as fh:
        rd = csv.DictReader(fh, delimiter=',')
        for row in rd:
            if not data_dict:
                for key in row.keys():
                    data_dict[key] = [row[key]]
            else:
                for key in row.keys():
                    data_dict[key].append(row[key])

    return data_dict


def train_tf(event, context):
    """Main training lambda function"""

    # Read in relevant environment variables, and allow for local run
    if event.get('runlocal'):
        print('Running local and using environment variable placeholders')
        bucket = 'wibsie-ml3-sagebucket-dev'
        bucket_prefix = 'sagemaker'
        region = 'us-east-1'
        stage = 'dev'
        filepath = ''

    else:
        print('Running using importing environments')
        bucket = os.environ['SAGE_BUCKET']
        bucket_prefix = os.environ['SAGE_BUCKET_PREFIX']
        region = os.environ['REGION']
        stage = os.environ['STAGE']
        service = os.environ['SERVICE']
        function_prefix = os.environ['FUNCTION_PREFIX']
        filepath = '/tmp/'

    if event.get('warm_only'):
        print('Warming only, exiting')
        return {"message": "Train function exiting for warm only",
                "event": event}


    dynamodb = boto3.resource('dynamodb', region_name=region)


    # Read in config
    config_stage = stage
    if event.get('config_stage'):
        config_stage = event['config_stage']
        print('Overriding stage with config_stage: ', stage, config_stage)

    config = dynamodb.Table('wibsie-config').query(
                    KeyConditionExpression=Key('stage').eq(config_stage))['Items'][0]
    print('Config: ', config)


    # Setup user
    now_epoch = round(time.time()*1000)
    user_id = event['user_id']
    user_bucket = os.path.join(bucket,bucket_prefix,user_id)


    # Read model data from user
    table_users = dynamodb.Table('wibsie-users-'+stage)
    response = table_users.query(
                    KeyConditionExpression=Key('id').eq(user_id))
    data_user = response['Items'][0]

    if not data_user.get('model') or not data_user['model'].get('train_created'):
        return {"message": "No model data to train, exiting",
                "event": event}


    # Download train and test files
    model_trainfiles_s3path = os.path.join(bucket_prefix,user_id,'trainingfiles', str(data_user['model']['train_created']))
    print('model_trainfiles_s3path: ', model_trainfiles_s3path)

    boto3.Session().resource('s3').Bucket(bucket).download_file(model_trainfiles_s3path+'/train.csv',filepath+'train_'+str(now_epoch)+'.csv')
    boto3.Session().resource('s3').Bucket(bucket).download_file(model_trainfiles_s3path+'/test.csv',filepath+'test_'+str(now_epoch)+'.csv')


    # Load train and test files as dicts
    train_dict = _csv_to_dict(filepath+'train_'+str(now_epoch)+'.csv', model.FEATURE_COLS+[model.LABEL_COL])
    test_dict = _csv_to_dict(filepath+'test_'+str(now_epoch)+'.csv', model.FEATURE_COLS+[model.LABEL_COL])


    # Retrieve feature columns
    my_numeric_columns = model.get_feature_columns()


    # Parse any training overrides
    input_config = {
        'epochs_train': 400, 'epochs_test': 400,
        'batches_train': 10,'batches_test': 10
    }

    for item in input_config:
        if config.get(item):
            print('Overriding input_config:', item, input_config[item], config[item])
            input_config[item] = config[item]


    # Setup input functions
    train_inpf = functools.partial(model.easy_input_function, data_dict=train_dict, shuffle=True,
                               label_key=model.LABEL_COL,
                               num_epochs=input_config['epochs_train'],
                               batch_size=input_config['batches_train'])

    test_inpf = functools.partial(model.easy_input_function, data_dict=test_dict, shuffle=True,
                                  label_key=model.LABEL_COL,
                                  num_epochs=input_config['epochs_test'],
                                  batch_size=input_config['batches_test'])


    # Train model w/specified save location
    tf_model = tf.estimator.LinearClassifier(
                feature_columns=my_numeric_columns,
                n_classes=3,
                model_dir=filepath+'model_'+str(now_epoch)+'/'
    )

    tf_model.train(train_inpf)


    # Evaluate model
    model_result = tf_model.evaluate(test_inpf)
    print('Model result:', model_result)


    # Zip up model files and store in s3
    with tarfile.open(filepath+'model.tar.gz', mode='w:gz') as archive:
        archive.add(filepath+'model_'+str(now_epoch)+'/', recursive=True)

    model_artifcats_s3path = os.path.join(bucket_prefix,user_id,'models', str(now_epoch), 'model.tar.gz')
    print('model_artifacts_s3path: ', model_artifcats_s3path)

    boto3.Session().resource('s3').Bucket(bucket).Object(model_artifcats_s3path).upload_file(filepath+'model.tar.gz')


    # Retrieve existing model information
    model_created_prev = 'none'
    if data_user['model'].get('model_created'):
        model_created_prev = data_user['model']['model_created']

    model_train_created_prev = 'none' # Prev record would need to be in upload_data

    model_job_name_prev = 'none'
    if data_user['model'].get('model_job_name'):
        model_job_name_prev = data_user['model']['model_job_name']

    model_blend_pct_prev = 'none' # Prev record would need to be in upload_data

    blend_pct = decimal.Decimal(100.0)
    if data_user['model'].get('blend_pct'):
        blend_pct = data_user['model']['blend_pct']

    model_completed_prev = 'none'
    if data_user['model'].get('model_completed'):
        model_completed_prev = data_user['model']['model_completed']


    # Write properties to dynamodb
    if 'save_model' not in config or config['save_model'] == True:
        print('Updating user with model information')
        response = table_users.update_item(
                        Key={'id': user_id},
                        UpdateExpression="""set model.model_created=:model_created,
                                                model.model_train_created=:model_train_created,
                                                model.model_blend_pct=:model_blend_pct,
                                                model.model_job_name=:model_job_name,
                                                model.model_completed=:model_completed,
                                                model.model_created_prev=:model_created_prev,
                                                model.model_train_created_prev=:model_train_created_prev,
                                                model.model_job_name_prev=:model_job_name_prev,
                                                model.model_completed_prev=:model_completed_prev""",
                        ExpressionAttributeValues={
                            ':model_created': now_epoch,
                            ':model_train_created': data_user['model']['train_created'],
                            ':model_blend_pct': blend_pct,
                            ':model_job_name': 'none',
                            ':model_completed': 'true',
                            ':model_created_prev': model_created_prev,
                            ':model_train_created_prev': model_train_created_prev,
                            ':model_job_name_prev': model_job_name_prev,
                            ':model_completed_prev': model_completed_prev
                        },
                        ReturnValues="UPDATED_NEW")
    else:
        print('Not saving user with model information')


    return {"message": "Train function executed successfully",
            "event": event}


def train_comp(event, context):
    """Update user with model completion flag based on s3 event"""

    print('train_comp not in use, exiting')
    return {"message": "train_comp exiting since not in use",
            "event": event}

    # Read in relevant environment variables, and allow for local run
    if event.get('runlocal'):
        print('Running local and using environment variable placeholders')
        bucket = 'wibsie-ml3-sagebucket-dev'
        bucket_prefix = 'sagemaker'
        region = 'us-east-1'
        stage = 'dev'
    else:
        print('Running using importing environments')
        bucket = os.environ['SAGE_BUCKET']
        bucket_prefix = os.environ['SAGE_BUCKET_PREFIX']
        region = os.environ['REGION']
        stage = os.environ['STAGE']
        service = os.environ['SERVICE']
        function_prefix = os.environ['FUNCTION_PREFIX']

    if event.get('warm_only'):
        print('Warming only, exiting')
        return {"message": "Train_comp function exiting for warm only",
                "event": event}

    print('Starting train_comp: ', event)

    dynamodb = boto3.resource('dynamodb', region_name=region)

    # Get configuration parameters
    config_stage = stage
    if event.get('config_stage'):
        config_stage = event['config_stage']
        print('Overriding config_stage: ', stage, config_stage)

    config = dynamodb.Table('wibsie-config').query(
                    KeyConditionExpression=Key('stage').eq(config_stage))['Items'][0]
    print('Config: ', config)

    # Iterate over records for s3 events
    for record in event['Records']:
        key_split = record['s3']['object']['key'].split('/')
        user_id_event = key_split[1]
        model_created_event = decimal.Decimal(key_split[3])

        # Read model data from user
        table_users = dynamodb.Table('wibsie-users-'+stage)
        response = table_users.query(
                        KeyConditionExpression=Key('id').eq(user_id_event))
        data_user = response['Items'][0]

        # Get existing completed variables as starting points
        model_completed = 'none'
        if data_user['model'].get('model_completed'):
            model_completed = data_user['model']['model_completed']

        model_completed_prev = 'none'
        if data_user['model'].get('model_completed_prev'):
            model_completed_prev = data_user['model']['model_completed_prev']

        # Check if completed should be updated based on event
        update_required = False

        if model_created_event==data_user['model'].get('model_created'):
            update_required = True
            model_completed = 'true'

        elif model_created_event==data_user['model'].get('model_created_prev'):
            update_required = True
            model_completed_prev = 'true'

        else:
            print('WARNING - model_created_event does not match user data: ', user_id_event, model_created_event)

        # Update user
        if update_required:
            print('Updating user based model_completed update: ', model_completed, model_completed_prev)
            response = table_users.update_item(
                            Key={'id': user_id_event},
                            UpdateExpression="""set model.model_completed=:model_completed,
                                                    model.model_completed_prev=:model_completed_prev""",
                            ExpressionAttributeValues={
                                ':model_completed': model_completed,
                                ':model_completed_prev': model_completed_prev
                            },
                            ReturnValues="UPDATED_NEW")

        else:
            print('Not updating user due to no change on model_completeds')


    return {"message": "User successfully updated",
            "event": event}

#
