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

import boto3
from boto3.dynamodb.conditions import Key, Attr
from sagemaker.tensorflow import TensorFlow
import sagemaker as sm
# import sagemaker.amazon.common as smac


def train_tf(event, context):
    # Read in relevant environment variables, and allow for local run
    if event.get('runlocal'):
        print('Running local and using environment variable placeholders')
        bucket = 'wibsie-ml3-sagebucket-dev'
        bucket_prefix = 'sagemaker'
        region = 'us-east-1'
        stage = 'dev'
        role = 'arn:aws:iam::530583866435:role/service-role/AmazonSageMaker-ExecutionRole-20180616T150039'
        filepath = '/Users/mmorit202/repos/wibsie_ml_lambda3/tf_funcs/'
    else:
        print('Running using importing environments')
        bucket = os.environ['SAGE_BUCKET']
        bucket_prefix = os.environ['SAGE_BUCKET_PREFIX']
        region = os.environ['REGION']
        stage = os.environ['STAGE']
        service = os.environ['SERVICE']
        function_prefix = os.environ['FUNCTION_PREFIX']
        role = os.environ['SAGE_ROLE']
        filepath = ''
        print('SM execution role: ', sm.get_execution_role())

    if event.get('warm_only'):
        print('Warming only, exiting')
        return {"message": "Train function exiting for warm only",
                "event": event}

    print('Starting train_tf: ', role)

    dynamodb = boto3.resource('dynamodb', region_name=region)

    # Get configuration parameters
    config_stage = stage
    if event.get('config_stage'):
        config_stage = event['config_stage']
        print('Overriding config_stage: ', stage, config_stage)

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

    # Bucket location where generated custom TF code will be uploadeds
    custom_code_upload_location = 's3://'+user_bucket+'/tfcode/'+str(now_epoch)+'/'
    print('custom_code_upload_location: ', custom_code_upload_location)

    # Bucket location where results of model training are saved
    model_artifacts_location = 's3://'+user_bucket+'/models/'+str(now_epoch)+'/'
    print('model_artifacts_location: ', model_artifacts_location)

    # Bucket location where training files are
    model_trainfiles_location = 's3://'+user_bucket+'/trainingfiles/'+str(data_user['model']['train_created'])+'/'
    print('model_trainfiles_location: ', model_trainfiles_location)

    # Create estimator
    job_name = user_id + '-' + stage + '-job-' + str(now_epoch)

    training_steps = 50
    if config.get('training_steps'):
        print('Overriding training_steps: ', training_steps, config['training_steps'])
        training_steps = int(config['training_steps'])

    evaluation_steps = 10
    if config.get('evaluation_steps'):
        print('Overriding evaluation_steps: ', evaluation_steps, config['evaluation_steps'])
        evaluation_steps = int(config['evaluation_steps'])

    entry_point = filepath + 'model.py'
    if config.get('model_type') and config['model_type'] == 'no_user':
        print('Overriding entry_point for no_user')
        entry_point = filepath + 'model_nouser.py'

    tf_estimator = TensorFlow(entry_point=entry_point,
                                role=role,
                                output_path=model_artifacts_location,
                                code_location=custom_code_upload_location,
                                train_instance_count=1,
                                train_instance_type='ml.c4.xlarge',
                                training_steps=training_steps,
                                evaluation_steps=evaluation_steps)

    tf_estimator.fit(inputs=model_trainfiles_location,
                        wait=False,
                        job_name=job_name)

    print('Finished tf_estimator fit call, not waiting form completion: ', job_name)

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

    # Update user with model information
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
                            ':model_job_name': job_name,
                            ':model_completed': 'false',
                            ':model_created_prev': model_created_prev,
                            ':model_train_created_prev': model_train_created_prev,
                            ':model_job_name_prev': model_job_name_prev,
                            ':model_completed_prev': model_completed_prev
                        },
                        ReturnValues="UPDATED_NEW")
    else:
        print('Not saving user with model information')

    # Deploy to sagemaker
    if config.get('deploy_sagemaker'):
        print('Deploying to sagemaker per config: ', job_name)
        tf_predictor = tf_estimator.deploy(initial_instance_count=1,
                                           instance_type='ml.m4.xlarge')
        print('Finished tf_predictor deploy')

    return {"message": "Train function executed successfully",
            "event": event}


def train_comp(event, context):
    """Update user with model completion flag based on s3 event"""

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
