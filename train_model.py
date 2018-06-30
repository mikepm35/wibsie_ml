try:
    import unzip_requirements
except ImportError:
    pass

import os
import datetime
import io
import decimal
import time
import json

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
    """Updates training for a given user based on latest data upload"""

    # Read in relevant environment variables, and allow for local run
    if event.get('runlocal'):
        print('Running local and using environment variable placeholders')
        bucket = 'wibsie-ml3-sagebucket-dev'
        bucket_prefix = 'sagemaker/trainingfiles'
        region = 'us-east-1'
        stage = 'dev'
        role = 'arn:aws:iam::530583866435:role/service-role/AmazonSageMaker-ExecutionRole-20180616T150039'
    else:
        bucket = os.environ['SAGE_BUCKET']
        bucket_prefix = os.environ['SAGE_BUCKET_PREFIX']
        region = os.environ['REGION']
        stage = os.environ['STAGE']
        service = os.environ['SERVICE']
        function_prefix = os.environ['FUNCTION_PREFIX']
        role = sm.get_execution_role()

    now_epoch = round(time.time()*1000)
    user_id = event['user_id']

    dynamodb = boto3.resource('dynamodb', region_name=region)

    # Read model data from user
    table_users = dynamodb.Table('wibsie-users-'+stage)
    response = table_users.query(
                    KeyConditionExpression=Key('id').eq(user_id))
    data_user = response['Items'][0]

    if not data_user.get('model') or not data_user['model'].get('train_file') \
    or not data_user['model'].get('validation_file'):
        return {"message": "No model data to train, exiting",
                "event": event}

    # Attempt to get existing job
    job_prev = 'none'
    endpoint_config_prev = 'none'
    endpoint_prev = 'none'
    if data_user['model'].get('job') and data_user['model'].get('endpoint_config') \
    and data_user['model'].get('endpoint'):
        job_prev = data_user['model']['job']
        endpoint_config_prev = data_user['model']['endpoint_config']
        endpoint_prev = data_user['model']['endpoint']
        created_prev = data_user['model']['created']

    # Settup training parameters
    linear_job = user_id + '-linearmodel-' + str(round(time.time()))

    print("Job name is:", linear_job)

    linear_training_params = {
        "RoleArn": role,
        "TrainingJobName": linear_job,
        "AlgorithmSpecification": {
            "TrainingImage": LINEAR_CONTAINERS[region],
            "TrainingInputMode": "File"
        },
        "ResourceConfig": {
            "InstanceCount": 1,
            "InstanceType": "ml.m4.xlarge",
            "VolumeSizeInGB": 10
        },
        "InputDataConfig": [
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": "s3://{}/{}/train/{}".format(bucket, bucket_prefix, data_user['model']['train_file']),
                        "S3DataDistributionType": "ShardedByS3Key"
                    }
                },
                "CompressionType": "None",
                "RecordWrapperType": "None"
            },
            {
                "ChannelName": "validation",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": "s3://{}/{}/validation/{}".format(bucket, bucket_prefix, data_user['model']['validation_file']),
                        "S3DataDistributionType": "FullyReplicated"
                    }
                },
                "CompressionType": "None",
                "RecordWrapperType": "None"
            }

        ],
        "OutputDataConfig": {
            "S3OutputPath": "s3://{}/{}/model/".format(bucket, bucket_prefix)
        },
        "HyperParameters": {
            "feature_dim": "16",
            "mini_batch_size": "5",
            "predictor_type": "binary_classifier",
            "epochs": "10",
            "num_models": "auto",
            "loss": "auto"
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": 60 * 60
        }
    }

    # Start training
    smcli = boto3.client('sagemaker', region_name=region)

    smcli.create_training_job(**linear_training_params)

    status = smcli.describe_training_job(TrainingJobName=linear_job)['TrainingJobStatus']
    print(status)

    # Wait for response
    smcli.get_waiter('training_job_completed_or_stopped').wait(TrainingJobName=linear_job)
    if status == 'Failed':
        message = smcli.describe_training_job(TrainingJobName=linear_job)['FailureReason']
        print('Training failed with the following error: {}'.format(message))
        raise Exception('Training job failed')

    # Setup hosting container
    linear_hosting_container = {
        'Image': LINEAR_CONTAINERS[region],
        'ModelDataUrl': smcli.describe_training_job(TrainingJobName=linear_job)['ModelArtifacts']['S3ModelArtifacts']
    }

    create_model_response = smcli.create_model(
        ModelName=linear_job,
        ExecutionRoleArn=role,
        PrimaryContainer=linear_hosting_container)

    print(create_model_response['ModelArn'])

    # Configure endpoint
    linear_endpoint_config = user_id + '-linearendpoint-config-' + str(round(time.time()))
    print(linear_endpoint_config)

    create_endpoint_config_response = smcli.create_endpoint_config(
        EndpointConfigName=linear_endpoint_config,
        ProductionVariants=[{
            'InstanceType': 'ml.m4.xlarge',
            'InitialInstanceCount': 1,
            'ModelName': linear_job,
            'VariantName': 'AllTraffic'}])

    print("Endpoint Config Arn: " + create_endpoint_config_response['EndpointConfigArn'])

    # Create endpoint
    linear_endpoint = user_id + '-linearendpoint-' + str(round(time.time()))
    print(linear_endpoint)

    create_endpoint_response = smcli.create_endpoint(
        EndpointName=linear_endpoint,
        EndpointConfigName=linear_endpoint_config)
    print(create_endpoint_response['EndpointArn'])

    resp = smcli.describe_endpoint(EndpointName=linear_endpoint)
    status = resp['EndpointStatus']
    print("Status: " + status)

    smcli.get_waiter('endpoint_in_service').wait(EndpointName=linear_endpoint)

    resp = smcli.describe_endpoint(EndpointName=linear_endpoint)
    status = resp['EndpointStatus']
    print("Arn: " + resp['EndpointArn'])
    print("Status: " + status)

    if status != 'InService':
        raise Exception('Endpoint creation did not succeed')

    # Update user table with latest data
    response = table_users.update_item(
                    Key={'id': user_id},
                    UpdateExpression="""set model.job=:job,
                                            model.endpoint_config=:endpoint_config,
                                            model.endpoint=:endpoint,
                                            model.train_file=:train_file,
                                            model.created=:created,
                                            model.job_prev=:job_prev,
                                            model.endpoint_config_prev=:endpoint_config_prev,
                                            model.endpoint_prev=:endpoint_prev,
                                            model.created_prev=:created_prev""",
                    ExpressionAttributeValues={
                        ':job': linear_job,
                        ':endpoint_config': linear_endpoint_config,
                        ':endpoint': linear_endpoint,
                        ':train_file': data_user['model']['train_file'],
                        ':created': now_epoch,
                        ':job_prev': job_prev,
                        ':endpoint_config_prev': endpoint_config_prev,
                        ':endpoint_prev': endpoint_prev,
                        ':created_prev': created_prev
                    },
                    ReturnValues="UPDATED_NEW")

    print('Update user succeeded')

    return {"message": "Train function executed successfully",
            "event": event}



#
