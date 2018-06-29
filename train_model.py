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
            "S3OutputPath": "s3://{}/{}/model/{}/".format(bucket, bucket_prefix, user_id)
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
    smcli.get_waiter('training_job_completed_or_stopped').wait(TrainingJobName=linear_job)
    if status == 'Failed':
        message = smcli.describe_training_job(TrainingJobName=linear_job)['FailureReason']
        print('Training failed with the following error: {}'.format(message))
        raise Exception('Training job failed')


    return {"message": "Train function executed successfully",
            "event": event}



#
