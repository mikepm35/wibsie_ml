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
import copy
from distutils.version import StrictVersion as ver

import boto3
from boto3.dynamodb.conditions import Key, Attr
import botocore
import tensorflow as tf
from tensorflow.contrib import predictor

from common import model_helper


def infer(event, context):
    """Deploy previously uploaded model locally and make a prediction"""

    print('Event: ', event)
    print('Context: ', context)

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

    if event.get('warm_only'):
        print('Warming only, exiting')
        return {"message": "Infer function exiting for warm only",
                "event": event}

    now_epoch = round(time.time()*1000)

    # Parse AWS HTTP request (optional)
    queryParams = None
    if 'body' in event:
        queryParams = event.get('queryStringParameters')
        event = json.loads(event['body'])

    # Load schema
    schema = None
    schema_obj = None
    if queryParams and queryParams.get('schema'):
        schema = queryParams['schema']
        schema_obj = ver(schema)
        print('Loaded schema version: ', schema, schema_obj)

    dynamodb = boto3.resource('dynamodb', region_name=region)

    # Get configuration parameters
    config_stage = stage
    if event.get('config_stage'):
        config_stage = event['config_stage']
        print('Overriding config_stage: ', stage, config_stage)

    config = dynamodb.Table('wibsie-config').query(
                    KeyConditionExpression=Key('stage').eq(config_stage))['Items'][0]
    print('Config: ', config)

    # Retrieve user info
    user_id = event['user_id']
    experience_created = int(event['experience_created'])
    table_users = dynamodb.Table('wibsie-users-'+stage)

    response = table_users.query(
                    KeyConditionExpression=Key('id').eq(user_id))
    data_users = response['Items']

    if len(data_users) == 0:
        print('No user found')
        return {"statusCode": 500,
                "body": "No user found",
                "event": event}
    else:
        data_user = data_users[0]

    # Determine if user has a model loaded
    user_has_model = False

    if data_user.get('model'):
        if data_user['model'].get('model_created') and \
        data_user['model'].get('model_completed') and \
        data_user['model']['model_completed'] == 'true':
            user_has_model = True

        elif data_user['model'].get('model_created_prev') and \
        data_user['model'].get('model_completed_prev') and \
        data_user['model']['model_completed_prev'] == 'true':
            user_has_model = True

        else:
            print('No completed model found')

    else:
        print('Model key is not loaded for user')

    # Setup user for model
    blend_pct = 0.0
    print('Starting user model parse: ', event.get('user_id_global'), config.get('user_id_global'), schema, user_has_model)

    if event.get('user_id_global'):
        print('Using event user_id: ', event['user_id_global'])
        user_id_global = event['user_id_global']

    elif config.get('user_id_global') and config['user_id_global'] != 'user':
        print('Using config user_id: ', config['user_id_global'])
        user_id_global = config['user_id_global']

    elif config.get('user_id_global') == 'user' and schema and user_has_model:
        print('Setting user_id_global to user_id based on config')
        user_id_global = user_id

        if data_user['model'].get('model_blend_pct'):
            blend_pct = float(data_user['model']['model_blend_pct'])
        else:
            blend_pct = 100.0

    else:
        user_id_global = 'be1f64e0-6c1d-11e8-b0b9-e3202dfd59eb' #'global'
        print('Using default user_id: ', user_id_global, user_has_model)

    user_bucket = os.path.join(bucket,bucket_prefix,user_id_global)

    # Retrieve model user info
    response = table_users.query(
                    KeyConditionExpression=Key('id').eq(user_id_global))
    data_user_global = response['Items'][0]

    # Check user model details for actual load
    model_valid = False
    model_keys_expected = ['model_created', 'model_job_name',
                            'model_created_prev', 'model_job_name_prev']

    if data_user_global.get('model'):
        model_keys = data_user_global['model'].keys()

        for k in model_keys_expected:
            if k not in model_keys:
                break

        # Convert created decimal to int
        if type(data_user_global['model']['model_created']) == Decimal:
            data_user_global['model']['model_created'] = int(data_user_global['model']['model_created'])

        if type(data_user_global['model']['model_created_prev']) == Decimal:
            data_user_global['model']['model_created_prev'] = int(data_user_global['model']['model_created_prev'])

        model_valid = True

    if not model_valid:
        print('Valid model details not found', data_user_global)
        return {"statusCode": 500,
                "body": "Valid model details not found",
                "event": event}

    data_user_global['model']['model_available'] = False

    suf_list = ['']
    if data_user_global['model']['model_created_prev'] != 'none':
        suf_list.append('_prev')

    for suf in suf_list:
        print('Attempting model suffix: ', suf_list.index(suf))
        model_artifacts_location = os.path.join(bucket_prefix,user_id_global,'models',str(data_user_global['model']['model_created'+suf]),data_user_global['model']['model_job_name'+suf],'output')
        model_prefix = 'model_' + user_id_global + '_' + str(data_user_global['model']['model_created'+suf])
        local_file = model_prefix + '.tar.gz'
        local_file_path = file_path + local_file
        extract_path = file_path + model_prefix

        # Only download and extract if data doesn't already exist
        if not os.path.exists(extract_path):
            print('Downloading and extracting data: ', model_artifacts_location, local_file_path, extract_path)
            try:
                boto3.Session().resource('s3').Bucket(bucket).download_file(model_artifacts_location+'/model.tar.gz',local_file_path)
                tarfile.open(local_file_path, 'r').extractall(extract_path)
                data_user_global['model']['model_available'] = True
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    print("Model zip file does not exist: ", e)
                else:
                    print("Model zip file download threw unexpected error: ", e)
                    raise
        else:
            print('Using locally available model')
            data_user_global['model']['model_available'] = True

        if data_user_global['model']['model_available']:
            print('Using model suffix: ', suf_list.index(suf))
            data_user_global['model']['model_created_available'] = data_user_global['model']['model_created'+suf]
            data_user_global['model']['model_job_name_available'] = data_user_global['model']['model_job_name'+suf]
            data_user_global['model']['model_extract_path_available'] = extract_path
            break

    if not data_user_global['model']['model_available']:
        print('No model could be resolved')
        return {"statusCode": 500,
                "body": "No model could be resolved",
                "event": event}

    # Get long path to extracted pb file
    data_user_global['model']['model_pb_path_available'] = None
    for root, dirs, files in os.walk(data_user_global['model']['model_extract_path_available']):
        for file in files:
            if file.endswith('.pb'):
                data_user_global['model']['model_pb_path_available'] = root
                break

    if not data_user_global['model']['model_pb_path_available']:
        print('Model pb path could not be resolved')
        return {"statusCode": 500,
                "body": "Model pb path could not be resolved",
                "event": event}

    # Retrieve experience data
    table_experiences = dynamodb.Table('wibsie-experiences-'+stage)

    response = table_experiences.query(
                    KeyConditionExpression=Key('created').eq(experience_created) & Key('user_id').eq(user_id)
                    )
    data_experiences = response['Items']

    if len(data_experiences) == 0:
        print('No experiences found')
        return {"statusCode": 500,
                "body": "No experiences found",
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
        print('No weather report found')
        return {"statusCode": 500,
                "body": "No weather report found",
                "event": event}
    else:
        data_weatherreport = data_weatherreports[0]


    # Get location loop
    infer_loc_loops = 2
    if config.get('infer_loc_loops'):
        infer_loc_loops = int(config['infer_loc_loops'])
        print('Overriding infer_loc_loops default: ', infer_loc_loops)

    infer_loc_sleep = 1
    if config.get('infer_loc_sleep'):
        infer_loc_sleep = int(config['infer_loc_sleep'])
        print('Overriding infer_loc_sleep default: ', infer_loc_sleep)

    for i in range(0,infer_loc_loops):
        table_locations = dynamodb.Table('wibsie-locations-'+stage)

        response = table_locations.query(
                        KeyConditionExpression=Key('zip').eq(data_experience['zip'])
                        )
        data_locations = response['Items']

        if len(data_locations) == 0:
            print('No location data found')
            return {"statusCode": 500,
                    "body": "No location data found",
                    "event": event}
        else:
            data_location = data_locations[0]

        if data_location.get('loc_type'):
            break
        else:
            print('loc_type not defined, sleeping and trying again')
            time.sleep(infer_loc_sleep)

    # Create input for model
    model_overrides = {}
    if config.get('model_overrides'):
        print('Found model_overrides:', config['model_overrides'])
        model_overrides = config['model_overrides']

    if config.get('model_type') and config['model_type'] == 'no_user':
        print('Overriding model_type to nouser')
        model_input = model_helper.table_to_floats_nouser(data_weatherreport,
                                                    data_experience, data_location,
                                                    model_overrides)
    else:
        model_input = model_helper.table_to_floats(data_user, data_weatherreport,
                                                    data_experience, data_location,
                                                    model_overrides)

    # Setup model and create prediction
    predictor_fn = predictor.from_saved_model(data_user_global['model']['model_pb_path_available'])

    predictor_input = tf.train.Example(
                                features=tf.train.Features(
                                feature={"inputs": tf.train.Feature(
                                float_list=tf.train.FloatList(value=model_input))}))

    predictor_string = predictor_input.SerializeToString()

    prediction = predictor_fn({"inputs": [predictor_string]})
    print('Prediction result: ', prediction)

    # Convert prediction ndarray to dict
    attribute_array = [{'blend': blend_pct}]
    prediction_json = prediction_to_dict(prediction, attribute_array, schema_obj)
    print('Prediction json: ', prediction_json)

    # Adds extended values to prediction result
    prediction_type = None
    if config.get('prediction_type'):
        print('Reading prediction_type from config:', config['prediction_type'])
        prediction_type = config['prediction_type']

    prediction_json_extended = prediction_extended(prediction_json, schema_obj,
                                                    prediction_type)

    print('Prediction json extended: ', prediction_json_extended)

    # Pull first value and add to experience table
    if len(prediction_json_extended) > 1:
        print('Skipping database storage due to len greater than 1')
    else:
        prediction_json_decimal = prediction_decimal(prediction_json_extended)

        response = table_experiences.update_item(
                        Key={'created': experience_created, 'user_id': user_id},
                        UpdateExpression="""set comfort_level_prediction=:comfort_level_prediction, prediction_result=:prediction_result""",
                        ExpressionAttributeValues={
                            ':comfort_level_prediction': prediction_json_decimal[0]['comfortable'],
                            ':prediction_result': prediction_json_decimal[0]
                        },
                        ReturnValues="UPDATED_NEW")

        print('table_experiences updated result: ', response)


    return {"statusCode": 200, "body": json.dumps(prediction_json_extended)}


#####################################################
# Test functions
#####################################################

def infer_model_direct(schema_str, stage, data, blend_pct=0, model_overrides=None, prediction_type=None):
    """Infer model with directly passing in all required experience/user data.
    Still downloads model to test.
    Expects data in the form of:
        data = {
            'weatherreport': {
                'apparentTemperature': <float>,
                'cloudCover': <float>,
                'humidity': <float>, # converted to humidity_temp
                'precipIntensity': <float>,
                'precipProbability': <float>,
                'temperature': <float>,
                'windGust': <float>,  # converted to burst
                'windSpeed': <float>,
                'precipType': <str or None>
            },
            'experience': {
                'activity': <str>,
                'upper_clothing': <str>,
                'lower_clothing': <str>
            },
            'location': {},
            'user': {
                'user_id': <str>,
                'model_created': <str>,
                'model_job_name': <str>
            }
        }
    """

    # Configuration variables
    bucket = 'wibsie-ml3-sagebucket-'+stage
    bucket_prefix = 'sagemaker'
    region = 'us-east-1'
    file_path = ''
    schema_obj = ver(schema_str)

    # Setup filesystem information
    model_artifacts_location = os.path.join(bucket_prefix,data['user']['user_id'],'models',data['user']['model_created'],data['user']['model_job_name'],'output')
    model_prefix = 'model_' + data['user']['user_id'] + '_' + data['user']['model_created']
    local_file = model_prefix + '.tar.gz'
    local_file_path = file_path + local_file
    extract_path = file_path + model_prefix

    # Only download and extract if data doesn't already exist
    if not os.path.exists(extract_path):
        print('Downloading and extracting data: ', model_artifacts_location, local_file_path, extract_path)
        try:
            boto3.Session().resource('s3').Bucket(bucket).download_file(model_artifacts_location+'/model.tar.gz',local_file_path)
            tarfile.open(local_file_path, 'r').extractall(extract_path)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("Model zip file does not exist: ", e)
            else:
                print("Model zip file download threw unexpected error: ", e)
                raise
    else:
        print('Using locally available model')

    # Modify data to support sun intensity
    data['weatherreport']['raw'] = {'daily': {'data': [{'sunriseTime': data['weatherreport']['sunrise'],
                                                        'sunsetTime': data['weatherreport']['sunset']}]}}

    # Get long path to extracted pb file
    model_pb_path_available = None
    for root, dirs, files in os.walk(extract_path):
        for file in files:
            if file.endswith('.pb'):
                model_pb_path_available = root
                break

    # Convert data to model inputs
    model_input = model_helper.table_to_floats_nouser(data['weatherreport'],
                                                data['experience'], data['location'],
                                                model_overrides)

    # Setup model and create prediction
    predictor_fn = predictor.from_saved_model(model_pb_path_available)

    predictor_input = tf.train.Example(
                                features=tf.train.Features(
                                feature={"inputs": tf.train.Feature(
                                float_list=tf.train.FloatList(value=model_input))}))

    predictor_string = predictor_input.SerializeToString()

    prediction = predictor_fn({"inputs": [predictor_string]})
    print('Prediction result: ', prediction)

    # Convert prediction ndarray to dict
    attribute_array = [{'blend': blend_pct}]
    prediction_json = prediction_to_dict(prediction, attribute_array, schema_obj)
    print('Prediction json: ', prediction_json)

    # Adds extended values to prediction result
    prediction_json_extended = prediction_extended(prediction_json, schema_obj, prediction_type)

    print('Prediction json extended: ', prediction_json_extended)

    return prediction_json_extended


#####################################################
# Helper functions
#####################################################

def prediction_decimal(prediction_json):
    """Convert all numbers in a prediction list(dict()) to decimal for dynamodb"""

    new_result = []

    for rind in range(len(prediction_json)):
        result = copy.deepcopy(prediction_json[rind])
        for key1 in result:
            if type(result[key1]) == float:
                result[key1] = Decimal(str(result[key1]))

            elif type(result[key1]) == dict:
                for key2 in result[key1]:
                    if type(result[key1][key2]) == float:
                        result[key1][key2] = Decimal(str(result[key1][key2]))

        new_result.append(result)

    return new_result


def prediction_extended(prediction_json, schema_obj, prediction_type=None):
    """Takes the list of prediction dicts and adds extended results"""

    # Schema check
    if schema_obj <= '1.0':
        print('Skipping prediction_extended due to too low schema: ', schema_obj)
        return prediction_json

    # Check for all keys (i.e. data uploaded before schema shift)
    for result in prediction_json:
        if 'uncomfortable_cold' not in result or 'uncomfortable_warm' not in result:
            print('Skipping prediction_extended due to missing cold/warm key', result)
            return prediction_json

    # Iterate over results
    for result in prediction_json:
        # Get max key
        max_key = max(result, key=lambda key: result[key] if key!='blend' else -1)
        primary_percent_raw = result[max_key]

        # Pretty key
        primary_result = pretty_comfort_result(max_key);

        # Set primary percent
        primary_percent = 0.5 + (result[max_key]-.333)/.667*0.5

        # Comfort scale ranges -1 to 1, comfort ~ -0.33 to 0.33
        comfort_scale = result['uncomfortable_warm']-result['uncomfortable_cold']

        # Confidence on a scale from 0 to 1
        confidence = (result[max_key]-0.333) / 0.68

        # Process config
        context = 'none'
        if prediction_type and prediction_type.startswith('comfort_scale_'):
            scale_split = float(prediction_type.split('_')[2]) / 100

            print('Processing prediction_type comfort_scale:', prediction_type, scale_split)

            if primary_percent_raw <= scale_split:
                print('Using comfort_scale for primary_result')
                context = 'comfort_scale'
                if comfort_scale < -0.333:
                    primary_result = pretty_comfort_result('uncomfortable_cold')
                elif comfort_scale <= 0.34:
                    primary_result = pretty_comfort_result('comfortable')
                else:
                    primary_result = pretty_comfort_result('uncomfortable_warm')


        result['attributes'] = {
            'primary_result': primary_result,
            'primary_result_raw': max_key,
            'primary_percent': primary_percent,
            'primary_percent_raw': primary_percent_raw,
            'confidence': confidence,
            'comfort_scale': comfort_scale,
            'context': context
        }

    return prediction_json


def prediction_to_dict(prediction, attribute_array, schema_obj):
    """Takes a prediction ndarray and converts to a list of dictionary results.
        Also takes an array of dicts for model attributes that are appeneded to result.
        Classes are converted back to string representations.
        result = [{<class1>: <score1>, <class2>: <score2>, ...}, ...]"""

    result = []

    try:
        for rind in range(len(prediction['classes'])):
            item = attribute_array[rind].copy()
            for cind in range(len(prediction['classes'][rind])):
                cls_str = model_helper.key_comfort_level_result(int(prediction['classes'][rind][cind]), schema_obj)
                if cls_str in item:
                    item[cls_str] += float(prediction['scores'][rind][cind])
                else:
                    item[cls_str] = float(prediction['scores'][rind][cind])

            result.append(item)

        return result

    except:
        print('prediction_to_dict failed to convert prediction')
        return 'ERROR'


def pretty_comfort_result(result):
    """Takes variations of the comfort result and converts to a pretty response"""

    if result == 'comfortable':
        return 'Comfortable'

    elif result == 'uncomfortable':
        return 'Uncomfortable'

    elif result == 'uncomfortable_warm':
        return 'Too Warm'

    elif result == 'uncomfortable_cold':
        return 'Too Cold'

    else:
        print('Unrecognized result')
        return 'Unknown'






#
