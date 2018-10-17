try:
    import unzip_requirements
except ImportError:
    pass

import os
import datetime
import io
import decimal
import random
import math
import json

import pandas as pd
# import numpy as np
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
        function_prefix = 'wibsie-ml3-dev-'
        region = 'us-east-1'
        stage = 'dev'
        bucket = 'wibsie-ml3-sagebucket-' + stage
        file_path = ''
    else:
        bucket = os.environ['SAGE_BUCKET']
        bucket_prefix = os.environ['SAGE_BUCKET_PREFIX']
        function_prefix = os.environ['FUNCTION_PREFIX']
        region = os.environ['REGION']
        stage = os.environ['STAGE']
        file_path = '/tmp/'

    index_id = 'be1f64e0-6c1d-11e8-b0b9-e3202dfd59eb'

    user_id = event['user_id']
    model_data = {'now_epoch': getEpochMs()}

    dynamodb = boto3.resource('dynamodb', region_name=region)
    lambdacli = boto3.client('lambda')

    # Get configuration parameters
    config_stage = stage
    if event.get('config_stage'):
        config_stage = event['config_stage']
        print('Overriding config_stage: ', stage, config_stage)

    config = dynamodb.Table('wibsie-config').query(
                    KeyConditionExpression=Key('stage').eq(config_stage))['Items'][0]
    print('Config: ', config)

    # Read in all table data (numbers are type Decimal) and organize by keys
    ##Users
    table_users = dynamodb.Table('wibsie-users-'+stage)
    data_users_requested = None

    if user_id == 'global':
        response = table_users.scan()
        data_users = response['Items']

        while 'LastEvaluatedKey' in response:
            response = table_users.scan(
                            ExclusiveStartKey=response['LastEvaluatedKey'])
            data_users += response['Items']

    elif config.get('blending') == 'index' and user_id != index_id:
        print('blending user with index')

        response = table_users.query(
                        KeyConditionExpression=Key('id').eq(user_id))
        data_users = response['Items']
        data_users_requested = response['Items'][0]

        response = table_users.query(
                        KeyConditionExpression=Key('id').eq(index_id))
        data_users += response['Items']

    else:
        response = table_users.query(
                        KeyConditionExpression=Key('id').eq(user_id))
        data_users = response['Items']
        data_users_requested = response['Items'][0]

    datakey_users = {}
    for u in data_users:
        datakey_users[u['id']] = u

    # Check to see if should skip uploading data
    if config.get('upload_skip_mins'):
        skip_ms = int(config['upload_skip_mins'])*60*1000
        print('Evaluating skip upload for mins, ms: ', config['upload_skip_mins'], skip_ms)

        if data_users_requested.get('model') and data_users_requested['model'].get('model_created'):
            diff_ms = model_data['now_epoch'] - data_users_requested['model']['model_created']


            if diff_ms <= skip_ms:
                print('Skipping upload per diff_ms: ', diff_ms)
                return {"message": "Skipping upload due to recent model creation: {}".format(diff_ms),
                        "train_file": "",
                        "test_file": "",
                        "event": event}

            else:
                print('Proceeding with upload per diff_ms: ', diff_ms)

        else:
            print('Skipping diff eval due to no prior model timestamp')

    ##Experiences
    table_experiences = dynamodb.Table('wibsie-experiences-'+stage)
    if user_id == 'global':
        response = table_experiences.scan()
        data_experiences = response['Items']

        while 'LastEvaluatedKey' in response:
            response = table_experiences.scan(
                            ExclusiveStartKey=response['LastEvaluatedKey'])
            data_experiences += response['Items']

        model_data['blend_pct'] = 0.0

    elif config.get('blending') == 'index' and user_id != index_id:
        print('blending experience data with index')

        response = table_experiences.query(
                        KeyConditionExpression=Key('user_id').eq(user_id))
        data_experiences = response['Items']

        if 'LastEvaluatedKey' in response:
            print('WARNING: LastEvaluatedKey in user_id experience response')

        response = table_experiences.query(
                        KeyConditionExpression=Key('user_id').eq(index_id))
        data_experiences += response['Items']

        if 'LastEvaluatedKey' in response:
            print('WARNING: LastEvaluatedKey in index_id experience response')

        # Don't know true blend yet...
        # model_data['blend_pct'] = getBlendPct(data_experiences, index_id, user_id)

    else:
        response = table_experiences.query(
                        KeyConditionExpression=Key('user_id').eq(user_id))
        data_experiences = response['Items']

        if 'LastEvaluatedKey' in response:
            print('WARNING: LastEvaluatedKey in user_id experience response')

        model_data['blend_pct'] = 100.0

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

    # Build a join around experiences - fields are trimmed in model*.py during train
    feature_columns = ['age', 'bmi', 'gender', 'lifestyle', 'loc_type',
                        'apparent_temperature', 'cloud_cover', 'humidity',
                        'precip_intensity', 'precip_probability', 'temperature',
                        'wind_gust', 'wind_speed', 'precip_type', 'activity_met',
                        'total_clo']
    label_column = 'comfort_level_result'
    results = []
    userid_map = []
    for e in data_experiences:
        # Join to other data
        user_row = datakey_users.get(e['user_id'])
        if not user_row:
            raise Exception('Did not find user row for: ', e['user_id'])

        userid_map.append(e['user_id'])

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

    # Parse result list if blending
    if config.get('blending') == 'index' and user_id != index_id:
        blend_result = parseResultsForBlending(results=results,
                                                blending_type=config.get('blending_type'),
                                                userid_map=userid_map,
                                                user_id=user_id,
                                                index_id=index_id)

        results = blend_result['results']
        model_data['blend_pct'] = blend_result['blend_pct']

    # Make result first column
    columns = [label_column] + feature_columns

    data = pd.DataFrame(results)
    data = data[columns]

    # Fill all Nones
    # data['precip_type'] = data['precip_type'].fillna(value='')
    data['comfort_level_result'] = data['comfort_level_result'].fillna(value=-1) # may not be needed

    # Remove all rows without a label
    data = data[data['comfort_level_result'] >= 0]

    # Resample data
    min_sample = 0
    if config.get('min_sample'):
        print('Overriding min_sample: ', int(config['min_sample']))
        min_sample = int(config['min_sample'])

    data = resampleData(data=data, min_sample=min_sample)

    # Split data
    split = 0.9
    if config.get('split'):
        print('Overriding split value with config: ', config['split'])
        split = float(config['split'])

    train_list, test_list = generateRandomDataLists(data, split=split)

    data_train = data[train_list]
    # data_val = data[val_list]
    data_test = data[test_list]

    # s3 upload training file
    train_file = 'train.csv'

    data_train.to_csv(path_or_buf=file_path+train_file, index=False)

    train_s3path = os.path.join(bucket_prefix,user_id,'trainingfiles',str(model_data['now_epoch']),train_file)

    boto3.Session().resource('s3').Bucket(bucket).Object(train_s3path).upload_file(file_path+train_file)

    # s3 upload test file
    test_file = 'test.csv'

    data_test.to_csv(path_or_buf=file_path+test_file, index=False)

    test_s3path = os.path.join(bucket_prefix,user_id,'trainingfiles',str(model_data['now_epoch']),test_file)

    boto3.Session().resource('s3').Bucket(bucket).Object(test_s3path).upload_file(file_path+test_file)

    # Update user table with latest data and optionally create model key
    print('model_data before decimal: ', model_data)
    model_data['blend_pct'] = decimal.Decimal(str(model_data['blend_pct']))
    print('model_data after decimal: ', model_data)

    if not datakey_users[user_id].get('model'):
        response = table_users.update_item(
                        Key={'id': user_id},
                        UpdateExpression="""set model=:model""",
                        ExpressionAttributeValues={
                            ':model': {'train_created': model_data['now_epoch'],
                                        'blend_pct': model_data['blend_pct']}
                        },
                        ReturnValues="UPDATED_NEW")

    else:
        response = table_users.update_item(
                        Key={'id': user_id},
                        UpdateExpression="""set model.train_created=:train_created, model.blend_pct=:blend_pct""",
                        ExpressionAttributeValues={
                            ':train_created': model_data['now_epoch'],
                            ':blend_pct': model_data['blend_pct']
                        },
                        ReturnValues="UPDATED_NEW")

    print('Update user succeeded')

    if config.get('train_autorun') == True and not event.get('disable_autorun'):
        print('Auto-running train model per config')
        lambdacli.invoke(
            FunctionName=function_prefix+'train_model',
            InvocationType='Event',
            Payload=json.dumps({'user_id': user_id})
        )

    return {"message": "Experiences uploaded",
            "train_file": train_file,
            "test_file": test_file,
            "event": event,
            "epoch": model_data['now_epoch']}


#####################################################
# Helper functions
#####################################################

def getEpochMs():
    """Helper function to current epoch int in ms"""
    now = datetime.datetime.utcnow()
    epoch = datetime.datetime.utcfromtimestamp(0)
    return int((now-epoch).total_seconds() * 1000.0)


def parseResultsForBlending(results, blending_type, userid_map, user_id, index_id):
    """Determine blending percentage and optionally substitute index results.
    Inputs: results: (list, full blended experiences),
            blending_type: (string),
            userid_map: (list, user_ids for each result entry),
            user_id: (string),
            index_id: (string)"""

    user_len = 0
    index_len = 0
    inds_to_add = [i for i in range(0,len(results))] # index is list index
    results_new = []

    if blending_type == 'substitute':
        print('Executing results substitution')
        inds_to_add = [] # results indicies that will get added

        # Outer loop for results entries
        for io in range(0,len(results)):
            # Check if entry is from the index and there is a result
            if userid_map[io]==index_id and results[io]['comfort_level_result'] >= 0:
                # Iterate over user results to see if it matches the index result
                user_matches = {}
                for ii in range(0,len(results)):
                    if userid_map[ii]==user_id and results[ii]['comfort_level_result'] >= 0:
                        match_score = model_helper.model_float_equivalent(results[ii], results[io])
                        if match_score >= 0:
                            user_matches[ii] = match_score
                            print('Found similar user exp: ', io, ii, match_score)

                if not user_matches:
                    inds_to_add.append(io)
                else:
                    min_ii = min(user_matches, key=user_matches.get)
                    print('Matching lowest score for: ', io, min_ii)
                    inds_to_add.append(min_ii)

            elif userid_map[io]==user_id and results[io]['comfort_level_result'] >= 0:
                inds_to_add.append(io)

    for ind in inds_to_add:
        if results[ind]['comfort_level_result'] >= 0:
            if userid_map[ind]==user_id:
                user_len += 1
            elif userid_map[ind]==index_id:
                index_len += 1

            results_new.append(results[ind])

    blend_pct = float(user_len) / (user_len + index_len)
    return {'results': results_new, 'blend_pct': blend_pct}


def resampleData(data, min_sample):
    """Take original data set and append full copies until the min_sample
    length is reached.  Assumes another function will take care of randomizing"""

    data_copy = data.copy()

    row_num = data.shape[0]
    print('resampleData starting length and min_sample: ', row_num, min_sample)

    loops = math.ceil(min_sample/row_num) - 1

    dflist = [data_copy]
    for i in range(loops):
        print('resampleData starting loop: ', i)
        dflist.append(data_copy)

    data_concat = pd.concat(dflist,axis=0)
    data_concat.reset_index(drop=True, inplace=True)

    return data_concat


def generateRandomDataLists(data, split):
    """np.random.rand is imperfect in how random the data is for important values,
    and also is imprecise in how large the lists are (especially for short lists).
    generateRandomDataLists takes the final dataframe and precisely randomizes.
    Returns bool list for train and test."""

    # Create normalized random list of indicies
    row_num = data.shape[0]
    ind_list = [i for i in range(row_num)]
    random.shuffle(ind_list) # modifies in place

    # Split indicies
    ind_list_train = ind_list[:int(row_num*split)]
    ind_list_test = ind_list[int(row_num*split):]
    print('ind_list len: ', len(ind_list_train), len(ind_list_test))

    if (len(ind_list_train)+len(ind_list_test)) != row_num:
        raise Exception('generateRandomDataLists ind_lists lengths are not valid')

    # Create new column to sort
    data_sorted = data.copy()
    data_sorted['ind_data'] = data_sorted.index
    data_sorted['sortkey'] = -1
    for index, row in data_sorted.iterrows():
        sortval = row['temperature'] + row['humidity']*50 + row['activity_met']*10 + row['total_clo']*100
        data_sorted.set_value(index,'sortkey',sortval)

    # Create sorted data
    data_sorted.sort_values(['sortkey'], ascending=True, inplace=True)
    sorted_inds = data_sorted['ind_data'].tolist()

    # Generate final bool lists
    bool_list_train = []
    bool_list_test = []

    for index, row in data.iterrows():
        if sorted_inds.index(index) in ind_list_train:
            bool_list_train.append(True)
            bool_list_test.append(False)

        elif sorted_inds.index(index) in ind_list_test:
            bool_list_train.append(False)
            bool_list_test.append(True)

        else:
            raise Exception('generateRandomDataLists has unmatched index')

    return bool_list_train, bool_list_test







#
