import datetime
from distutils.version import StrictVersion as ver


#################################################################################
# Convert table definitions to float list
#################################################################################

def table_to_floats(data_user, data_weatherreport, data_experience, data_location, overrides):
    """Must match the following order:
        feature_columns = ['age', 'bmi', 'gender', 'lifestyle', 'loc_type',
                            'apparent_temperature', 'cloud_cover', 'humidity_temp',
                            'precip_intensity', 'precip_probability', 'temperature',
                            'wind_burst', 'wind_speed', 'precip_type', 'activity_met',
                            'total_clo']
    """

    # Process overrides
    windGust = float(data_weatherreport['windGust'])
    if 'windGust' in overrides:
        print('Processing windGust overrides:', overrides['windGust'])
        if overrides['windGust'] == 'windBurst':
            windGust = float(data_weatherreport['windGust'])-float(data_weatherreport['windSpeed'])
        elif overrides['windGust'] == 'null':
            windGust = float(1.0)

    humidity = float(data_weatherreport['humidity'])
    if 'humidity' in overrides:
        print('Processing humidity overrides:', overrides['humidity'])
        if overrides['humidity'] == 'temp_multiply':
            humidity = float(data_weatherreport['humidity']) * (float(data_weatherreport['apparentTemperature']) / 100)
        elif overrides['humidity'] == 'temp_add':
            humidity = float(data_weatherreport['humidity']) + (float(data_weatherreport['apparentTemperature']) / 100)


    return [hash_age(birth_year_to_age(int(data_user['birth_year']))),
            float(data_user['bmi']),
            hash_gender(data_user['gender']),
            hash_lifestyle(data_user['lifestyle']),
            hash_loc_type(data_location['loc_type']),
            float(data_weatherreport['apparentTemperature']),
            float(data_weatherreport['cloudCover']),
            humidity,
            float(data_weatherreport['precipIntensity']),
            float(data_weatherreport['precipProbability']),
            float(data_weatherreport['temperature']),
            windGust,
            float(data_weatherreport['windSpeed']),
            hash_precip_type(data_weatherreport.get('precipType')),
            activity_to_met(data_experience['activity']),
            upper_clothing_to_clo(data_experience['upper_clothing'])+lower_clothing_to_clo(data_experience['lower_clothing'])
            ]


def table_to_floats_nouser(data_weatherreport, data_experience, data_location, overrides={}):
    """Must match the following order:
        feature_columns = [loc_type',
                            'apparent_temperature', 'cloud_cover', 'humidity_temp',
                            'precip_intensity', 'precip_probability', 'temperature',
                            'wind_burst', 'wind_speed', 'precip_type', 'activity_met',
                            'total_clo']
    """

    # Process overrides
    windGust = float(data_weatherreport['windGust'])
    if 'windGust' in overrides:
        print('Processing windGust overrides:', overrides['windGust'])
        if overrides['windGust'] == 'windBurst':
            windGust = float(data_weatherreport['windGust'])-float(data_weatherreport['windSpeed'])
        elif overrides['windGust'] == 'null':
            windGust = float(1.0)

    humidity = float(data_weatherreport['humidity'])
    if 'humidity' in overrides:
        print('Processing humidity overrides:', overrides['humidity'])
        if overrides['humidity'] == 'temp_multiply':
            humidity = float(data_weatherreport['humidity']) * (float(data_weatherreport['apparentTemperature']) / 100)
        elif overrides['humidity'] == 'temp_add':
            humidity = float(data_weatherreport['humidity']) + (float(data_weatherreport['apparentTemperature']) / 100)


    return [float(data_weatherreport['apparentTemperature']),
            float(data_weatherreport['cloudCover']),
            humidity,
            float(data_weatherreport['precipIntensity']),
            float(data_weatherreport['precipProbability']),
            float(data_weatherreport['temperature']),
            windGust,
            float(data_weatherreport['windSpeed']),
            hash_precip_type(data_weatherreport.get('precipType')),
            activity_to_met(data_experience['activity']),
            upper_clothing_to_clo(data_experience['upper_clothing'])+lower_clothing_to_clo(data_experience['lower_clothing'])
            ]


#################################################################################
# Get label string as integer
#################################################################################

def hash_comfort_level_result(comfort_level_result):
    if not comfort_level_result or comfort_level_result.lower() == 'none':
        return -1

    if type(comfort_level_result) not in [str]:
        raise Exception('hash_comfort_level_result received unexpected type')

    if comfort_level_result.lower() == 'uncomfortable_cold':
        return 0
    elif comfort_level_result.lower() == 'comfortable':
        return 1
    elif comfort_level_result.lower() == 'uncomfortable_warm':
        return 2
    else:
        raise Exception('hash_comfort_level_result received unexpected value')


def key_comfort_level_result(comfort_level_result, schema_obj):
    if type(comfort_level_result) not in [int]:
        raise Exception('key_comfort_level_result received unexpected type')

    if comfort_level_result == 2:
        if schema_obj > '1.0':
            return 'uncomfortable_warm'
        else:
            return 'uncomfortable'
    elif comfort_level_result == 1:
        return 'comfortable'
    elif comfort_level_result == 0:
        if schema_obj > '1.0':
            return 'uncomfortable_cold'
        else:
            return 'uncomfortable'
    else:
        raise Exception('key_comfort_level_result received unexpected comfort_level_result')


#################################################################################
# Convert experience data strings to continuous representations
#################################################################################

def activity_to_met(activity):
    if type(activity) not in [str]:
        raise Exception('activity_to_met received unexpected type')

    if activity.lower() == 'standing':
        return 1.1
    elif activity.lower() == 'walking':
        return 2.5
    elif activity.lower() == 'exercising':
        return 6.0
    else:
        raise Exception('Unrecognized activity: ', activity)


def upper_clothing_to_clo(upper_clothing):
    if type(upper_clothing) not in [str]:
        raise Exception('upper_clothing_to_clo received unexpected type')

    # Convert clothing to clo
    if upper_clothing.lower() == 'no_shirt':
        return 0.0
    elif upper_clothing.lower() == 'tank':
        return 0.1
    elif upper_clothing.lower() == 'short_sleeves':
        return 0.2
    elif upper_clothing.lower() == 'long_sleeves':
        return 0.4
    elif upper_clothing.lower() in ['jacket','light_jacket']:
        return 0.6
    elif upper_clothing.lower() == 'heavy_jacket':
        return 0.8
    else:
        raise Exception('Unrecognized upper clothing: ', upper_clothing)


def lower_clothing_to_clo(lower_clothing):
    if type(lower_clothing) not in [str]:
        raise Exception('lower_clothing_to_clo received unexpected type')

    if lower_clothing.lower() == 'shorts':
        return 0.2
    elif lower_clothing.lower() == 'pants':
        return 0.4
    else:
        raise Exception('Unrecognized lower clothing: ', lower_clothing)


def birth_year_to_age(birth_year):
    if type(birth_year) not in [int]:
        raise Exception('birth_year_to_age received unexpected type')

    return datetime.datetime.now().year - birth_year


#################################################################################
# Get categorical features as floats (throws exception if not found, or wrong type)
#################################################################################

def hash_age(age):
    """Create buckets without dependencies on external libraries, e.g pandas"""
    if type(age) not in [int, float]:
        raise Exception('hash_age received unexpected type')

    age_buckets = [0,18, 25, 30, 35, 40, 45, 50, 55, 60, 65,150]

    for i in range(0, len(age_buckets)-1):
        if age >= age_buckets[i] and age < age_buckets[i+1]:
            break

    return float(i)


def hash_gender(gender):
    if type(gender) not in [str]:
        raise Exception('hash_gender received unexpected type')

    genders = ['male', 'female']

    return float(genders.index(gender.lower())) # Throws exception if not found


def hash_lifestyle(lifestyle):
    if type(lifestyle) not in [str]:
        raise Exception('hash_lifestyle received unexpected type')

    lifestyles = ['sedentary', 'moderate_activity', 'high_activity']

    return float(lifestyles.index(lifestyle.lower())) # Throws exception if not found


def hash_loc_type(loc_type):
    if type(loc_type) not in [str]:
        raise Exception('hash_loc_type received unexpected type')

    loc_types = ['rural', 'suburban', 'urban', 'ultra_urban']

    return float(loc_types.index(loc_type.lower())) # Throws exception if not found


def hash_precip_type(precip_type):
    if not precip_type:
        precip_type = ''

    if type(precip_type) not in [str]:
        raise Exception('hash_precip_type received unexpected type')

    precip_types = ['', 'rain', 'snow', 'sleet'] # From darksky api docs

    return float(precip_types.index(precip_type.lower())) # Throws exception if not found


#################################################################################
# Model utilities
#################################################################################

def model_float_equivalent(resultA, resultB):
    """Determine if results row converted to floats are equivalent.
    Returns a score with -1 if match failed, otherwise closer to zero is better.
    Expects result to be a dictionary with correct columns as keys."""

    fail_list = []

    score = 0.0

    humidty_score = float(abs(resultA['humidity_temp']-resultB['humidity_temp'])) / 0.2
    score += humidty_score
    if humidty_score > 1:
        fail_list.append('humidity_temp')

    # if abs(resultA['loc_type']-resultB['loc_type']) > 0:
    #     fail_list.append('loc_type')
    #
    # if abs(resultA['cloud_cover']-resultB['cloud_cover']) > 0.25:
    #     fail_list.append('cloud_cover')

    precip_score = abs(int(resultA['precip_probability'])-int(resultB['precip_probability']))
    score += precip_score
    if precip_score > 0:
        fail_list.append('precip_probability')

    # elif abs(resultA['precip_type']-resultB['precip_type']) > 0:
    #     fail_list.append('precip_type')

    temp_score = float(abs(resultA['temperature']-resultB['temperature'])) / 10
    score += temp_score
    if temp_score > 1:
        fail_list.append('temperature')

    # elif abs(resultA['wind_speed']-resultB['wind_speed']) > 4:
    #     fail_list.append('wind_speed')

    activity_score = abs(resultA['activity_met']-resultB['activity_met'])
    score += activity_score
    if activity_score > 1:
        fail_list.append('activity_met')

    clo_score = abs(resultA['total_clo']-resultB['total_clo'])
    score += clo_score
    if clo_score> 0:
        fail_list.append('total_clo')


    if len(fail_list) == 0:
        return score
    else:
        return -1




#
