import datetime

import pandas as pd


#################################################################################
# Get label string as integer
#################################################################################

def hash_comfort_level_result(comfort_level_result):
    if type(comfort_level_result) not in [str]:
        raise Exception('hash_comfort_level_result received unexpected type')

    if comfort_level_result.lower() == 'comfortable':
        return 1
    else:
        return 0


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
    elif upper_clothing.lower() == 'short_sleeves':
        return 0.2
    elif upper_clothing.lower() == 'long_sleeves':
        return 0.4
    elif upper_clothing.lower() == 'jacket':
        return 0.6
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

def hash_comfort_level_result(comfort_level_result):
    if type(comfort_level_result) not in [str]:
        raise Exception('hash_comfort_level_result received unexpected type')

    if comfort_level_result.lower() == 'comfortable':
        return 1
    else:
        return 0


def hash_age(age):
    if type(age) not in [int, float]:
        raise Exception('hash_age received unexpected type')

    age_buckets = [0,18, 25, 30, 35, 40, 45, 50, 55, 60, 65,150]

    age_interval = pd.IntervalIndex.from_breaks(age_buckets)

    return float(age_interval.get_loc(age))


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






#
