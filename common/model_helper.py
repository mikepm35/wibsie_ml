import datetime
from distutils.version import StrictVersion as ver
import math


FEATURE_COLS_ALL = [
    'user_id',
    'age',
    'bmi',
    'gender',
    'lifestyle',
    'loc_type',
    'apparent_temperature',
    'cloud_cover',
    'humidity_temp',
    'precip_intensity',
    'precip_probability',
    'temperature',
    'wind_burst',
    'wind_speed',
    'precip_type',
    'activity_met',
    'total_clo',
    'humidity',
    'wind_chill',
    'meanradiant_temp',
    'uvindex',
    'h_intprod',
    'ed_watvapdiff',
    'esw_sweat',
    'ere_resp',
    'l_dryresp',
    'r_radiation',
    'c_convection',
    'pmv',
    'ppd'
]

LABEL_COL = 'comfort_level_result'

INDEX_USERID = 'be1f64e0-6c1d-11e8-b0b9-e3202dfd59eb'


#################################################################################
# Convert table definitions to float list
#################################################################################

def table_to_floats(data_user, data_weatherreport, data_experience, data_location, overrides):
    """Main function for generating ordered float values for csv training"""

    # print(data_weatherreport['raw']['daily']['data'][0]['sunriseTime']) #s
    # print(data_experience['created']) #ms
    # raise Exception()

    # Process overrides
    windGust = float(data_weatherreport['windGust'])
    if 'windGust' in overrides:
        # print('Processing windGust overrides:', overrides['windGust'])
        if overrides['windGust'] == 'windBurst':
            windGust = float(data_weatherreport['windGust'])-float(data_weatherreport['windSpeed'])
        elif overrides['windGust'] == 'null':
            windGust = float(1.0)
        elif overrides['windGust'] == 'windchill':
            windGust = get_windchill(data_weatherreport['temperature'], data_weatherreport['windGust'])
        elif overrides['windGust'] == 'sqrt':
            windGust = math.sqrt(data_weatherreport['windGust']) # returns float
        elif overrides['windGust'] == 'ln':
            windGust = math.log(data_weatherreport['windGust'] + 1)
        elif overrides['windGust'] == 'thirds':
            if 0.0 <= windGust < 10.0:
                windGust = 0.0
            elif 10.0 <= windGust < 22.0:
                windGust = 1.0
            elif windGust >= 22.0:
                windGust = 2.0
        else:
            print('Ignoring windGust override')

    windSpeed = float(data_weatherreport['windSpeed'])
    if 'windSpeed' in overrides:
        # print('Processing windSpeed overrides:', overrides['windSpeed'])
        if overrides['windSpeed'] == 'null':
            windSpeed = float(1.0)
        elif overrides['windSpeed'] == 'windchill':
            windSpeed = get_windchill(float(data_weatherreport['temperature']), float(data_weatherreport['windSpeed']))
        elif overrides['windSpeed'] == 'sqrt':
            windSpeed = math.sqrt(data_weatherreport['windSpeed']) # returns float
        elif overrides['windSpeed'] == 'ln':
            windSpeed = math.log(data_weatherreport['windSpeed'] + 1)
        elif overrides['windSpeed'] == 'humidity_multiply':
            windSpeed = float(data_weatherreport['windSpeed']) * float(data_weatherreport['humidity'])
        elif overrides['windSpeed'] == 'temp_multiply':
            windSpeed = float(data_weatherreport['windSpeed']) * float(data_weatherreport['temperature'])
        elif overrides['windSpeed'] == 'humidity_temp_multiply':
            windSpeed = float(data_weatherreport['windSpeed']) * float(data_weatherreport['humidity']) * float(data_weatherreport['temperature'])
        elif overrides['windSpeed'] == 'quarters':
            if 0.0 <= windSpeed < 5.0:
                windSpeed = 0.0
            elif 5.0 <= windSpeed < 10.0:
                windSpeed = 1.0
            elif 10.0 <= windSpeed < 15.0:
                windSpeed = 2.0
            elif windSpeed >= 15.0:
                windSpeed = 3.0
        elif overrides['windSpeed'] == 'thirds':
            if 0.0 <= windSpeed < 10.0:
                windSpeed = 0.0
            elif 10.0 <= windSpeed < 22.0:
                windSpeed = 1.0
            elif windSpeed >= 22.0:
                windSpeed = 2.0
        else:
            print('Ignoring windSpeed override')

    humidity = float(data_weatherreport['humidity'])
    if 'humidity' in overrides:
        # print('Processing humidity overrides:', overrides['humidity'])
        if overrides['humidity'] == 'apptemp_multiply':
            humidity = float(data_weatherreport['humidity']) * (float(data_weatherreport['apparentTemperature']) / 100)
        elif overrides['humidity'] == 'temp_multiply':
            humidity = float(data_weatherreport['humidity']) * (float(data_weatherreport['temperature']) / 100)
        elif overrides['humidity'] == 'apptemp_add':
            humidity = float(data_weatherreport['humidity']) + (float(data_weatherreport['apparentTemperature']) / 100)
        else:
            print('Ignoring humidity override')

    cloudCover = float(data_weatherreport['cloudCover'])
    if 'cloudCover' in overrides:
        # print('Processing cloudCover overrides:', overrides['cloudCover'])
        if overrides['cloudCover'] == 'null':
            cloudCover = float(1.0)
        elif overrides['cloudCover'] == 'apptemp_multiply':
            cloudCover = float(data_weatherreport['cloudCover']) * (float(data_weatherreport['apparentTemperature']) / 100)
        elif overrides['cloudCover'] == 'temp_multiply':
            cloudCover = float(data_weatherreport['cloudCover']) * (float(data_weatherreport['temperature']) / 100)
        elif overrides['cloudCover'] == 'sunintensity':
            cloudCover = get_sunintensity(
                            float(data_weatherreport['cloudCover']),
                            float(data_experience['created']),
                            data_weatherreport['raw']
                        )
        elif overrides['cloudCover'] == 'quarters':
            if 0.0 <= cloudCover < 0.25:
                cloudCover = 0.0
            elif 0.25 <= cloudCover < 0.5:
                cloudCover = 1.0
            elif 0.5 <= cloudCover < 0.75:
                cloudCover = 2.0
            elif cloudCover >= 0.75:
                cloudCover = 3.0
        elif overrides['cloudCover'] == 'thirds':
            if 0.0 <= cloudCover < 0.333:
                cloudCover = 0.0
            elif 0.333 <= cloudCover < 0.667:
                cloudCover = 1.0
            elif cloudCover >= 0.667:
                cloudCover = 2.0
        else:
            print('Ignoring cloudCover override')

    totalClo = upper_clothing_to_clo(data_experience['upper_clothing'])+lower_clothing_to_clo(data_experience['lower_clothing'])
    if 'totalClo' in overrides:
        # print('Processing totalClo overrides:', overrides['totalClo'])
        if overrides['totalClo'] == 'temp_multiply':
            totalClo = totalClo * float(data_weatherreport['temperature'])
        elif overrides['totalClo'] == 'met_multiply':
            totalClo = totalClo * activity_to_met(data_experience['activity'])
        else:
            print('Ignoring totalClo override')

    meanradiant_temp = get_meanradianttemp(float(data_weatherreport['uvIndex']), float(data_weatherreport['temperature']))

    # get PMV parameters
    pmv_params = getPmvParams({
        'met': activity_to_met(data_experience['activity']),
        'clo': upper_clothing_to_clo(data_experience['upper_clothing'])+lower_clothing_to_clo(data_experience['lower_clothing']),
        'ta_airtmp': float(data_weatherreport['temperature']),
        'tr_radtmp': meanradiant_temp,
        'var_airvel': float(data_weatherreport['windSpeed']),
        'rel_humidity': float(data_weatherreport['humidity'])*100
    })

    # retrieve user data
    list_user = []
    if data_user:
        list_user = [hash_userid(data_user['id']),
                        hash_age(birth_year_to_age(int(data_user['birth_year']))),
                        float(data_user['bmi']),
                        hash_gender(data_user['gender']),
                        hash_lifestyle(data_user['lifestyle'])]

    return list_user + [
                hash_loc_type(data_location['loc_type']),
                float(data_weatherreport['apparentTemperature']),
                cloudCover,
                humidity,
                float(data_weatherreport['precipIntensity']),
                float(data_weatherreport['precipProbability']),
                float(data_weatherreport['temperature']),
                windGust,
                windSpeed,
                hash_precip_type(data_weatherreport.get('precipType')),
                activity_to_met(data_experience['activity']),
                totalClo,
                float(data_weatherreport['humidity']),
                get_windchill(float(data_weatherreport['temperature']), float(data_weatherreport['windSpeed'])),
                meanradiant_temp,
                float(data_weatherreport['uvIndex'])
            ] + [
                pmv_params['h_intprod'],
                pmv_params['ed_watvapdiff'],
                pmv_params['esw_sweat'],
                pmv_params['ere_resp'],
                pmv_params['l_dryresp'],
                pmv_params['r_radiation'],
                pmv_params['c_convection'],
                pmv_params['pmv'],
                pmv_params['ppd']
            ]


def table_to_floats_nouser(data_weatherreport, data_experience, data_location, overrides={}):
    """Must output the following order:
        feature_columns = ['apparent_temperature', 'cloud_cover', 'humidity_temp',
                            'precip_intensity', 'precip_probability', 'temperature',
                            'wind_burst', 'wind_speed', 'precip_type', 'activity_met',
                            'total_clo']
    """

    float_list = table_to_floats(data_user={},
                                data_weatherreport=data_weatherreport,
                                data_experience=data_experience,
                                data_location=data_location,
                                overrides=overrides)

    return float_list[1:] # -1 to remove location


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
        return 3.5
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

def hash_userid(user_id):
    """Sets index user_id to 0, and all others to 1"""

    if type(user_id) not in [str]:
        raise Exception('hash_userid received unexpected type')

    if user_id == INDEX_USERID:
        return float(0)
    else:
        return float(1)


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

def get_meanradianttemp(uvindex, airtemp):
    """Calculate mean radiant temperature in F from an approximation for
    solar radiance calculated via uvindex.  Airtemp provided in F.
    Max Tmrt practically should be ~60C/140F.

    Tmrt = (Sstr/(ep*omega))^0.25 - 273.15 (C)
        Sstr = Mean radiant flux density (W/m2)
        ep = Emissivity of the human, assume const = 0.97 (unitless)
        omega = Stefan-boltzmann constant = 5.67*10^-8 (W*m2*K^-4)

    Pokhrel, Rudra & Bhattarai, Binod. (2011). Relation between Global Solar Radiation
    and Solar Ultraviolet Radiation in Different Parts of Nepal. Journal of Institute of Engineering

    uvindex = 8.6*10^-6*solarradiance^2 - 0.0021*solarradiance + 0.22 (unitless)
        solarradiance = irradiance by pyanometer (W/m2)
    solarradiance = (0.0021 + sqrt{0.0021^2 - 4*8.6*10^-6*(0.22-uvindex)}) / (2*8.6*10^-6)
        Set Sstr=solarradiance
    """

    # If UV index is zero, set equal to air temperature
    if uvindex == 0:
        return airtemp

    # Calculate solar radiance
    solarrad = (0.0021+math.sqrt(0.0021**2-4*8.6*(10**-6)*(0.22-uvindex)))/(2*8.6*(10**-6))
    solarrad = solarrad * 0.55 # empirical fudge factor

    # Calculate mean radiant temperature
    temp_mrt = math.pow(solarrad/(0.97*5.67*(10**-8)),0.25) - 273.15

    return temp_mrt*(9/5)+32 #degF


def get_sunintensity(cloud_cover, exp_created, weather_raw):
    """Create a sunintensity metric by combining cloudCover and current time
    relative to sunrise. Currently extracts sunrise data from raw weather report.
    exp_created comes as epoch time in ms, while sunrise is in seconds."""

    exp_created_s = exp_created / 1000

    # Extract sunrise and sunset time
    sunrise_s = float(weather_raw['daily']['data'][0]['sunriseTime'])
    sunset_s = float(weather_raw['daily']['data'][0]['sunsetTime'])

    # Calculate twilights
    twilight_factor_s = 1.5*3600
    sunrise_pre_s = sunrise_s - twilight_factor_s
    sunrise_post_s = sunrise_s + twilight_factor_s
    sunset_pre_s = sunset_s - twilight_factor_s
    sunset_post_s = sunset_s + twilight_factor_s

    # Get sun factor
    sun_map = {'night': 0.0, 'dawn': 0.85, 'dusk': 0.85, 'day': 1.0}
    if exp_created_s < sunrise_pre_s and (sunrise_pre_s-exp_created_s) < 8*3600:
        sun_factor = sun_map['night']
    elif exp_created_s >= sunrise_pre_s and exp_created_s < sunrise_post_s:
        sun_factor = sun_map['dawn']
    elif exp_created_s >= sunrise_post_s and exp_created_s < sunset_pre_s:
        sun_factor = sun_map['day']
    elif exp_created_s >= sunset_pre_s and exp_created_s < sunset_post_s:
        sun_factor = sun_map['dusk']
    elif exp_created_s >= sunset_post_s and (exp_created_s-sunset_post_s) < 8*3600:
        sun_factor = sun_map['night']
    else:
        raise Exception('Unable to resolve sun_factor', exp_created)

    # print('sunfactor', sun_factor, exp_created)

    # Convert cloud cover to categories
    if 0.0 <= cloud_cover < 0.333:
        cloud_cover = 3.0
    elif 0.333 <= cloud_cover < 0.667:
        cloud_cover = 2.0
    elif cloud_cover >= 0.667:
        cloud_cover = 1.0

    # Return sun intensity
    return cloud_cover * sun_factor


def get_windchill(temp_f, wind_mph):
    """Returns an adjusted temperature in F. Is almost identical to apparentTemperature."""

    return 35.74 + 0.6215*temp_f - 35.75*math.pow(wind_mph,0.16) + 0.4275*temp_f*math.pow(wind_mph,0.16)


def model_float_equivalent(resultA, resultB):
    """Determine if results row converted to floats are equivalent.
    Returns a score with -1 if match failed, otherwise closer to zero is better.
    Expects result to be a dictionary with correct columns as keys."""

    fail_list = []

    score = 0.0

    # humidty_score = float(abs(resultA['humidity_temp']-resultB['humidity_temp'])) / 0.2
    # score += humidty_score
    # if humidty_score > 1:
    #     fail_list.append('humidity_temp')

    humidty_score = float(abs(resultA['humidity']-resultB['humidity'])) / 0.3
    score += humidty_score
    if humidty_score > 1:
        fail_list.append('humidity')

    windspeed_score = float(abs(resultA['wind_speed']-resultB['wind_speed'])) / 0.75
    score += windspeed_score
    if windspeed_score > 1:
        fail_list.append('windspeed_score')

    # if abs(resultA['loc_type']-resultB['loc_type']) > 0:
    #     fail_list.append('loc_type')
    #
    # if abs(resultA['cloud_cover']-resultB['cloud_cover']) > 0.25:
    #     fail_list.append('cloud_cover')

    # precip_score = abs(int(resultA['precip_probability'])-int(resultB['precip_probability']))
    # score += precip_score
    # if precip_score > 0:
    #     fail_list.append('precip_probability')

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


#################################################################################
# ISO 7730 parameters
#################################################################################

def getPmvParams(func_input):
    """
    Expects input parameters as follows:
        met (converted into rate), clo (converted to insulation),
        ta_airtmp (F converted to C), tr_radtmp (F converted to C),
        var_airvel (mph converted to m/s),
        rel_humidity (% converted to partial pressure)

    Returns the constituent parts of the PMV function, and PMV, as a dictionary.

    Input details:
        M is the metabolic rate, in watts per square metre (W/m2);
        W is the effective mechanical power, in watts per square metre (W/m2);
        Icl is the clothing insulation, in square metres kelvin per watt (m2 ⋅ K/W);
        fcl is the clothing surface area factor;
        ta is the air temperature, in degrees Celsius (°C);
        tr is the mean radiant temperature, in degrees Celsius (°C);
        var is the relative air velocity, in metres per second (m/s);
        pa is the water vapour partial pressure, in pascals (Pa);
        hc is the convective heat transfer coefficient, in watts per square metre kelvin [W/(m2 ⋅ K)];
        tcl is the clothing surface temperature, in degrees Celsius (°C).
        Note: 1 metabolic unit = 1 met = 58,2 W/m2; 1 clothing unit = 1 clo = 0,155 m2 ⋅ °C/W

    Output equation details:
        L = H-Ed-Esw-Ere-L-R-C
        H = internal heat production
        Ed = heat loss due to water vapour diffusion through the skin
        Esw = heat loss due to sweating
        Ere = latent heat loss due to respiration
        L = dry respiration heat loss
        R = heat loss by radiation from the surface of the clothed body
        C = heat loss by convection from the surface of the clothed body
        PMV = exp[met]*L
    """

    # Define limits where function is valid
    limits = {
        'm_metrate': [46, 232],
        'w_mechpow': [],
        'icl_cloins': [0, 0.310],
        # 'clo': [0, 2],
        'fcl_closa': [],
        'ta_airtmp': [10, 30],
        'tr_radtmp': [10, 40],
        'var_airvel': [0, 1],
        'pa_wtrpres': [0, 2700],
        'hc_convht': [],
        'pmv': [-2, 2]
    }

    # Inputs
    inputs = {
        'm_metrate': 58.2 * func_input['met'], #M, W/m^2
        'w_mechpow': 58.2 * 0.001, #W, W/m^2
        'icl_cloins': 0.155 * func_input['clo'], #Pa, m^2*K/W
        'fcl_closa': None, #fcl, unitless
        'ta_airtmp': (func_input['ta_airtmp'] - 32) * 0.5556, #ta, C
        'tr_radtmp': (func_input['tr_radtmp'] - 32) * 0.5556, #tr, C
        'var_airvel': func_input['var_airvel'] * 0.44704, #var, m/s
        'pa_wtrpres': None, #pa, Pa
        'hc_convht': None, #hc, W/(m^2*K)
        'tcl_clotmp': None #tcl, C
    }

    # Calculate partial pressure water vapor from RH
    inputs['pa_wtrpres'] = func_input['rel_humidity'] * 10 * math.exp(16.6536 - 4030.183 / (inputs['ta_airtmp'] + 235))

    # Calculate fcl
    if inputs['icl_cloins'] <= 0.078:
        inputs['fcl_closa'] = 1.00 + 1.290*inputs['icl_cloins']
    else:
        inputs['fcl_closa'] = 1.05 + 0.645*inputs['icl_cloins']

    # Calculate tcl through an optimization function
    ### tcl function, which also updates hc and fcl
    def getTclZero(tcl_in):
        inputs['hc_convht'] = 2.38*math.pow(abs(tcl_in-inputs['ta_airtmp']),0.25)
        if inputs['hc_convht'] < 12.1*math.sqrt(inputs['var_airvel']):
            inputs['hc_convht'] = 12.1*math.sqrt(inputs['var_airvel'])

        tcl_zero = 35.7-0.028*(inputs['m_metrate']-inputs['w_mechpow'])\
            -inputs['icl_cloins']*((3.96*10**-8)*inputs['fcl_closa']*((tcl_in+273)**4-(inputs['tr_radtmp']+273)**4)\
            +inputs['fcl_closa']*inputs['hc_convht']*(tcl_in-inputs['ta_airtmp']))\
            -tcl_in

        return tcl_zero

    ### run iteration, stop for step size or precision goal
    tcl_current = 200 # starting tcl value, C
    gamma = 0.01 # step size multiplier
    precision = 0.001
    prev_stepsize = 1
    max_iters = 500
    cnt_iters = 0
    while prev_stepsize > precision:
        tcl_prev = tcl_current
        tcl_zero = getTclZero(tcl_prev)
        tcl_current -= gamma * abs(tcl_zero) # calc step size based on tcl diff
        prev_stepsize = abs(tcl_prev-tcl_current) # as step sizes get smaller precision is increased
        # print('iteration:', cnt_iters, tcl_prev, tcl_current, tcl_zero, prev_stepsize)

        cnt_iters += 1
        if cnt_iters == max_iters:
            print('Breaking tcl iteration due to max iterations')
            break

    print('Final tcl:', tcl_current, prev_stepsize, cnt_iters)
    print('Final hc, fcl:', inputs['hc_convht'], inputs['fcl_closa'])
    inputs['tcl_clotmp'] = tcl_current

    # Add convenience inputs
    inputs['mw_neteng'] = inputs['m_metrate'] - inputs['w_mechpow']
    inputs['t_nettmp'] = inputs['tcl_clotmp'] - inputs['ta_airtmp']
    inputs['metfactor'] = 0.303*math.exp(-0.036*inputs['m_metrate'])+0.028
    print('m-w, tcl-ta', inputs['mw_neteng'], inputs['t_nettmp'])

    # Calculate outputs, which are heat values scaled by metabolic rate
    outputs = {
        'h_intprod': inputs['metfactor']*inputs['mw_neteng'],
        'ed_watvapdiff': inputs['metfactor']*(3.05*10**-3)*(5733-6.99*inputs['mw_neteng']-inputs['pa_wtrpres']),
        'esw_sweat': inputs['metfactor']*0.42*(inputs['mw_neteng']-58.15) if inputs['mw_neteng']>58.15 else 0, # if/else from comfort_models
        'ere_resp': inputs['metfactor']*(1.7*10**-5)*inputs['m_metrate']*(5867-inputs['pa_wtrpres']),
        'l_dryresp': inputs['metfactor']*0.0014*inputs['m_metrate']*(34-inputs['ta_airtmp']),
        'r_radiation': inputs['metfactor']*(3.96*10**-8)*inputs['fcl_closa']*((inputs['tcl_clotmp']+273)**4-(inputs['tr_radtmp']+273)**4),
        'c_convection': inputs['metfactor']*inputs['fcl_closa']*inputs['hc_convht']*inputs['t_nettmp'],
    }

    for key in ['h_intprod','ed_watvapdiff','esw_sweat','ere_resp','l_dryresp','r_radiation','c_convection']:
        print(key, outputs[key]/inputs['metfactor'])

    # Calculate PMV and PPD
    outputs['pmv'] = outputs['h_intprod']\
                    -outputs['ed_watvapdiff']-outputs['esw_sweat']-outputs['ere_resp']\
                    -outputs['l_dryresp']-outputs['r_radiation']-outputs['c_convection']

    outputs['ppd'] = 100 - 95 * math.exp(-0.3353*outputs['pmv']**4 - 0.2179*outputs['pmv']**2)

    print('PMV, PPD, metfactor:', outputs['pmv'], outputs['ppd'], inputs['metfactor'])

    # Check if function meets limits
    for key in limits:
        if not limits[key]:
            continue

        prop = None
        if key in inputs:
            prop = inputs[key]
        elif key in outputs:
            prop = outputs[key]
        else:
            print('WARNING key not found for limit check:', key)

        if prop and (prop<limits[key][0] or prop>limits[key][1]):
            print('WARNING variable outside limits:', key, prop, limits[key])

    # Return outputs
    return outputs




#
