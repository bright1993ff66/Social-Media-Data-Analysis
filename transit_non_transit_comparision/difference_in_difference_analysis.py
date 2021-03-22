import pandas as pd
import numpy as np
import csv
import os
from collections import Counter

import pytz
from datetime import datetime
from dateutil.relativedelta import relativedelta

import data_paths
from transit_non_transit_comparision.before_and_after_final_tpu import pos_percent_minus_neg_percent
import utils

# statistics
import statsmodels.formula.api as smf

# Hong Kong and Shanghai share the same time zone.
# Hence, we transform the utc time in our dataset into Shanghai time
time_zone_hk = pytz.timezone('Asia/Shanghai')
october_1_start = datetime(2016, 10, 1, 0, 0, 0, tzinfo=time_zone_hk)
october_31_end = datetime(2016, 10, 31, 23, 59, 59, tzinfo=time_zone_hk)
december_1_start = datetime(2016, 12, 1, 0, 0, 0, tzinfo=time_zone_hk)
december_31_end = datetime(2016, 12, 31, 23, 59, 59, tzinfo=time_zone_hk)


class DID_Analysis_Three_Areas(object):

    def __init__(self, regression_formula, interested_variables, regres_data_save_path,
                 regres_data_filename, did_result_save_path, did_result_filename, combined_or_not):
        self.regres_formula = regression_formula
        self.variable_list = interested_variables
        self.regres_data_save_path = regres_data_save_path
        self.regres_data_filename = regres_data_filename
        self.did_result_save_path = did_result_save_path
        self.did_result_filename = did_result_filename
        self.combined_or_not = combined_or_not

    def did_regres(self):
        pass


def transform_string_time_to_datetime(string):
    """
    :param string: the string which records the time of the posted tweets
    :return: a datetime object which could get access to the year, month, day easily
    """
    assert isinstance(string, str)
    datetime_object = datetime.strptime(string, '%Y-%m-%d %H:%M:%S+08:00')
    final_time_object = datetime_object.replace(tzinfo=time_zone_hk)
    return final_time_object


def add_post_variable(string, opening_start_date, opening_end_date, check_window=0, consider_lag_effect=False):
    """
    Add the value of the POST variable in the DID analysis
    :param string: the time string
    :param opening_start_date: the opening date of the studied station
    :param opening_end_date: the closing date of the studied station
    :param check_window: the month window size used to check the temporal effect of the studied station
    :param consider_lag_effect: Whether consider the lead-lag effect or not. If False, this function just considers
    the months immediate after the introduction of subway stations
    :return: the post variable based on the time of one tweet
    """
    time_object = transform_string_time_to_datetime(string)
    if check_window == 0:
        if time_object > opening_end_date:
            return 1
        elif time_object < opening_start_date:
            return 0
        else:
            return 'not considered'
    else:
        left_time_range = opening_start_date - relativedelta(months=check_window)
        if consider_lag_effect:
            right_time_range = opening_end_date + relativedelta(months=check_window * 2)
            opening_end_date = opening_end_date + relativedelta(months=check_window)
        else:
            right_time_range = opening_end_date + relativedelta(months=check_window)

        # print("The left time is: {}".format(left_time_range))
        # print("The right time is: {}".format(right_time_range))

        if left_time_range <= time_object < opening_start_date:
            return 0
        elif opening_end_date < time_object <= right_time_range:
            return 1
        else:
            return 'not considered'


def get_population_one_area_combined(dataframe: pd.DataFrame, census_dict: dict):
    """
    Get the population data based on treatment and control setting
    :param dataframe: a pandas dataframe containing the data for DID analysis
    :param census_dict: dictionary saving the population and median income information
    :return: dataframe containing the population
    """
    assert 'T_i_t' in dataframe, 'The dataframe should have treatment and control indicator'
    dataframe['Population_log'] = dataframe.apply(
        lambda row: np.log(census_dict['treatment'][0]) if row['T_i_t'] == 1 else np.log(census_dict['control'][0]),
        axis=1)
    return dataframe


def get_population_one_area_seperate(dataframe: pd.DataFrame, census_dict: dict):
    """
    Get the population data based on treatment and control setting
    :param dataframe: a pandas dataframe containing the data for DID analysis
    :param census_dict: dictionary saving the population and median income information
    :return: dataframe containing the population
    """
    assert 'T_i_t' in dataframe, 'The dataframe should have treatment and control indicator'
    dataframe['Population_log'] = dataframe.apply(lambda row: np.log(census_dict[row['SmallTPU']][0]), axis=1)
    return dataframe


def get_median_income_one_area_combined(dataframe: pd.DataFrame, census_dict: dict):
    """
    Get the population data based on treatment and control setting
    :param dataframe: a pandas dataframe containing the data for DID analysis
    :param census_dict: dictionary saving the population and median income information
    :return: dataframe containing the population
    """
    assert 'T_i_t' in dataframe, 'The dataframe should have treatment and control indicator'
    dataframe['Median_Income_log'] = dataframe.apply(
        lambda row: np.log(census_dict['treatment'][1]) if row['T_i_t'] == 1 else np.log(census_dict['control'][1]),
        axis=1)
    return dataframe


def get_median_income_one_area_seperate(dataframe: pd.DataFrame, census_dict: dict):
    """
    Get the population data based on treatment and control setting
    :param dataframe: a pandas dataframe containing the data for DID analysis
    :param census_dict: dictionary saving the population and median income information
    :return: dataframe containing the population
    """
    assert 'T_i_t' in dataframe, 'The dataframe should have treatment and control indicator'
    dataframe['Median_Income_log'] = dataframe.apply(lambda row: np.log(census_dict[row['SmallTPU']][1]), axis=1)
    return dataframe


def get_population_three_areas_combined(dataframe: pd.DataFrame, census_dict: dict):
    """
    Get the population data based on treatment and control setting
    :param dataframe: a pandas dataframe containing the data for DID analysis
    :param census_dict: dictionary saving the population and median income information
    :return: dataframe containing the population
    """
    assert 'T_i_t' in dataframe, 'The dataframe should have treatment and control indicator'
    assert 'Area_name' in dataframe, "The dataframe should have one column saving the area name"
    assert 'kwun_tong' in census_dict, 'The dictionary should contain whampoa & ho man tin data'
    assert 'south_horizons' in census_dict, 'The dictionary should contain south horizons & lei tung data'
    assert 'ocean_park' in census_dict, 'The dictionary should contain ocean park & wong chuk hang data'
    result_population_log_list = []
    dataframe_copy = dataframe.copy()
    for index, row in dataframe_copy.iterrows():
        if (row['T_i_t'] == 1) and (row['Area_name'] == 'kwun_tong'):
            result_population_log_list.append(np.log(census_dict['kwun_tong'][0][0]))
        elif (row['T_i_t'] == 0) and (row['Area_name'] == 'kwun_tong'):
            result_population_log_list.append(np.log(census_dict['kwun_tong'][1][0]))
        elif (row['T_i_t'] == 1) and (row['Area_name'] == 'south_horizons'):
            result_population_log_list.append(np.log(census_dict['south_horizons'][0][0]))
        elif (row['T_i_t'] == 0) and (row['Area_name'] == 'south_horizons'):
            result_population_log_list.append(np.log(census_dict['south_horizons'][1][0]))
        elif (row['T_i_t'] == 1) and (row['Area_name'] == 'ocean_park'):
            result_population_log_list.append(np.log(census_dict['ocean_park'][0][0]))
        elif (row['T_i_t'] == 0) and (row['Area_name'] == 'ocean_park'):
            result_population_log_list.append(np.log(census_dict['ocean_park'][1][0]))
        else:
            raise ValueError('Something wrong with the area name...')
    dataframe_copy['Population_log'] = result_population_log_list
    return dataframe_copy


def get_median_income_three_areas_combined(dataframe: pd.DataFrame, census_dict: dict):
    """
    Get the median income data based on treatment and control setting
    :param dataframe: a pandas dataframe containing the data for DID analysis
    :param census_dict: dictionary saving the population and median income information
    :return: dataframe containing the median income
    """
    assert 'T_i_t' in dataframe, 'The dataframe should have treatment and control indicator'
    assert 'Area_name' in dataframe, "The dataframe should have one column saving the area name"
    assert 'kwun_tong' in census_dict, 'The dictionary should contain whampoa & ho man tin data'
    assert 'south_horizons' in census_dict, 'The dictionary should contain south horizons & lei tung data'
    assert 'ocean_park' in census_dict, 'The dictionary should contain ocean park & wong chuk hang data'
    result_median_income_log_list = []
    dataframe_copy = dataframe.copy()
    for index, row in dataframe_copy.iterrows():
        if (row['T_i_t'] == 1) and (row['Area_name'] == 'kwun_tong'):
            result_median_income_log_list.append(np.log(census_dict['kwun_tong'][0][1]))
        elif (row['T_i_t'] == 0) and (row['Area_name'] == 'kwun_tong'):
            result_median_income_log_list.append(np.log(census_dict['kwun_tong'][1][1]))
        elif (row['T_i_t'] == 1) and (row['Area_name'] == 'south_horizons'):
            result_median_income_log_list.append(np.log(census_dict['south_horizons'][0][1]))
        elif (row['T_i_t'] == 0) and (row['Area_name'] == 'south_horizons'):
            result_median_income_log_list.append(np.log(census_dict['south_horizons'][1][1]))
        elif (row['T_i_t'] == 1) and (row['Area_name'] == 'ocean_park'):
            result_median_income_log_list.append(np.log(census_dict['ocean_park'][0][1]))
        elif (row['T_i_t'] == 0) and (row['Area_name'] == 'ocean_park'):
            result_median_income_log_list.append(np.log(census_dict['ocean_park'][1][1]))
        else:
            raise ValueError('Something wrong with the area name...')
    dataframe_copy['Median_Income_log'] = result_median_income_log_list
    return dataframe_copy


def build_dataframe_based_on_set(datapath, tpu_set, selected_user_set):
    """
    Build the dataframes based on the given tpu set
    :param datapath: the datapath saving the tweets posted in each tpu
    :param tpu_set: a python set saving the considered tpu names
    :param selected_user_set: a set containing the id of users we are interested in
    :return: a pandas dataframe saving the tweets posted in the considered tpus
    """
    tpu_name_list = []
    dataframe_list = []
    for tpu in tpu_set:
        tpu_name_list.append(tpu)
        dataframe = pd.read_csv(os.path.join(datapath, tpu, tpu + '_data.csv'), encoding='utf-8', dtype='str',
                                quoting=csv.QUOTE_NONNUMERIC)
        dataframe['user_id_str'] = dataframe.apply(lambda row: np.int64(float(row['user_id_str'])), axis=1)
        dataframe_select = dataframe.loc[dataframe['user_id_str'].isin(selected_user_set)]
        dataframe_list.append(dataframe_select)
    combined_dataframe = pd.concat(dataframe_list, axis=0)
    return combined_dataframe


def build_regress_data_three_areas_combined(kwun_tong_treatment, kwun_tong_control, south_horizons_treatment,
                                            south_horizons_control, ocean_park_treatment, ocean_park_control,
                                            tpu_info_dataframe, check_window_value=0, sentiment_did=False):
    """
    Build dataframes for the combined DID analysis based on treatment & control dataframes of each station
    :param kwun_tong_treatment: the dataframe saving tweets for kwun tong treatment area
    :param kwun_tong_control: the dataframe saving tweets for kwun tong control area
    :param south_horizons_treatment: the dataframe saving tweets for south horizons treatment area
    :param south_horizons_control: the dataframe saving tweets for south horizons control area
    :param ocean_park_treatment: the dataframe saving tweets for ocean park treatment area
    :param ocean_park_control: the dataframe saving tweets for ocean park control area
    :param tpu_info_dataframe: the dataframe saving the census data for each tpu setting
    :param check_window_value: the month window we consider when doing the DID analysis
    :param sentiment_did: whether doing the did analysis on sentiment or not
    :return: a combined dataframe which could be used for combined DID analysis
    """
    result_dataframe = pd.DataFrame()
    kwun_tong_line_treatment_tpu_set = {'236', '243', '245'}
    kwun_tong_line_control_tpu_set = {'247', '234', '242', '212', '235'}
    south_horizons_lei_tung_treatment_tpu_set = {'174'}
    south_horizons_lei_tung_control_tpu_set = {'172', '181 - 182'}
    ocean_park_wong_chuk_hang_treatment_tpu_set = {'175 - 176'}
    ocean_park_wong_chuk_hang_control_tpu_set = {'181 - 182', '183 - 184'}
    treatment_set = set(list(kwun_tong_line_treatment_tpu_set) + list(south_horizons_lei_tung_treatment_tpu_set) +
                        list(ocean_park_wong_chuk_hang_treatment_tpu_set))
    control_set = set(list(kwun_tong_line_control_tpu_set) + list(south_horizons_lei_tung_control_tpu_set) +
                      list(ocean_park_wong_chuk_hang_control_tpu_set))
    print('The treatment set is: {}'.format(treatment_set))
    print('The control set is: {}'.format(control_set))
    # build the treatment control binary variable
    kwun_tong_treatment['T_i_t'] = [1] * kwun_tong_treatment.shape[0]
    kwun_tong_treatment['Area_num'] = [1] * kwun_tong_treatment.shape[0]
    kwun_tong_treatment['Area_name'] = ['kwun_tong'] * kwun_tong_treatment.shape[0]
    kwun_tong_control['T_i_t'] = [0] * kwun_tong_control.shape[0]
    kwun_tong_control['Area_num'] = [2] * kwun_tong_control.shape[0]
    kwun_tong_control['Area_name'] = ['kwun_tong'] * kwun_tong_control.shape[0]
    south_horizons_treatment['T_i_t'] = [1] * south_horizons_treatment.shape[0]
    south_horizons_treatment['Area_num'] = [3] * south_horizons_treatment.shape[0]
    south_horizons_treatment['Area_name'] = ['south_horizons'] * south_horizons_treatment.shape[0]
    south_horizons_control['T_i_t'] = [0] * south_horizons_control.shape[0]
    south_horizons_control['Area_num'] = [4] * south_horizons_control.shape[0]
    south_horizons_control['Area_name'] = ['south_horizons'] * south_horizons_control.shape[0]
    ocean_park_treatment['T_i_t'] = [1] * ocean_park_treatment.shape[0]
    ocean_park_treatment['Area_num'] = [5] * ocean_park_treatment.shape[0]
    ocean_park_treatment['Area_name'] = ['ocean_park'] * ocean_park_treatment.shape[0]
    ocean_park_control['T_i_t'] = [0] * ocean_park_control.shape[0]
    ocean_park_control['Area_num'] = [6] * ocean_park_control.shape[0]
    ocean_park_control['Area_name'] = ['ocean_park'] * ocean_park_control.shape[0]
    # add the post variable
    kwun_tong_treatment['Post'] = kwun_tong_treatment.apply(
        lambda row: add_post_variable(row['hk_time'], opening_start_date=october_1_start,
                                      opening_end_date=october_31_end, check_window=check_window_value,
                                      consider_lag_effect=False), axis=1)
    kwun_tong_control['Post'] = kwun_tong_control.apply(
        lambda row: add_post_variable(row['hk_time'], opening_start_date=october_1_start,
                                      opening_end_date=october_31_end, check_window=check_window_value,
                                      consider_lag_effect=False), axis=1)
    south_horizons_treatment['Post'] = south_horizons_treatment.apply(
        lambda row: add_post_variable(row['hk_time'], opening_start_date=december_1_start,
                                      opening_end_date=december_31_end, check_window=check_window_value,
                                      consider_lag_effect=False), axis=1)
    south_horizons_control['Post'] = south_horizons_control.apply(
        lambda row: add_post_variable(row['hk_time'], opening_start_date=december_1_start,
                                      opening_end_date=december_31_end, check_window=check_window_value,
                                      consider_lag_effect=False), axis=1)
    ocean_park_treatment['Post'] = ocean_park_treatment.apply(
        lambda row: add_post_variable(row['hk_time'], opening_start_date=december_1_start,
                                      opening_end_date=december_31_end, check_window=check_window_value,
                                      consider_lag_effect=False), axis=1)
    ocean_park_control['Post'] = ocean_park_control.apply(
        lambda row: add_post_variable(row['hk_time'], opening_start_date=december_1_start,
                                      opening_end_date=december_31_end, check_window=check_window_value,
                                      consider_lag_effect=False), axis=1)

    # Construct the dictionary having the census data for the treatment area and control area
    tpu_info_dataframe['SmallTPU'] = tpu_info_dataframe.apply(lambda row: str(row['SmallTPU']), axis=1)
    treatment_info_data = tpu_info_dataframe.loc[tpu_info_dataframe['SmallTPU'].isin(treatment_set)]
    control_info_data = tpu_info_dataframe.loc[tpu_info_dataframe['SmallTPU'].isin(control_set)]
    kwun_tong_treatment_info_data = tpu_info_dataframe.loc[tpu_info_dataframe['SmallTPU'].isin(
        kwun_tong_line_treatment_tpu_set)]
    kwun_tong_control_info_data = tpu_info_dataframe.loc[tpu_info_dataframe['SmallTPU'].isin(
        kwun_tong_line_control_tpu_set)]
    south_horizons_treatment_info_data = tpu_info_dataframe.loc[tpu_info_dataframe['SmallTPU'].isin(
        south_horizons_lei_tung_treatment_tpu_set)]
    south_horizons_control_info_data = tpu_info_dataframe.loc[tpu_info_dataframe['SmallTPU'].isin(
        south_horizons_lei_tung_control_tpu_set)]
    ocean_park_treatment_info_data = tpu_info_dataframe.loc[tpu_info_dataframe['SmallTPU'].isin(
        ocean_park_wong_chuk_hang_treatment_tpu_set)]
    ocean_park_control_info_data = tpu_info_dataframe.loc[tpu_info_dataframe['SmallTPU'].isin(
        ocean_park_wong_chuk_hang_control_tpu_set)]
    census_dict = {}
    for _, row in tpu_info_dataframe.iterrows():
        census_dict[str(row['SmallTPU'])] = [row['population'], row['m_income']]
    census_dict_update = {'treatment': [0, 0], 'control': [0, 0]}
    for tpu in census_dict:
        if tpu in treatment_set:
            census_dict_update['treatment'][0] += census_dict[tpu][0]
        elif tpu in control_set:
            census_dict_update['control'][0] += census_dict[tpu][0]
        else:
            raise ValueError('The area name is not true')
    census_dict_area = {'kwun_tong': [[0, 0], [0, 0]], 'south_horizons': [[0, 0], [0, 0]],
                        'ocean_park': [[0, 0], [0, 0]]}  # kwun_tong: [[pop, income]: treatment, [pop, income]: control]
    for tpu in census_dict:
        if tpu in kwun_tong_line_treatment_tpu_set:
            census_dict_area['kwun_tong'][0][0] += census_dict[tpu][0]
        elif tpu in kwun_tong_line_control_tpu_set:
            census_dict_area['kwun_tong'][1][0] += census_dict[tpu][0]
        elif tpu in south_horizons_lei_tung_treatment_tpu_set:
            census_dict_area['south_horizons'][0][0] += census_dict[tpu][0]
        elif tpu in south_horizons_lei_tung_control_tpu_set:
            census_dict_area['south_horizons'][1][0] += census_dict[tpu][0]
        elif tpu in ocean_park_wong_chuk_hang_treatment_tpu_set:
            census_dict_area['ocean_park'][0][0] += census_dict[tpu][0]
        elif tpu in ocean_park_wong_chuk_hang_control_tpu_set:
            census_dict_area['ocean_park'][1][0] += census_dict[tpu][0]
    census_dict_update['treatment'][1] = utils.weighted_average(treatment_info_data,
                                                                value_col='m_income', weight_col='population')
    census_dict_update['control'][1] = utils.weighted_average(control_info_data, value_col='m_income',
                                                              weight_col='population')
    census_dict_area['kwun_tong'][0][1] = utils.weighted_average(kwun_tong_treatment_info_data,
                                                                 value_col='m_income', weight_col='population')
    census_dict_area['kwun_tong'][1][1] = utils.weighted_average(kwun_tong_control_info_data,
                                                                 value_col='m_income', weight_col='population')
    census_dict_area['south_horizons'][0][1] = utils.weighted_average(south_horizons_treatment_info_data,
                                                                      value_col='m_income', weight_col='population')
    census_dict_area['south_horizons'][1][1] = utils.weighted_average(south_horizons_control_info_data,
                                                                      value_col='m_income', weight_col='population')
    census_dict_area['ocean_park'][0][1] = utils.weighted_average(ocean_park_treatment_info_data,
                                                                  value_col='m_income', weight_col='population')
    census_dict_area['ocean_park'][1][1] = utils.weighted_average(ocean_park_control_info_data,
                                                                  value_col='m_income', weight_col='population')
    # Create the tweet dataframe containing the tweets with year_month information
    dataframe_list = [kwun_tong_treatment, kwun_tong_control, south_horizons_treatment,
                      south_horizons_control, ocean_park_treatment, ocean_park_control]
    combined_dataframe = pd.concat(dataframe_list, axis=0, sort=True)
    combined_dataframe = combined_dataframe.reset_index(drop=True)
    combined_dataframe_without_not_considered = combined_dataframe.loc[combined_dataframe['Post'] != 'not considered']
    combined_data_copy = combined_dataframe_without_not_considered.copy()
    combined_data_copy['month_plus_year'] = combined_data_copy.apply(
        lambda row: str(int(float(row['year']))) + '_' + str(int(float(row['month']))), axis=1)
    # Construct the data for the difference in difference analysis
    result_dataframe_copy = result_dataframe.copy()
    if sentiment_did:
        area_name_list = []
        time_list = []
        t_i_t_list = []
        post_list = []
        sentiment_list = []
        sentiment_dict = {}
        for _, dataframe in combined_data_copy.groupby(['month_plus_year', 'T_i_t', 'Post', 'Area_name']):
            time = str(list(dataframe['month_plus_year'])[0])
            t_i_t = str(list(dataframe['T_i_t'])[0])
            post = str(list(dataframe['Post'])[0])
            area_name = list(dataframe['Area_name'])[0]
            sentiment_dict[time + '+' + t_i_t + '+' + post + '+' + area_name] = pos_percent_minus_neg_percent(dataframe)
        for key in list(sentiment_dict.keys()):
            # don't consider the tweets posted in 2016_10(for Whampoa and Ho Man Tin) or 2016_12(for other stations)
            info_list = key.split('+')
            if info_list[0] not in ['2016_10', '2016_12']:
                time_list.append(info_list[0])
                t_i_t_list.append(int(info_list[1]))
                post_list.append(int(info_list[2]))
                area_name_list.append(info_list[3])
                sentiment_list.append(sentiment_dict[key])
        result_dataframe_copy['Time'] = time_list
        result_dataframe_copy['T_i_t'] = t_i_t_list
        result_dataframe_copy['Area_name'] = area_name_list
        result_dataframe_copy['Post'] = post_list
        result_dataframe_copy['Sentiment'] = sentiment_list
        result_dataframe_copy['month'] = result_dataframe_copy.apply(lambda row: int(row['Time'][5:]), axis=1)
        dataframe_with_population = get_population_three_areas_combined(result_dataframe_copy,
                                                                        census_dict=census_dict_area)
        final_dataframe = get_median_income_three_areas_combined(dataframe_with_population,
                                                                 census_dict=census_dict_area)
    else:
        area_name_list = []
        time_list = []
        t_i_t_list = []
        post_list = []
        activity_list, log_activity_list = [], []
        activity_dict, log_activity_dict = {}, {}
        for _, dataframe in combined_data_copy.groupby(['month_plus_year', 'T_i_t', 'Post', 'Area_name']):
            time = str(list(dataframe['month_plus_year'])[0])
            t_i_t = str(list(dataframe['T_i_t'])[0])
            post = str(list(dataframe['Post'])[0])
            area_name = list(dataframe['Area_name'])[0]
            activity_dict[time + '+' + t_i_t + '+' + post + '+' + area_name] = dataframe.shape[0]
            log_activity_dict[time + '+' + t_i_t + '+' + post + '+' + area_name] = np.log(dataframe.shape[0])
        for key in list(activity_dict.keys()):
            # don't consider the tweets posted in 2016_10(for Whampoa and Ho Man Tin) or 2016_12(for other stations)
            info_list = key.split('+')
            if info_list[0] not in ['2016_10', '2016_12']:
                time_list.append(info_list[0])
                t_i_t_list.append(int(info_list[1]))
                post_list.append(int(info_list[2]))
                activity_list.append(activity_dict[key])
                log_activity_list.append(log_activity_dict[key])
                area_name_list.append(info_list[3])
        result_dataframe_copy['Time'] = time_list
        result_dataframe_copy['T_i_t'] = t_i_t_list
        result_dataframe_copy['Post'] = post_list
        result_dataframe_copy['Area_name'] = area_name_list
        result_dataframe_copy['Activity'] = activity_list
        result_dataframe_copy['log_Activity'] = log_activity_list
        result_dataframe_copy['month'] = result_dataframe_copy.apply(lambda row: int(row['Time'][5:]), axis=1)
        dataframe_with_population = get_population_three_areas_combined(result_dataframe_copy,
                                                                        census_dict=census_dict_area)
        final_dataframe = get_median_income_three_areas_combined(dataframe_with_population,
                                                                 census_dict=census_dict_area)
    remove_mask = (final_dataframe['Area_name'] == 'south_horizons') & (final_dataframe['Time'].isin(
        ['2018_11', '2018_12']))
    final_dataframe_select = final_dataframe.loc[~remove_mask]
    return final_dataframe_select


def build_regress_data_three_areas_seperate(kwun_tong_treatment, kwun_tong_control, south_horizons_treatment,
                                            south_horizons_control, ocean_park_treatment, ocean_park_control,
                                            tpu_info_dataframe, check_window_value=0, sentiment_did=False):
    """
    Build dataframes for the combined DID analysis based on treatment & control dataframes of each station
    :param kwun_tong_treatment: the dataframe saving tweets for kwun tong treatment area
    :param kwun_tong_control: the dataframe saving tweets for kwun tong control area
    :param south_horizons_treatment: the dataframe saving tweets for south horizons treatment area
    :param south_horizons_control: the dataframe saving tweets for south horizons control area
    :param ocean_park_treatment: the dataframe saving tweets for ocean park treatment area
    :param ocean_park_control: the dataframe saving tweets for ocean park control area
    :param tpu_info_dataframe: the dataframe saving the census data for each tpu setting
    :param check_window_value: the month window we consider when doing the DID analysis
    :param sentiment_did: whether doing the did analysis on sentiment or not
    :return: a combined dataframe which could be used for combined DID analysis
    """
    result_dataframe = pd.DataFrame()
    kwun_tong_line_treatment_tpu_set = {'236', '243', '245'}
    kwun_tong_line_control_tpu_set = {'247', '234', '242', '212', '235'}
    south_horizons_lei_tung_treatment_tpu_set = {'174'}
    south_horizons_lei_tung_control_tpu_set = {'172', '181 - 182'}
    ocean_park_wong_chuk_hang_treatment_tpu_set = {'175 - 176'}
    ocean_park_wong_chuk_hang_control_tpu_set = {'181 - 182', '183 - 184'}
    treatment_set = set(list(kwun_tong_line_treatment_tpu_set) + list(south_horizons_lei_tung_treatment_tpu_set) +
                        list(ocean_park_wong_chuk_hang_treatment_tpu_set))
    control_set = set(list(kwun_tong_line_control_tpu_set) + list(south_horizons_lei_tung_control_tpu_set) +
                      list(ocean_park_wong_chuk_hang_control_tpu_set))
    print('The treatment set is: {}'.format(treatment_set))
    print('The control set is: {}'.format(control_set))
    # build the treatment control binary variable
    kwun_tong_treatment['T_i_t'] = [1] * kwun_tong_treatment.shape[0]
    kwun_tong_control['T_i_t'] = [0] * kwun_tong_control.shape[0]
    south_horizons_treatment['T_i_t'] = [1] * south_horizons_treatment.shape[0]
    south_horizons_control['T_i_t'] = [0] * south_horizons_control.shape[0]
    ocean_park_treatment['T_i_t'] = [1] * ocean_park_treatment.shape[0]
    ocean_park_control['T_i_t'] = [0] * ocean_park_control.shape[0]
    # add the post variable
    kwun_tong_treatment['Post'] = kwun_tong_treatment.apply(
        lambda row: add_post_variable(row['hk_time'], opening_start_date=october_1_start,
                                      opening_end_date=october_31_end, check_window=check_window_value,
                                      consider_lag_effect=False), axis=1)
    kwun_tong_control['Post'] = kwun_tong_control.apply(
        lambda row: add_post_variable(row['hk_time'], opening_start_date=october_1_start,
                                      opening_end_date=october_31_end, check_window=check_window_value,
                                      consider_lag_effect=False), axis=1)
    south_horizons_treatment['Post'] = south_horizons_treatment.apply(
        lambda row: add_post_variable(row['hk_time'], opening_start_date=december_1_start,
                                      opening_end_date=december_31_end, check_window=check_window_value,
                                      consider_lag_effect=False), axis=1)
    south_horizons_control['Post'] = south_horizons_control.apply(
        lambda row: add_post_variable(row['hk_time'], opening_start_date=december_1_start,
                                      opening_end_date=december_31_end, check_window=check_window_value,
                                      consider_lag_effect=False), axis=1)
    ocean_park_treatment['Post'] = ocean_park_treatment.apply(
        lambda row: add_post_variable(row['hk_time'], opening_start_date=december_1_start,
                                      opening_end_date=december_31_end, check_window=check_window_value,
                                      consider_lag_effect=False), axis=1)
    ocean_park_control['Post'] = ocean_park_control.apply(
        lambda row: add_post_variable(row['hk_time'], opening_start_date=december_1_start,
                                      opening_end_date=december_31_end, check_window=check_window_value,
                                      consider_lag_effect=False), axis=1)

    # Construct the dictionary having the census data for the treatment area and control area
    tpu_info_dataframe['SmallTPU'] = tpu_info_dataframe.apply(lambda row: str(row['SmallTPU']), axis=1)
    census_dict = {}  # [log(population), log(median_income), tpu_index]
    for _, row in tpu_info_dataframe.iterrows():
        census_dict[row['SmallTPU']] = [row['population'], row['m_income'], row['tpu_index']]
    # Create the tweet dataframe containing the tweets with year_month information
    dataframe_list = [kwun_tong_treatment, kwun_tong_control, south_horizons_treatment,
                      south_horizons_control, ocean_park_treatment, ocean_park_control]
    combined_dataframe = pd.concat(dataframe_list, axis=0, sort=True)
    combined_dataframe = combined_dataframe.reset_index(drop=True)
    combined_dataframe_without_not_considered = combined_dataframe.loc[combined_dataframe['Post'] != 'not considered']
    combined_data_copy = combined_dataframe_without_not_considered.copy()
    combined_data_copy['month_plus_year'] = combined_data_copy.apply(
        lambda row: str(int(float(row['year']))) + '_' + str(int(float(row['month']))), axis=1)
    # Construct the data for the difference in difference analysis
    result_dataframe_copy = result_dataframe.copy()
    if sentiment_did:
        tpu_list = []
        time_list = []
        t_i_t_list = []
        post_list = []
        sentiment_list = []
        sentiment_dict = {}
        for _, dataframe in combined_data_copy.groupby(['TPU_cross_sectional', 'month_plus_year', 'T_i_t', 'Post']):
            time = str(list(dataframe['month_plus_year'])[0])
            t_i_t = str(list(dataframe['T_i_t'])[0])
            post = str(list(dataframe['Post'])[0])
            tpu_info = str(list(dataframe['TPU_cross_sectional'])[0])
            sentiment_dict[time + '+' + t_i_t + '+' + post + '+' + tpu_info] = pos_percent_minus_neg_percent(dataframe)
        for key in list(sentiment_dict.keys()):
            # don't consider the tweets posted in 2016_10(for Whampoa and Ho Man Tin) or 2016_12(for other stations)
            info_list = key.split('+')
            if info_list[0] not in ['2016_10', '2016_12']:
                time_list.append(info_list[0])
                t_i_t_list.append(int(info_list[1]))
                post_list.append(int(info_list[2]))
                tpu_list.append(str(info_list[3]))
                sentiment_list.append(sentiment_dict[key])
        result_dataframe_copy['TPU'] = tpu_list
        result_dataframe_copy['Time'] = time_list
        result_dataframe_copy['T_i_t'] = t_i_t_list
        result_dataframe_copy['Post'] = post_list
        result_dataframe_copy['Sentiment'] = sentiment_list
        result_dataframe_copy['month'] = result_dataframe_copy.apply(lambda row: int(row['Time'][5:]), axis=1)
    else:
        tpu_list = []
        time_list = []
        t_i_t_list = []
        post_list = []
        activity_list, log_activity_list = [], []
        activity_dict, log_activity_dict = {}, {}
        for _, dataframe in combined_data_copy.groupby(['TPU_cross_sectional', 'month_plus_year', 'T_i_t', 'Post']):
            time = str(list(dataframe['month_plus_year'])[0])
            t_i_t = str(list(dataframe['T_i_t'])[0])
            post = str(list(dataframe['Post'])[0])
            tpu_info = str(list(dataframe['TPU_cross_sectional'])[0])
            log_activity_dict[time + '+' + t_i_t + '+' + post + '+' + tpu_info] = np.log(dataframe.shape[0])
            activity_dict[time + '+' + t_i_t + '+' + post + '+' + tpu_info] = dataframe.shape[0]
        for key in list(log_activity_dict.keys()):
            # don't consider the tweets posted in 2016_10(for Whampoa and Ho Man Tin) or 2016_12(for other stations)
            info_list = key.split('+')
            if info_list[0] not in ['2016_10', '2016_12']:
                time_list.append(info_list[0])
                t_i_t_list.append(int(info_list[1]))
                post_list.append(int(info_list[2]))
                tpu_list.append(str(info_list[3]))
                activity_list.append(activity_dict[key])
                log_activity_list.append(log_activity_dict[key])
        result_dataframe_copy['TPU'] = tpu_list
        result_dataframe_copy['Time'] = time_list
        result_dataframe_copy['month'] = result_dataframe_copy.apply(lambda row: int(row['Time'][5:]), axis=1)
        result_dataframe_copy['T_i_t'] = t_i_t_list
        result_dataframe_copy['Post'] = post_list
        result_dataframe_copy['Activity'] = activity_list
        result_dataframe_copy['log_Activity'] = log_activity_list
    # Add the population, median income and tpu index information
    result_dataframe_copy['Population_log'] = result_dataframe_copy.apply(
        lambda row: np.log(census_dict[row['TPU']][0]),
        axis=1)
    result_dataframe_copy['Median_Income_log'] = result_dataframe_copy.apply(
        lambda row: np.log(census_dict[row['TPU']][1]),
        axis=1)
    result_dataframe_copy['TPU_Index'] = result_dataframe_copy.apply(lambda row: census_dict[row['TPU']][2],
                                                                     axis=1)
    return result_dataframe_copy


def build_regress_dataframe_for_one_station_combined(treatment_dataframe, control_dataframe,
                                                     station_open_month_start, station_open_month_end,
                                                     open_year_plus_month, tpu_info_dataframe,
                                                     check_window_value=0, check_area_name=None):
    """
    Build the dataframe for one influenced area
    :param treatment_dataframe: the tweet dataframe for treatment area
    :param control_dataframe: the tweet dataframe for control area
    :param station_open_month_start: the starting time of the month when the studied station opens
    :param station_open_month_end: the ending time of the month when the studied station opens
    :param open_year_plus_month: the month plus year information
    :param tpu_info_dataframe: the dataframe saving the census data for each tpu setting
    :param check_window_value: the window size for DID analysis
    :param check_area_name: the name of the study area
    :return: a pandas dataframe which could be used for the following DID analysis
    """
    # check the date
    assert open_year_plus_month in ['2016_10', '2016_12']
    result_dataframe = pd.DataFrame()
    # build the T_i_t variable
    ones_list = [1] * treatment_dataframe.shape[0]
    treatment_dataframe['T_i_t'] = ones_list
    zeros_list = [0] * control_dataframe.shape[0]
    control_dataframe['T_i_t'] = zeros_list
    # build the post variable
    treatment_dataframe['Post'] = treatment_dataframe.apply(
        lambda row: add_post_variable(row['hk_time'], opening_start_date=station_open_month_start,
                                      opening_end_date=station_open_month_end, check_window=check_window_value,
                                      consider_lag_effect=False), axis=1)
    control_dataframe['Post'] = control_dataframe.apply(
        lambda row: add_post_variable(row['hk_time'], opening_start_date=station_open_month_start,
                                      opening_end_date=station_open_month_end, check_window=check_window_value,
                                      consider_lag_effect=False), axis=1)
    # Check the distribution of T_i_t and POST variables
    print('Check the post variable distribution of treatment group: {}'.format(
        Counter(treatment_dataframe['Post'])))
    print('Check the T_i_t variable distribution of treatment group: {}'.format(
        Counter(treatment_dataframe['T_i_t'])))
    print('Check the post variable distribution of control group: {}'.format(
        Counter(control_dataframe['Post'])))
    print('Check the T_i_t variable distribution of control group: {}'.format(
        Counter(control_dataframe['T_i_t'])))

    # Construct the dictionary having the census data for the treatment area and control area
    treatment_dataframe['TPU_cross_sectional'] = treatment_dataframe['TPU_cross_sectional'].astype(str)
    control_dataframe['TPU_cross_sectional'] = control_dataframe['TPU_cross_sectional'].astype(str)
    treatment_set = set(treatment_dataframe['TPU_cross_sectional'])
    control_set = set(control_dataframe['TPU_cross_sectional'])
    tpu_info_dataframe['SmallTPU'] = tpu_info_dataframe.apply(lambda row: str(row['SmallTPU']), axis=1)
    treatment_info_data = tpu_info_dataframe.loc[tpu_info_dataframe['SmallTPU'].isin(treatment_set)]
    control_info_data = tpu_info_dataframe.loc[tpu_info_dataframe['SmallTPU'].isin(control_set)]
    census_dict = {'treatment': [0, 0], 'control': [0, 0]}  # [population, median_income]
    for _, row in tpu_info_dataframe.iterrows():
        if row.SmallTPU in treatment_set:
            census_dict['treatment'][0] += tpu_info_dataframe.loc[tpu_info_dataframe.SmallTPU == row.SmallTPU,
                                                                  'population'].values[0]
        else:
            census_dict['control'][0] += tpu_info_dataframe.loc[tpu_info_dataframe.SmallTPU == row.SmallTPU,
                                                                'population'].values[0]
    census_dict['treatment'][1] = utils.weighted_average(treatment_info_data,
                                                         value_col='m_income', weight_col='population')
    census_dict['control'][1] = utils.weighted_average(control_info_data, value_col='m_income', weight_col='population')

    # Construct the dataframe for the DID regression analysis
    combined_dataframe = pd.concat([treatment_dataframe, control_dataframe], axis=0)
    combined_dataframe = combined_dataframe.reset_index(drop=True)
    # We don't consider the tweets posted on the open month of the MTR stations
    combined_dataframe_without_not_considered = combined_dataframe.loc[combined_dataframe['Post'] != 'not considered']
    combined_data_copy = combined_dataframe_without_not_considered.copy()
    combined_data_copy['month_plus_year'] = combined_data_copy.apply(
        lambda row: str(int(float(row['year']))) + '_' + str(int(float(row['month']))), axis=1)
    sentiment_dict = {}
    activity_dict = {}
    activity_dict_log = {}
    for _, dataframe in combined_data_copy.groupby(['month_plus_year', 'T_i_t', 'Post']):
        time = str(list(dataframe['month_plus_year'])[0])
        t_i_t = str(list(dataframe['T_i_t'])[0])
        post = str(list(dataframe['Post'])[0])
        sentiment_dict[time + '+' + t_i_t + '+' + post] = pos_percent_minus_neg_percent(dataframe)
        activity_dict[time + '+' + t_i_t + '+' + post] = dataframe.shape[0]
        activity_dict_log[time + '+' + t_i_t + '+' + post] = np.log(dataframe.shape[0])
    result_dataframe_copy = result_dataframe.copy()
    t_i_t_list = []
    time_list, post_list = [], []
    sentiment_list = []
    activity_list, activity_log_list = [], []
    for key in list(sentiment_dict.keys()):
        # don't consider the tweets posted in 2016_10(for Whampoa and Ho Man Tin) or 2016_12(for other stations)
        info_list = key.split('+')
        if info_list[0] != open_year_plus_month:
            time_list.append(info_list[0])
            t_i_t_list.append(int(info_list[1]))
            post_list.append(int(info_list[2]))
            sentiment_list.append(sentiment_dict[key])
            activity_list.append(activity_dict[key])
            activity_log_list.append(activity_dict_log[key])
    result_dataframe_copy['Time'] = time_list
    result_dataframe_copy['month'] = result_dataframe_copy.apply(lambda row: int(row['Time'][5:]), axis=1)
    result_dataframe_copy['T_i_t'] = t_i_t_list
    result_dataframe_copy['Post'] = post_list
    result_dataframe_copy['Sentiment'] = sentiment_list
    result_dataframe_copy['Activity'] = activity_list
    result_dataframe_copy['log_Activity'] = activity_log_list
    dataframe_with_population = get_population_one_area_combined(result_dataframe_copy, census_dict=census_dict)
    final_dataframe = get_median_income_one_area_combined(dataframe_with_population, census_dict=census_dict)
    if 'south_horizons' in check_area_name:  # For South Horizons & Lei Tung, do not consider the last two months
        final_dataframe = final_dataframe.loc[~final_dataframe['Time'].isin(['2018_11', '2018_12'])]
    return final_dataframe


def build_regress_dataframe_for_one_station_seperate(treatment_dataframe, control_dataframe,
                                                     station_open_month_start, station_open_month_end,
                                                     open_year_plus_month, tpu_info_dataframe,
                                                     check_window_value=0, check_area_name=None):
    """
    Build the dataframe for one influenced area
    :param treatment_dataframe: the tweet dataframe for treatment area
    :param control_dataframe: the tweet dataframe for control area
    :param station_open_month_start: the starting time of the month when the studied station opens
    :param station_open_month_end: the ending time of the month when the studied station opens
    :param open_year_plus_month: the month plus year information
    :param tpu_info_dataframe: the dataframe saving the census data for each tpu setting
    :param check_window_value: the window size for DID analysis
    :param check_area_name: the name of the study area
    :return: a pandas dataframe which could be used for the following DID analysis
    """
    # check the date
    assert open_year_plus_month in ['2016_10', '2016_12']
    result_dataframe = pd.DataFrame()
    # build the T_i_t variable
    ones_list = [1] * treatment_dataframe.shape[0]
    treatment_dataframe['T_i_t'] = ones_list
    zeros_list = [0] * control_dataframe.shape[0]
    control_dataframe['T_i_t'] = zeros_list
    # build the post variable
    treatment_dataframe['Post'] = treatment_dataframe.apply(
        lambda row: add_post_variable(row['hk_time'], opening_start_date=station_open_month_start,
                                      opening_end_date=station_open_month_end, check_window=check_window_value,
                                      consider_lag_effect=False), axis=1)
    control_dataframe['Post'] = control_dataframe.apply(
        lambda row: add_post_variable(row['hk_time'], opening_start_date=station_open_month_start,
                                      opening_end_date=station_open_month_end, check_window=check_window_value,
                                      consider_lag_effect=False), axis=1)
    # Check the distribution of T_i_t and POST variables
    print('Check the post variable distribution of treatment group: {}'.format(
        Counter(treatment_dataframe['Post'])))
    print('Check the T_i_t variable distribution of treatment group: {}'.format(
        Counter(treatment_dataframe['T_i_t'])))
    print('Check the post variable distribution of control group: {}'.format(
        Counter(control_dataframe['Post'])))
    print('Check the T_i_t variable distribution of control group: {}'.format(
        Counter(control_dataframe['T_i_t'])))

    # Construct the dictionary having the census data for the treatment area and control area
    treatment_dataframe['TPU_cross_sectional'] = treatment_dataframe['TPU_cross_sectional'].astype(str)
    control_dataframe['TPU_cross_sectional'] = control_dataframe['TPU_cross_sectional'].astype(str)
    tpu_info_dataframe['SmallTPU'] = tpu_info_dataframe.apply(lambda row: str(row['SmallTPU']), axis=1)
    census_dict = {}
    for _, row in tpu_info_dataframe.iterrows():
        census_dict[row['SmallTPU']] = [row['population'], row['m_income']]

    # Construct the dataframe for the DID regression analysis
    combined_dataframe = pd.concat([treatment_dataframe, control_dataframe], axis=0)
    combined_dataframe = combined_dataframe.reset_index(drop=True)
    # We don't consider the tweets posted on the open month of the MTR stations
    combined_dataframe_without_not_considered = combined_dataframe.loc[combined_dataframe['Post'] != 'not considered']
    combined_data_copy = combined_dataframe_without_not_considered.copy()
    # combined_data_copy['month_plus_year'] = combined_data_copy.apply(
    #     lambda row: str(int(float(row['year']))) + '_' + str(int(float(row['month']))), axis=1)
    sentiment_dict = {}
    activity_dict = {}
    activity_dict_log = {}
    for _, dataframe in combined_data_copy.groupby(['month_plus_year', 'T_i_t', 'Post', 'TPU_cross_sectional']):
        time = str(list(dataframe['month_plus_year'])[0])
        t_i_t = str(list(dataframe['T_i_t'])[0])
        post = str(list(dataframe['Post'])[0])
        tpu_info = str(list(dataframe['TPU_cross_sectional'])[0])
        sentiment_dict[time + '+' + t_i_t + '+' + post + '+' + tpu_info] = pos_percent_minus_neg_percent(dataframe)
        activity_dict[time + '+' + t_i_t + '+' + post + '+' + tpu_info] = dataframe.shape[0]
        activity_dict_log[time + '+' + t_i_t + '+' + post + '+' + tpu_info] = np.log(dataframe.shape[0])
    result_dataframe_copy = result_dataframe.copy()
    t_i_t_list, tpu_info_list = [], []
    time_list, post_list = [], []
    sentiment_list = []
    activity_list, activity_log_list = [], []
    for key in list(sentiment_dict.keys()):
        # don't consider the tweets posted in 2016_10(for Whampoa and Ho Man Tin) or 2016_12(for other stations)
        info_list = key.split('+')
        if info_list[0] != open_year_plus_month:
            time_list.append(info_list[0])
            t_i_t_list.append(int(info_list[1]))
            post_list.append(int(info_list[2]))
            tpu_info_list.append(info_list[3])
            sentiment_list.append(sentiment_dict[key])
            activity_list.append(activity_dict[key])
            activity_log_list.append(activity_dict_log[key])
    result_dataframe_copy['Time'] = time_list
    result_dataframe_copy['month'] = result_dataframe_copy.apply(lambda row: int(row['Time'][5:]), axis=1)
    result_dataframe_copy['T_i_t'] = t_i_t_list
    result_dataframe_copy['SmallTPU'] = tpu_info_list
    result_dataframe_copy['Post'] = post_list
    result_dataframe_copy['Sentiment'] = sentiment_list
    result_dataframe_copy['Activity'] = activity_list
    result_dataframe_copy['log_Activity'] = activity_log_list
    dataframe_with_population = get_population_one_area_seperate(result_dataframe_copy, census_dict=census_dict)
    final_dataframe = get_median_income_one_area_seperate(dataframe_with_population, census_dict=census_dict)
    if 'south_horizons' in check_area_name:  # For South Horizons & Lei Tung, do not consider the last two months
        final_dataframe = final_dataframe.loc[~final_dataframe['Time'].isin(['2018_11', '2018_12'])]
    return final_dataframe


def output_did_result(ols_model, variable_list: list, time_window):
    """
    Create a pandas dataframe saving the DID regression analysis result
    :param ols_model: a linear model containing the regression result.
    type: statsmodels.regression.linear_model.RegressionResultsWrapper
    :param variable_list: a list of interested variable names
    :param time_window: the time window
    :return: a pandas dataframe saving the regression coefficient, pvalues, standard errors, aic,
    number of observations, adjusted r squared
    """
    coef_dict = ols_model.params.to_dict()  # coefficient dictionary
    std_error_dict = ols_model.bse.to_dict()  # standard error dictionary
    pval_dict = ols_model.pvalues.to_dict()  # pvalues dictionary
    num_observs = np.int(ols_model.nobs)  # number of observations
    aic_val = round(ols_model.aic, 2)  # aic value
    adj_rsqured = round(ols_model.rsquared_adj, 3)  # adjusted rsqured
    info_index = ['Num', 'AIC', 'Adjusted R2']
    index_list = variable_list + info_index

    for variable in variable_list:
        assert variable in coef_dict, 'Something wrong with variable name!'

    coef_vals = []

    for variable in variable_list:
        coef_val = coef_dict[variable]
        std_val = std_error_dict[variable]
        p_val = pval_dict[variable]
        if p_val <= 0.01:
            coef_vals.append('{}***({})'.format(round(coef_val, 4), round(std_val, 3)))
        elif 0.01 < p_val <= 0.05:
            coef_vals.append('{}**({})'.format(round(coef_val, 4), round(std_val, 3)))
        elif 0.05 < p_val <= 0.1:
            coef_vals.append('{}*({})'.format(round(coef_val, 4), round(std_val, 3)))
        else:
            coef_vals.append('{}({})'.format(round(coef_val, 4), round(std_val, 3)))

    coef_vals.extend([num_observs, aic_val, adj_rsqured])

    result_data = pd.DataFrame()
    if time_window == 0:
        result_colname = 'All'
    else:
        result_colname = '{}-month'.format(time_window)
    result_data[result_colname] = coef_vals
    result_data_reindex = result_data.set_index(pd.Index(index_list))

    return result_data_reindex


def conduct_combined_did_analysis(kwun_tong_treatment_dataframe, kwun_tong_control_dataframe,
                                  south_horizons_treatment_dataframe, south_horizons_control_dataframe,
                                  ocean_park_treatment_dataframe, ocean_park_control_dataframe,
                                  dataframe_saving_path, filename, tpu_info_data,
                                  check_window_value=0, for_sentiment=False):
    longitudinal_dataframe = build_regress_data_three_areas_combined(
        kwun_tong_treatment=kwun_tong_treatment_dataframe,
        kwun_tong_control=kwun_tong_control_dataframe,
        south_horizons_treatment=south_horizons_treatment_dataframe,
        south_horizons_control=south_horizons_control_dataframe, ocean_park_treatment=ocean_park_treatment_dataframe,
        ocean_park_control=ocean_park_control_dataframe, tpu_info_dataframe=tpu_info_data,
        check_window_value=check_window_value,
        sentiment_did=for_sentiment)
    longitudinal_dataframe.to_csv(os.path.join(dataframe_saving_path, filename))
    if for_sentiment:
        # For the combined setting...
        # 'Sentiment ~ T_i_t:Post+C(T_i_t)+C(Time)+Population_log+Median_Income_log'
        # 'Sentiment ~ T_i_t:Post+T_i_t+Post+Population_log+Median_Income_log'
        # For the seperate setting...# Don't use this setting, since the tweet num of some TPUs is not big enough
        # 'Sentiment ~ T_i_t:Post+C(TPU)+C(Time)+Population_log+Median_Income_log'
        reg_combined_sentiment = smf.ols(
            'Sentiment ~ T_i_t:Post+T_i_t+Post+Population_log+Median_Income_log+C(Area_name)+C(Time)',
            longitudinal_dataframe).fit()
        print('----The sentiment did result-----')
        print(reg_combined_sentiment.summary())
        result_dataframe = output_did_result(reg_combined_sentiment,
                                             variable_list=['T_i_t:Post', 'T_i_t', 'Post', 'Population_log',
                                                            'Median_Income_log'],
                                             time_window=check_window_value)
    else:
        # For the combined setting
        # 'log_Activity ~ T_i_t:Post+C(T_i_t)+C(Time)+Population_log+Median_Income_log'
        # 'log_Activity ~ T_i_t:Post+T_i_t+Post+Population_log+Median_Income_log'
        # For the seperate setting
        # 'log_Activity ~ T_i_t:Post+C(TPU)+C(Time)+Population_log+Median_Income_log'
        reg_combined_activity = smf.ols(
            'log_Activity ~ T_i_t:Post+T_i_t+Post+Population_log+Median_Income_log+C(Area_name)+C(Time)',
            longitudinal_dataframe).fit()
        print('----The activity did result-----')
        print(reg_combined_activity.summary())
        result_dataframe = output_did_result(reg_combined_activity,
                                             variable_list=['T_i_t:Post', 'T_i_t', 'Post', 'Population_log',
                                                            'Median_Income_log'],
                                             time_window=check_window_value)
    print(result_dataframe)
    return result_dataframe


def conduct_did_analysis_one_area(treatment_considered_dataframe, control_considered_dataframe, opening_start_date,
                                  opening_end_date, open_year_month, window_size_value, tpu_info_data,
                                  file_path, filename):
    constructed_dataframe = build_regress_dataframe_for_one_station_combined(
        treatment_dataframe=treatment_considered_dataframe,
        control_dataframe=control_considered_dataframe,
        station_open_month_start=opening_start_date,
        station_open_month_end=opening_end_date, tpu_info_dataframe=tpu_info_data,
        open_year_plus_month=open_year_month,
        check_window_value=window_size_value,
        check_area_name=filename)
    constructed_dataframe.to_csv(os.path.join(file_path, filename), encoding='utf-8')
    # For the combined setting
    # 'Sentiment ~ T_i_t:Post+C(T_i_t)+C(Time)+Population_log+Median_Income_log'
    # 'Sentiment ~ T_i_t:Post+T_i_t+Post+Population_log+Median_Income_log'
    # 'log_Activity ~ T_i_t:Post+C(T_i_t)+C(Time)+Population_log+Median_Income_log'
    # 'log_Activity ~ T_i_t:Post+T_i_t+Post+Population_log+Median_Income_log'
    # For the seperate setting
    # 'Sentiment ~ T_i_t:Post+C(SmallTPU)+C(Time)+Population_log+Median_Income_log'
    # 'log_Activity ~ T_i_t:Post+C(SmallTPU)+C(Time)+Population_log+Median_Income_log'
    combined_sentiment = smf.ols(
        'Sentiment ~ T_i_t:Post+T_i_t+Post+Population_log+Median_Income_log+C(Time)',
        constructed_dataframe).fit()
    combined_activity = smf.ols(
        'log_Activity ~ T_i_t:Post+T_i_t+Post+Population_log+Median_Income_log+C(Time)',
        constructed_dataframe).fit()
    result_sent = output_did_result(combined_sentiment,
                                    variable_list=['T_i_t:Post', 'T_i_t', 'Post', 'Population_log',
                                                   'Median_Income_log'],
                                    time_window=window_size_value)
    result_act = output_did_result(combined_activity,
                                   variable_list=['T_i_t:Post', 'T_i_t', 'Post', 'Population_log',
                                                  'Median_Income_log'],
                                   time_window=window_size_value)
    print('----The sentiment did result-----')
    print(combined_sentiment.summary())
    print('----The activity did result-----')
    print(combined_activity.summary())
    print('-------------------------------------------------------\n')
    return result_sent, result_act


if __name__ == '__main__':
    path = os.path.join(data_paths.tweet_combined_path, 'longitudinal_tpus')

    # # Before the first revision
    # kwun_tong_line_treatment_tpu_set = {'243', '245', '236', '213'}
    # kwun_tong_line_control_tpu_set = {'247', '234', '242', '212', '235'}
    # south_horizons_lei_tung_treatment_tpu_set = {'174'}
    # south_horizons_lei_tung_control_tpu_set = {'172', '182'}
    # ocean_park_wong_chuk_hang_treatment_tpu_set = {'175'}
    # ocean_park_wong_chuk_hang_control_tpu_set = {'184', '183', '182'}

    kwun_tong_line_treatment_tpu_set = {'236', '243', '245'}
    kwun_tong_line_control_tpu_set = {'247', '234', '242', '212', '235'}
    south_horizons_lei_tung_treatment_tpu_set = {'174'}
    south_horizons_lei_tung_control_tpu_set = {'172', '181', '182'}
    ocean_park_wong_chuk_hang_treatment_tpu_set = {'175', '176'}
    ocean_park_wong_chuk_hang_control_tpu_set = {'184', '183', '182', '181'}

    # load the tpu info dataframe
    tpu_info_data = pd.read_excel(os.path.join(data_paths.tpu_info_path, 'final_data_2011.xlsx'))
    # Load the users not visitors
    users_not_visitors = np.load(os.path.join(
        data_paths.transit_non_transit_compare_code_path, 'users_not_visitors.npy'), allow_pickle=True).item()

    print('Load the treatment and control groups in three areas...')
    kwun_tong_line_treatment_dataframe = build_dataframe_based_on_set(datapath=path,
                                                                      tpu_set=kwun_tong_line_treatment_tpu_set,
                                                                      selected_user_set=users_not_visitors)
    kwun_tong_line_control_dataframe = build_dataframe_based_on_set(datapath=path,
                                                                    tpu_set=kwun_tong_line_control_tpu_set,
                                                                    selected_user_set=users_not_visitors)
    south_horizons_treatment_dataframe = build_dataframe_based_on_set(datapath=path,
                                                                      tpu_set=south_horizons_lei_tung_treatment_tpu_set,
                                                                      selected_user_set=users_not_visitors)
    south_horizons_control_dataframe = build_dataframe_based_on_set(datapath=path,
                                                                    tpu_set=south_horizons_lei_tung_control_tpu_set,
                                                                    selected_user_set=users_not_visitors)
    ocean_park_treatment_dataframe = build_dataframe_based_on_set(datapath=path,
                                                                  tpu_set=ocean_park_wong_chuk_hang_treatment_tpu_set,
                                                                  selected_user_set=users_not_visitors)
    ocean_park_control_dataframe = build_dataframe_based_on_set(datapath=path,
                                                                tpu_set=ocean_park_wong_chuk_hang_control_tpu_set,
                                                                selected_user_set=users_not_visitors)
    # Output the number of tweets and number of unique social media users for each study area
    print('Whampoa & Ho Man Tin...')
    utils.number_of_tweet_user(kwun_tong_line_treatment_dataframe, print_values=True)
    utils.number_of_tweet_user(kwun_tong_line_control_dataframe, print_values=True)
    print('South Horizons & Lei Tung...')
    utils.number_of_tweet_user(south_horizons_treatment_dataframe, print_values=True)
    utils.number_of_tweet_user(south_horizons_control_dataframe, print_values=True)
    print('Ocean Park & Wong Chuk Hang...')
    utils.number_of_tweet_user(ocean_park_treatment_dataframe, print_values=True)
    utils.number_of_tweet_user(ocean_park_control_dataframe, print_values=True)
    print('Done!')

    print('************************DID Analysis Starts....************************')
    dataframe_saving_path = os.path.join(data_paths.tweet_combined_path, 'longitudinal_did_analysis_dataframes')
    dataframe_lag_effect_path = os.path.join(data_paths.tweet_combined_path, 'longitudinal_lag_effect_data')
    print('For Sentiment...')
    print('Overall Treatment and Control Comparison for sentiment(12 months)...')
    sent_12_month = conduct_combined_did_analysis(kwun_tong_treatment_dataframe=kwun_tong_line_treatment_dataframe,
                                                  kwun_tong_control_dataframe=kwun_tong_line_control_dataframe,
                                                  south_horizons_treatment_dataframe=south_horizons_treatment_dataframe,
                                                  south_horizons_control_dataframe=south_horizons_control_dataframe,
                                                  ocean_park_treatment_dataframe=ocean_park_treatment_dataframe,
                                                  ocean_park_control_dataframe=ocean_park_control_dataframe,
                                                  dataframe_saving_path=dataframe_saving_path,
                                                  filename='longitudinal_did_dataframe_12_months_sentiment.csv',
                                                  check_window_value=12, for_sentiment=True,
                                                  tpu_info_data=tpu_info_data)
    print('Overall Treatment and Control Comparison for sentiment(18 months)...')
    sent_18_month = conduct_combined_did_analysis(kwun_tong_treatment_dataframe=kwun_tong_line_treatment_dataframe,
                                                  kwun_tong_control_dataframe=kwun_tong_line_control_dataframe,
                                                  south_horizons_treatment_dataframe=south_horizons_treatment_dataframe,
                                                  south_horizons_control_dataframe=south_horizons_control_dataframe,
                                                  ocean_park_treatment_dataframe=ocean_park_treatment_dataframe,
                                                  ocean_park_control_dataframe=ocean_park_control_dataframe,
                                                  dataframe_saving_path=dataframe_saving_path,
                                                  filename='longitudinal_did_dataframe_18_months_sentiment.csv',
                                                  check_window_value=18, for_sentiment=True,
                                                  tpu_info_data=tpu_info_data)
    print('Overall Treatment and Control Comparison(sentiment)...')
    sent_all = conduct_combined_did_analysis(kwun_tong_treatment_dataframe=kwun_tong_line_treatment_dataframe,
                                             kwun_tong_control_dataframe=kwun_tong_line_control_dataframe,
                                             south_horizons_treatment_dataframe=south_horizons_treatment_dataframe,
                                             south_horizons_control_dataframe=south_horizons_control_dataframe,
                                             ocean_park_treatment_dataframe=ocean_park_treatment_dataframe,
                                             ocean_park_control_dataframe=ocean_park_control_dataframe,
                                             dataframe_saving_path=dataframe_saving_path,
                                             filename='longitudinal_did_dataframe_all_sentiment.csv',
                                             check_window_value=0,
                                             for_sentiment=True, tpu_info_data=tpu_info_data)
    sent_result_data = pd.concat([sent_12_month, sent_18_month, sent_all], axis=1)
    sent_result_data.to_excel(os.path.join(data_paths.did_result_path, 'overall_sentiment_did_combined_tit_post.xlsx'),
                              encoding='utf-8')
    print('For Activity...')
    print('Overall Treatment and Control Comparison for activity(12 months)...')
    act_12_month = conduct_combined_did_analysis(kwun_tong_treatment_dataframe=kwun_tong_line_treatment_dataframe,
                                                 kwun_tong_control_dataframe=kwun_tong_line_control_dataframe,
                                                 south_horizons_treatment_dataframe=south_horizons_treatment_dataframe,
                                                 south_horizons_control_dataframe=south_horizons_control_dataframe,
                                                 ocean_park_treatment_dataframe=ocean_park_treatment_dataframe,
                                                 ocean_park_control_dataframe=ocean_park_control_dataframe,
                                                 dataframe_saving_path=dataframe_saving_path,
                                                 filename='longitudinal_did_dataframe_12_months_activity.csv',
                                                 check_window_value=12, for_sentiment=False,
                                                 tpu_info_data=tpu_info_data)
    act_18_month = conduct_combined_did_analysis(kwun_tong_treatment_dataframe=kwun_tong_line_treatment_dataframe,
                                                 kwun_tong_control_dataframe=kwun_tong_line_control_dataframe,
                                                 south_horizons_treatment_dataframe=south_horizons_treatment_dataframe,
                                                 south_horizons_control_dataframe=south_horizons_control_dataframe,
                                                 ocean_park_treatment_dataframe=ocean_park_treatment_dataframe,
                                                 ocean_park_control_dataframe=ocean_park_control_dataframe,
                                                 dataframe_saving_path=dataframe_saving_path,
                                                 filename='longitudinal_did_dataframe_18_months_activity.csv',
                                                 check_window_value=18, for_sentiment=False,
                                                 tpu_info_data=tpu_info_data)
    print('Overall Treatment and Control Comparison(activity)...')
    act_all = conduct_combined_did_analysis(kwun_tong_treatment_dataframe=kwun_tong_line_treatment_dataframe,
                                            kwun_tong_control_dataframe=kwun_tong_line_control_dataframe,
                                            south_horizons_treatment_dataframe=south_horizons_treatment_dataframe,
                                            south_horizons_control_dataframe=south_horizons_control_dataframe,
                                            ocean_park_treatment_dataframe=ocean_park_treatment_dataframe,
                                            ocean_park_control_dataframe=ocean_park_control_dataframe,
                                            dataframe_saving_path=dataframe_saving_path,
                                            filename='longitudinal_did_dataframe_all_activity.csv',
                                            check_window_value=0,
                                            for_sentiment=False, tpu_info_data=tpu_info_data)
    act_result_data = pd.concat([act_12_month, act_18_month, act_all], axis=1)
    act_result_data.to_excel(os.path.join(data_paths.did_result_path, 'overall_activity_did_combined_tit_post.xlsx'),
                             encoding='utf-8')

    print('Cope with the three areas seperately...')
    print('---------------------Kwun Tong Line---------------------------')
    print('For 12 months....')
    kwun_tong_did_sent_12, kwun_tong_did_act_12 = conduct_did_analysis_one_area(
        treatment_considered_dataframe=kwun_tong_line_treatment_dataframe,
        control_considered_dataframe=kwun_tong_line_control_dataframe,
        opening_start_date=october_1_start, opening_end_date=october_31_end,
        open_year_month='2016_10', window_size_value=12, file_path=dataframe_saving_path,
        filename='kwun_tong_did_12_months.csv', tpu_info_data=tpu_info_data)
    print('For 18 months...')
    kwun_tong_did_sent_18, kwun_tong_did_act_18 = conduct_did_analysis_one_area(
        treatment_considered_dataframe=kwun_tong_line_treatment_dataframe,
        control_considered_dataframe=kwun_tong_line_control_dataframe,
        opening_start_date=october_1_start, opening_end_date=october_31_end,
        open_year_month='2016_10', window_size_value=18, file_path=dataframe_saving_path,
        filename='kwun_tong_did_18_months.csv', tpu_info_data=tpu_info_data)
    print('For all combined did analysis....')
    kwun_tong_did_sent_all, kwun_tong_did_act_all = conduct_did_analysis_one_area(
        treatment_considered_dataframe=kwun_tong_line_treatment_dataframe,
        control_considered_dataframe=kwun_tong_line_control_dataframe,
        opening_start_date=october_1_start, opening_end_date=october_31_end,
        open_year_month='2016_10', window_size_value=0, file_path=dataframe_saving_path,
        filename='kwun_tong_did_all_months.csv', tpu_info_data=tpu_info_data)
    kwun_tong_sent_did_combined = pd.concat([kwun_tong_did_sent_12, kwun_tong_did_sent_18,
                                             kwun_tong_did_sent_all], axis=1)
    kwun_tong_act_did_combined = pd.concat([kwun_tong_did_act_12, kwun_tong_did_act_18,
                                            kwun_tong_did_act_all], axis=1)
    kwun_tong_sent_did_combined.to_excel(os.path.join(data_paths.did_result_path,
                                                      'kwun_tong_sent_did_combined_tit_post.xlsx'))
    kwun_tong_act_did_combined.to_excel(os.path.join(data_paths.did_result_path,
                                                     'kwun_tong_act_did_combined_tit_post.xlsx'))
    print('-------------------------------------------------------\n')

    print('---------------------South Horizons & Lei Tung---------------------------')
    print('For 12 months....')
    south_horizons_did_sent_12, south_horizons_did_act_12 = conduct_did_analysis_one_area(
        treatment_considered_dataframe=south_horizons_treatment_dataframe,
        control_considered_dataframe=south_horizons_control_dataframe,
        opening_start_date=december_1_start, opening_end_date=december_31_end,
        open_year_month='2016_12', window_size_value=12, file_path=dataframe_saving_path,
        filename='south_horizons_did_12_months.csv', tpu_info_data=tpu_info_data)
    print('For 18 months....')
    south_horizons_did_sent_18, south_horizons_did_act_18 = conduct_did_analysis_one_area(
        treatment_considered_dataframe=south_horizons_treatment_dataframe,
        control_considered_dataframe=south_horizons_control_dataframe,
        opening_start_date=december_1_start, opening_end_date=december_31_end,
        open_year_month='2016_12', window_size_value=18, file_path=dataframe_saving_path,
        filename='south_horizons_did_18_months.csv', tpu_info_data=tpu_info_data)
    print('For all combined did analysis....')
    south_horizons_did_sent_all, south_horizons_did_act_all = conduct_did_analysis_one_area(
        treatment_considered_dataframe=south_horizons_treatment_dataframe,
        control_considered_dataframe=south_horizons_control_dataframe,
        opening_start_date=december_1_start, opening_end_date=december_31_end,
        open_year_month='2016_12', window_size_value=0, file_path=dataframe_saving_path,
        filename='south_horizons_did_all.csv', tpu_info_data=tpu_info_data)
    south_horizons_sent_did_combined = pd.concat([south_horizons_did_sent_12, south_horizons_did_sent_18,
                                                  south_horizons_did_sent_all], axis=1)
    south_horizons_act_did_combined = pd.concat([south_horizons_did_act_12, south_horizons_did_act_18,
                                                 south_horizons_did_act_all], axis=1)
    south_horizons_sent_did_combined.to_excel(os.path.join(data_paths.did_result_path,
                                                           'south_horizons_sent_did_combined_tit_post.xlsx'))
    south_horizons_act_did_combined.to_excel(os.path.join(data_paths.did_result_path,
                                                          'south_horizons_act_did_combined_tit_post.xlsx'))
    print('-------------------------------------------------------\n')

    print('---------------------Ocean Park & Wong Chuk Hang---------------------------')
    print('For 12 months....')
    ocean_park_did_sent_12, ocean_park_did_act_12 = conduct_did_analysis_one_area(
        treatment_considered_dataframe=ocean_park_treatment_dataframe,
        control_considered_dataframe=ocean_park_control_dataframe,
        opening_start_date=december_1_start, opening_end_date=december_31_end,
        open_year_month='2016_12', window_size_value=12, file_path=dataframe_saving_path,
        filename='ocean_park_did_12_months.csv', tpu_info_data=tpu_info_data)
    print('For 18 months....')
    ocean_park_did_sent_18, ocean_park_did_act_18 = conduct_did_analysis_one_area(
        treatment_considered_dataframe=ocean_park_treatment_dataframe,
        control_considered_dataframe=ocean_park_control_dataframe,
        opening_start_date=december_1_start, opening_end_date=december_31_end,
        open_year_month='2016_12', window_size_value=18, file_path=dataframe_saving_path,
        filename='ocean_park_did_18_months.csv', tpu_info_data=tpu_info_data)
    print('For all combined did analysis....')
    ocean_park_did_sent_all, ocean_park_did_act_all = conduct_did_analysis_one_area(
        treatment_considered_dataframe=ocean_park_treatment_dataframe,
        control_considered_dataframe=ocean_park_control_dataframe,
        opening_start_date=december_1_start, opening_end_date=december_31_end,
        open_year_month='2016_12', window_size_value=0, file_path=dataframe_saving_path,
        filename='ocean_park_did_all.csv', tpu_info_data=tpu_info_data)
    ocean_park_sent_did_combined = pd.concat([ocean_park_did_sent_12, ocean_park_did_sent_18,
                                              ocean_park_did_sent_all], axis=1)
    ocean_park_act_did_combined = pd.concat([ocean_park_did_act_12, ocean_park_did_act_18,
                                             ocean_park_did_act_all], axis=1)
    ocean_park_sent_did_combined.to_excel(
        os.path.join(data_paths.did_result_path, 'ocean_park_sent_did_combined_tit_post.xlsx'))
    ocean_park_act_did_combined.to_excel(
        os.path.join(data_paths.did_result_path, 'ocean_park_act_did_combined_tit_post.xlsx'))
    print('-------------------------------------------------------\n')
