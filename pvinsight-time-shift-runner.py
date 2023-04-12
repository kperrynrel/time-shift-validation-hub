"""
Runner script for assessing the time shift validation algorithm. In this
script, the following occurs:
    1. Pull down all of the metadata associated with the data sets
    2. Loop through all metadata cases, pull down the associated data, and
    run the associated submission on it
    3. Aggregate the results for the entire data set and generate assessment 
    metrics. Assessment metrics will vary based on the type of analysis being
    run. For this analysis, the following is calculated:
        1. Mean Absolute Error between predicted time shift series and ground
        truth time series (in minutes)
        2. Average run time for each data set (in seconds)
    4. Further aggregate performance metrics into visualizations:
        -Distribution Histograms
        -Scatter plots
      This section will be dependent on the type of analysis being run.
"""

import pandas as pd
import ast
import os
from importlib import import_module
import inspect
import time
from collections import ChainMap
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
import json

if __name__ == '__main__':
    # Load in the module that we're going to test on.
    module_to_import = 'pvanalytics-cpd-module'
    # Generate list for us to store all of our results for the module
    results_list = list()
    # Load in data set that we're going to analyze. 
    # System metadata: This CSV represents the system_metadata table, which is
    # a master table for associated system metadata (system_id, name, azimuth,
    # tilt, etc.) 
    system_metadata = pd.read_csv("./data/system_metadata.csv")
    # File metadata: This file represents the file_metadata table, which is
    # the master table for files associated with different tests (az-tilt, 
    # time shifts, degradation, etc.). Contains file name, associated file
    # information (sampling frequency, specific test, timezone, etc) as well
    # as ground truth information to test against in the validation_dictionary
    # field
    file_metadata = pd.read_csv("./data/file_metadata.csv")
    # Validation tests: This file represents the validation_tests table,
    # which is the master table associated with each of the tests run on the
    # PVInsight Validation Hub. This table contains information on test type
    # (example: time_shifts, az_tilt_detection, etc), as well as function name
    # for each test type, performance metrics outputs (how to assess
    # test performance), as well as expected function outputs and function
    # output types (in order). This provides a standardized template of
    # what to expect (input, output, naming conventions) for each test
    validation_tests = pd.read_csv("./data/validation_tests.csv")
    # File category link: This file represents the file_category_link table,
    # which links specific files in the file_metadata table to categories in
    # the validation_tests table. This table exists specifically to allow for
    # many-to-many relationships, where one file can link to multiple
    # categories/tests, and multiple categories/tests can link to multiple
    # files. This table exists solely to link these two tables together
    # when performing testing.
    file_category_link = pd.read_csv("./data/file_test_link.csv")
    # Link the above tables together to get all of the files associated
    # with the time_shift category in the validation_tests table.
    time_shift_test_information = dict(validation_tests[
        validation_tests['category_name'] == 'time_shifts'].iloc[0])
    # Get the associated metrics we're supposed to calculate
    performance_metrics = ast.literal_eval(time_shift_test_information[
            'performance_metrics'])
    # Get all of the linked files for time shift analysis via a series
    # of dataframe joins
    associated_file_ids = list(file_category_link[
        file_category_link['category_id'] == 
        time_shift_test_information['category_id']]['file_id'])
    associated_files = file_metadata[file_metadata['file_id'].isin(
        associated_file_ids)]
    # Loop through each file and generate predictions 
    for index, row in associated_files.iterrows():
        # Get file_name, which will be pulled from database or S3 for
        # each analysis
        file_name = row['file_name']
        # Get associated system ID
        system_id = row['system_id']
        # Get all of the associated metadata for the particular file based
        # on its system ID. This metadata will be passed in via kwargs for
        # any necessary arguments
        associated_metadata = dict(system_metadata[
            system_metadata['system_id'] == system_id].iloc[0])
        # Create master dictionary of all possible function kwargs
        kwargs_dict = dict(ChainMap(dict(row), associated_metadata))
        # Get the name of the function we want to import associated with this
        # test
        function_name = time_shift_test_information['function_name']
        # Now that we've collected all of the information associated with the
        # test, let's read in the file as a pandas dataframe (this data
        # would most likely be stored in an S3 bucket)
        time_series = pd.read_csv(os.path.join("./data/file_data/", file_name),
                                index_col = 0,
                                parse_dates = True).squeeze()
        time_series = time_series.asfreq(str(row['data_sampling_frequency']) + "T")
        # Read in the associated validation time series (this would act as a
        # fixture or similar, and validation data would be stored in an
        # associated folder on S3 or similar)
        ground_truth_series = pd.read_csv(
            os.path.join("./data/validation_data/", file_name),
            index_col = 0,
            parse_dates = True).squeeze()
        # Import designated module via importlib
        module = import_module(module_to_import)
        function = getattr(module, function_name)
        function_parameters = list(inspect.signature(function).parameters)
        # Filter the kwargs dictionary based on required function params
        kwargs = dict((k, kwargs_dict[k]) for k in function_parameters
                      if k in kwargs_dict)
        # Time function execution if 'run_time' is in performance metrics
        # list
        if 'run_time' in performance_metrics:
            start_time = time.time()
            time_shift_series = function(time_series, **kwargs)
            end_time = time.time()
            function_run_time = (end_time - start_time)
        else:
            time_shift_series = function(time_series, **kwargs)
        # Run routine for all of the performance metrics and append 
        # results to the dictionary
        results_dictionary = dict()
        results_dictionary['file_name'] = file_name
        for metric in performance_metrics:
            if metric == 'run_time':
                results_dictionary[metric] = function_run_time
            if metric == 'mean_absolute_error':
                mae = mean_absolute_error(ground_truth_series,
                                          time_shift_series)
                results_dictionary[metric] = mae
        results_list.append(results_dictionary)
    # Convert the results to a pandas dataframe and perform all of the
    # post-processing in the script
    results_df = pd.DataFrame(results_list)        
    # Build out the final processed results:
    #   1) Public reporting: mean MAE, mean run time, etc.
    #   2) Private reporting: graphics and tables split by different factors
    # First get mean value for all the performance metrics and save (this will
    # be saved to a public metrics dictionary)
    public_metrics_dict = dict()
    for metric in performance_metrics:
        mean_value = results_df[metric].mean()
        public_metrics_dict['mean_' + metric] = mean_value
    # Write public metric information to a JSON
    with open('./results/time-shift-public-metrics.json', 'w') as fp:
        json.dump(public_metrics_dict, fp)
    # Now generate private results. These will be more specific to the
    # type of analysis being run as results will be color-coded by certain
    # parameters. These params will be available as columns in the 
    # 'associated_files' dataframe
    color_code_params = ['data_sampling_frequency', 'issue']
    results_df_private = pd.merge(results_df, 
                                  associated_files[['file_name'] +
                                                   color_code_params],
                                  on = 'file_name')
    for param in color_code_params:
        # Mean absolute error histogram
        sns.displot(results_df_private,
                    x='mean_absolute_error', hue=param,
                    multiple="stack", bins=30)
        plt.gca().set_yscale('log')
        plt.title('MAE by ' + str(param))
        # Save to a folder
        plt.savefig(os.path.join("./results",
                                 str(param) + '_mean_absolute_error.png'))
        plt.close()
        plt.clf()
        # Run time histogram
        sns.displot(results_df_private,
                    x='run_time', hue=param,
                    multiple="stack", bins=30)
        plt.title('Run time (s) by ' + str(param))
        # Save to a folder
        plt.savefig(os.path.join("./results",
                                 str(param) + '_run_time.png'))
        plt.close()
        plt.clf()