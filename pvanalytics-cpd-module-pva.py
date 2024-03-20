"""
PVAnalytics-CPD based module. This module will be uploaded by the user
and tested using the data sets accordingly. 
"""

import pandas as pd
import daytime
from pvanalytics.quality.outliers import zscore
from pvanalytics.quality import gaps
import pvlib
import ruptures as rpt
import numpy as np
import matplotlib.pyplot as plt


def shifts_ruptures(event_times, reference_times,
                    period_min=15,
                    shift_min=15,
                    prediction_penalty=20,
                    zscore_cutoff=2,
                    bottom_quantile_threshold=0,
                    top_quantile_threshold=.5):
    """Identify time shifts using the ruptures library.

    Compares the event time in the expected time zone (`reference_times`)
    with the actual event time in `event_times`.

    The Binary Segmentation changepoint detection method is applied to the
    difference between `event_times` and `reference_times`. For each period
    between change points the mode of the difference is rounded to a
    multiple of `shift_min` and returned as the time-shift for all days in
    that period.

    Parameters
    ----------
    event_times : Series
        Time of an event in minutes since midnight. Should be a time series
        of integers with a single value per day. Typically the time mid-way
        between sunrise and sunset.
    reference_times : Series
        Time of event in minutes since midnight for each day in the expected
        timezone. For example, passing solar transit time in a fixed offset
        time zone can be used to detect daylight savings shifts when it is
        unknown whether or not `event_times` is in a fixed offset time zone.
    period_min : int, default 15
        Minimum number of days between shifts. Must be less than or equal to
        the number of days in `event_times`. [days]
        Increasing this parameter will make the result less sensitive to
        transient shifts. For example if your intent is to find and correct
        daylight savings time shifts passing `period_min=60` can give good
        results while excluding shorter periods that appear shifted.
    shift_min : int, default 15
        Minimum shift amount in minutes. All shifts are rounded to a multiple
        of `shift_min`. [minutes]
    prediction_penalty : int, default 13
        Penalty used in assessing change points.
        See :py:meth:`ruptures.detection.Pelt.predict` for more information.
    zscore_cutoff=2,
    bottom_quantile_threshold=.5,
    top_quantile_threshold=1

    Returns
    -------
    shifted : Series
        Boolean series indicating whether there appears to be a time
        shift on that day.
    shift_amount : Series
        Time shift in minutes for each day in `event_times`. These times
        can be used to shift the data into the same time zone as
        `reference_times`.

    Raises
    ------
    ValueError
        If the number of days in `event_times` is less than `period_min`.

    Notes
    -----
    Timestamped data from monitored PV systems may not always be localized
    to a consistent timezone. In some cases, data is timestamped with
    local time that may or may not be adjusted for daylight savings time
    transitions. This function helps detect issues of this sort, by
    detecting points where the time of some daily event (e.g. solar noon)
    changes significantly with respect to a reference time for the event.
    If the data's timestamps have not been adjusted for daylight savings
    transitions, the time of day at solar noon will change by roughly 60
    minutes in the days before and after the transition.

    To use this changepoint detection method to determine if your data's
    timestamps involve daylight savings transitions, first reduce your PV
    system data (irradiance or power) to a daily time series, with each
    point being the observed midday time in minutes. For example, if
    sunrise and sunset are inferred from the PV system data, the midday
    time can be inferred as the average of each day's sunrise and sunset
    time of day. To establish the expected midday time, calculate solar
    transit time in time of day.

    Derived from the PVFleets QA project.

    References
    -------
    .. [1] Perry K., Meyers B., and Muller, M. "Survey of Time Shift
        Detection Algorithms for Measured PV Data", 2023 PV Reliability
        Workshop (PVRW).
    """
    try:
        import ruptures
    except ImportError:
        raise ImportError("time.shifts_ruptures() requires ruptures.")

    if period_min > len(event_times):
        raise ValueError("period_min exceeds number of days in event_times")
    # Drop timezone information. At this point there is one value per day
    # so the timezone is irrelevant. Get the time difference in minutes.
    time_diff = \
        (event_times.tz_localize(None) -
         reference_times.tz_localize(None))
    # Get the index before removing NaN's
    time_diff_orig_index = time_diff.index
    # # Remove any outliers that may skew the results
    # zscore_outlier_mask = zscore(time_diff, zmax=zscore_cutoff,
    #                              nan_policy='omit')
    # time_diff.loc[zscore_outlier_mask] = np.nan
    # Remove NaN's from the time_diff series, because NaN's screw up the
    # ruptures prediction
    time_diff = time_diff.dropna()
    # Run changepoint detection to find breaks
    break_points = ruptures.Binseg(model='rbf',
                                   jump=1,
                                   min_size=period_min).fit_predict(
                                       signal=time_diff.values,
                                       pen=prediction_penalty)
    # Make sure the entire series is covered by the intervals between
    # the breakpoints that were identified above. This means adding a
    # breakpoint at the beginning of the series (0) and at the end if
    # one does not already exist.
    break_points.insert(0, 0)
    if break_points[-1] != len(time_diff):
        break_points.append(len(time_diff))
    # Go throguh each time shift segment and perform the following steps:
    # 1) Remove the outliers from each segment using a z-score filter
    # 2) Remove any cases that are not within the bottom and top quantile
    #    parameters for the segment. This helps to remove more erroneous
    #    data.
    # 3) Take the mean of each segment and round it to the nearest shift_min
    #    multiplier
    shift_amount = time_diff.copy()
    for index in range(len(break_points) - 1):
        segment = time_diff[break_points[index]:
                            break_points[index + 1]]
        segment = segment[~zscore(segment, zmax=zscore_cutoff,
                                  nan_policy='omit')]
        segment = segment[
            (segment >= segment.quantile(bottom_quantile_threshold)) &
            (segment <= segment.quantile(top_quantile_threshold))]
        shift_amount.iloc[break_points[index]: break_points[index + 1]] = \
            shift_min * round(float(segment.mean())/shift_min)
    # Update the shift_amount series with the original time series
    shift_amount = shift_amount.reindex(time_diff_orig_index).ffill()
    # localize the shift_amount series to the timezone of the input
    shift_amount = shift_amount.tz_localize(event_times.index.tz)
    return shift_amount != 0, shift_amount

    
def detect_time_shifts(time_series,
                       latitude, longitude,
                       data_sampling_frequency):
    """
    Master function for testing for time shifts in a series and returning
    time-shifted periods
    
    """
    # Save the dates of the time series index for reindexing at the end
    date_index = pd.Series(time_series.index.date).drop_duplicates()
    # Data pre-processing:
    # 1) Removal of frozen/stuck data
    # 2) Removal of data periods with low data 'completeness'
    # 3) Removal of negative data
    # 4) Removal of outliers via Hampel + outlier filter
    # Trim based on frozen data values
    stale_data_mask = gaps.stale_values_diff(time_series)
    time_series = time_series[~stale_data_mask]
    time_series = time_series.asfreq(str(data_sampling_frequency) + 'T')
    # Remove negative data
    time_series = time_series[(time_series >= 0) | (time_series.isna())]
    time_series = time_series.asfreq(str(data_sampling_frequency) + 'T')
    # Remove any outliers via z-score filter
    zscore_outlier_mask = zscore(time_series, zmax=2,
                                 nan_policy='omit')
    time_series = time_series[(~zscore_outlier_mask)]
    time_series = time_series.asfreq(str(data_sampling_frequency) + 'T')
    # Remove any incomplete days (less than 33% data)
    completeness = gaps.complete(time_series,
                                minimum_completeness=0.33)
    time_series = time_series[completeness]
    time_series = time_series.asfreq(str(data_sampling_frequency) + 'T')
    if len(time_series) > 0:
        # Calculate a nighttime offset 
        # Mask daytime periods for the time series
        daytime_mask = daytime.power_or_irradiance(time_series,
                                                   freq=str(data_sampling_frequency) + 'T',
                                                   low_value_threshold=.005)
        # Get the modeled sunrise and sunset time series based on the system's
        # latitude-longitude coordinates
        modeled_sunrise_sunset_df = pvlib.solarposition.sun_rise_set_transit_spa(
            time_series.index,
            latitude, longitude)
        modeled_sunrise_sunset_df.index = modeled_sunrise_sunset_df.index.date
        modeled_sunrise_sunset_df = modeled_sunrise_sunset_df.drop_duplicates()
        # Calculate the midday point between sunrise and sunset for each day
        # in the modeled irradiance series
        modeled_midday_series = modeled_sunrise_sunset_df['sunrise'] + \
                                (modeled_sunrise_sunset_df['sunset'] - \
                                 modeled_sunrise_sunset_df['sunrise']) / 2
        modeled_midday_series.index = pd.to_datetime(modeled_midday_series.index)
        #Generate the sunrise, sunset, and halfway pts for the data stream
        sunrise_series = daytime.get_sunrise(daytime_mask)
        sunset_series = daytime.get_sunset(daytime_mask)
        midday_series = (sunrise_series + ((sunset_series - sunrise_series) / 2)).resample("D").mean()
        #Compare the data stream's daily halfway point to the modeled halfway point
        midday_diff_series = (modeled_midday_series.tz_localize(None) - 
                              midday_series.tz_localize(None)).dt.total_seconds() / 60
        # Express both midday series as minutes after midnight
        modeled_midday_series = (modeled_midday_series - modeled_midday_series.dt.normalize()).dt.total_seconds() / 60
        midday_series = (midday_series - midday_series.dt.normalize()).dt.total_seconds() / 60
        __, time_shift_series = shifts_ruptures(event_times = midday_series,
                                                reference_times = modeled_midday_series)
        time_shift_series = -1 * time_shift_series
        midday_diff_series = midday_diff_series.dropna()
        midday_diff_series.plot()
        plt.show()
        plt.close()
        return time_shift_series
    else:
        return pd.Series(0, index=date_index)