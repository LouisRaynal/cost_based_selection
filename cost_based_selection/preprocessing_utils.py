# -*- coding: utf-8 -*-
""" Utility and preprocessing functions. """

import pandas as pd
import numpy as np

def drop_redundant_features(dfModIndex, dfSummaries, dfIsDisc, dfTimes):
    """ Function to drop the redundant features of the reference table.
    
    This function drops the features that contains the same value no matter the
    model used. This can be the case with the BA models.

    Args:
        dfModIndex (pandas.core.frame.DataFrame):
            the panda DataFrame containing the model indexes. Not necessary
            for this function but it allows to directly input the results of the
            reference table simulation functions.
        dfSummaries (pandas.core.frame.DataFrame):
            the pandas DataFrame containing, for each simulated data (in rows),
            the summary statistic values (in columns).
        dfIsDisc (pandas.core.frame.DataFrame):
            the pandas DataFrame containing in each column the nature of the
            corresponding summary statistic: True for a discrete summary,
            False for a continuous one.
        dfTimes (pandas.core.frame.DataFrame):
            the pandas DataFrame containing, for each simulated data (in rows),
            the time in seconds to compute each summary statistic values
            (in columns).
            
    Returns:
        Same than in input after removing the redundant columns in the
        summary statistic values, nature and times.
            
    """

    nunique = dfSummaries.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique==1].index
    
    # Drop the redundant columns
    dfSummaries_reduced = dfSummaries.drop(cols_to_drop, axis=1)
    dfIsDisc_reduced = dfIsDisc.drop(cols_to_drop, axis=1)
    dfTimes_reduced = dfTimes.drop(cols_to_drop, axis=1)
    
    print("Columns dropped:", list(cols_to_drop))
    
    return dfModIndex, dfSummaries_reduced, dfIsDisc_reduced, dfTimes_reduced

def noise_position(dfSummaries):
    """Utility function to recover the column indices of the noise features.
    
    Args:
        dfSummaries (pandas.core.frame.DataFrame):
            the pandas DataFrame containing, for each simulated data (in rows),
            the summary statistic values (in columns).        
    
    Returns:
        noise_idx (list):
            list containing the indices of the noise features.

    """
    
    noise_idx = [dfSummaries.columns.get_loc('noise_Gauss'),
                 dfSummaries.columns.get_loc('noise_Unif'),
                 dfSummaries.columns.get_loc('noise_Bern'),
                 dfSummaries.columns.get_loc('noise_disc_Unif')]
    
    return noise_idx


def data_reordering_by_avg_cost(dfModIndex, dfSummaries, dfIsDisc, dfTimes):
    """ Reordering of the reference table columns according to the cost.
    
    This function reorders the columns of the simulated data (dfSummaries, dfIsDisc,
    dfTimes) according to the average cost, in increasing order.
    
    Args:
        dfModIndex (pandas.core.frame.DataFrame):
            the panda DataFrame containing the model indexes. Not necessary
            for this function but it allows to directly input the results of the
            reference table simulation functions.
        dfSummaries (pandas.core.frame.DataFrame):
            the pandas DataFrame containing, for each simulated data (in rows),
            the summary statistic values (in columns).
        dfIsDisc (pandas.core.frame.DataFrame):
            the pandas DataFrame containing in each column the nature of the
            corresponding summary statistic: True for a discrete summary,
            False for a continuous one.
        dfTimes (pandas.core.frame.DataFrame):
            the pandas DataFrame containing, for each simulated data (in rows),
            the time in seconds to compute each summary statistic values
            (in columns).
            
    Returns:
        Same than in input after reordering the columns so that low column
        indexes are the features that require low average computational cost, 
        and high indexes the ones that require the highest average cost.
    
    """
    
    avg_cost_vec = dfTimes.apply(np.mean)
    order_idx = np.argsort(avg_cost_vec)
    
    sorted_col_names = list(dfSummaries.columns[order_idx])

    dfSummaries_ordered = dfSummaries[sorted_col_names]
    dfTimes_ordered = dfTimes[sorted_col_names]
    dfIsDisc_ordered = dfIsDisc[sorted_col_names]
    
    return dfModIndex, dfSummaries_ordered, dfIsDisc_ordered, dfTimes_ordered


def data_reordering_identical(dfModIndex_to_order, dfSummaries_to_order, dfIsDisc_to_order, dfTimes_to_order,
                              dfSummaries_ordered):
    """ Reordering of the reference table columns according to the order of a second one.
    
    This function reorders the columns of the simulated data (summaries, is_disc,
    times) according to the order of a second table.
    
    Args:
        dfModIndex_to_order (pandas.core.frame.DataFrame):
            the panda DataFrame containing the model indexes. Not necessary
            for this function but it allows to directly input the results of the
            reference table simulation functions.
        dfSummaries_to_order (pandas.core.frame.DataFrame):
            the pandas DataFrame containing, for each simulated data (in rows),
            the summary statistic values (in columns).
        dfIsDisc_to_order (pandas.core.frame.DataFrame):
            the pandas DataFrame containing in each column the nature of the
            corresponding summary statistic: True for a discrete summary,
            False for a continuous one.
        dfTimes_to_order (pandas.core.frame.DataFrame):
            the pandas DataFrame containing, for each simulated data (in rows),
            the time in seconds to compute each summary statistic values
            (in columns).
        dfSummaries_ordered (pandas.core.frame.DataFrame):
            a pandas DataFrame of an already ordered matrix of summary statistics.
            
    Returns:
        Same than in input (excepted for dfSummaries_ordered) after reordering 
        the columns into the same order than dfSummaries_order.
    
    """
    
    sorted_col_names = list(dfSummaries_ordered.columns)

    dfSummaries_ordered_1 = dfSummaries_to_order[sorted_col_names]
    dfTimes_ordered_1 = dfTimes_to_order[sorted_col_names]
    dfIsDisc_ordered_1 = dfIsDisc_to_order[sorted_col_names]
    
    return dfModIndex_to_order, dfSummaries_ordered_1, dfIsDisc_ordered_1, dfTimes_ordered_1


def compute_avg_cost(dfTimes):
    """
    This function computes the average summary statistic costs, and normalizes
    the resulting values between 0 and 1, and to sum to 1.

    Args:    
        dfTimes (pandas.core.frame.DataFrame):
            the pandas DataFrame containing, for each simulated data (in rows),
            the time in seconds to compute each summary statistic values
            (in columns).
            
    Returns:
        avg_cost (numpy.ndarray):
            a numpy array with one row and as many columns as in dfTimes, 
            where each value is the normalized average summary statistic cost.
            
    """
    
    avg_cost = list(dfTimes.apply(np.mean))
    avg_cost = (np.array(avg_cost) - np.min(avg_cost))/(np.max(avg_cost) - np.min(avg_cost))
    avg_cost = avg_cost/np.sum(avg_cost)
    
    return avg_cost