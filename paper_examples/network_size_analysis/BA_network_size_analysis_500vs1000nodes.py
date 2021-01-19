# -*- coding: utf-8 -*-
"""
In this file, we study the impact of using small networks for summary statistic
selection, to classify larger networks.

First, we study the number of commonly selected features when using small
networks and large networks. We also compute relative areas under the evolution
of the number of common features, depending on the size of the feature subset.

Second, we study the decrease of accuracy obtained when using smaller networks
for the summary statistic selection step, rather than large networks, to
classify large networks.

The classification accuracy is measured with a k-nearest-neighbors classifier, 
and a Support Vector Machine classifier.
"""

import pandas as pd
import multiprocessing
from cost_based_selection import preprocessing_utils
from cost_based_selection import network_size_analysis
from pkg_resources import resource_filename

# Specify the size of the small and large networks
small_size = 500
large_size = 1000

# Specify the number of CPU cores to use when parallelism can be adopted,
# to change if needed
num_cores = max(1, multiprocessing.cpu_count() - 1)

###### Load the data

# We here load the data available in the data folder of the package
dfModIndex_small = pd.read_csv(filepath_or_buffer = resource_filename('cost_based_selection', 'data/modIndex_10000sim_'+str(small_size)+'nodes_BA.csv'))
dfSummaries_small = pd.read_csv(filepath_or_buffer = resource_filename('cost_based_selection', 'data/summaries_10000sim_'+str(small_size)+'nodes_BA.csv'))
dfIsDisc_small = pd.read_csv(filepath_or_buffer = resource_filename('cost_based_selection', 'data/isDisc_10000sim_'+str(small_size)+'nodes_BA.csv'))
dfTimes_small = pd.read_csv(filepath_or_buffer = resource_filename('cost_based_selection', 'data/times_10000sim_'+str(small_size)+'nodes_BA.csv'))

dfModIndex_large = pd.read_csv(filepath_or_buffer = resource_filename('cost_based_selection', 'data/modIndex_10000sim_'+str(large_size)+'nodes_BA.csv'))
dfSummaries_large = pd.read_csv(filepath_or_buffer = resource_filename('cost_based_selection', 'data/summaries_10000sim_'+str(large_size)+'nodes_BA.csv'))
dfIsDisc_large = pd.read_csv(filepath_or_buffer = resource_filename('cost_based_selection', 'data/isDisc_10000sim_'+str(large_size)+'nodes_BA.csv'))
dfTimes_large = pd.read_csv(filepath_or_buffer = resource_filename('cost_based_selection', 'data/times_10000sim_'+str(large_size)+'nodes_BA.csv'))

# We loaded the data that we used in our paper, if you would like to use another
# reference table, please uncomment the corresponding lines below and specify
# the folder directory where the data are saved in loc_data

#loc_data = ''
#
#dfModIndex_small = pd.read_csv(filepath_or_buffer = loc_data+'modIndex_10000sim_'+str(small_size)+'nodes_BA.csv')
#dfSummaries_small = pd.read_csv(filepath_or_buffer = loc_data+'summaries_10000sim_'+str(small_size)+'nodes_BA.csv')
#dfIsDisc_small = pd.read_csv(filepath_or_buffer = loc_data+'isDisc_10000sim_'+str(small_size)+'nodes_BA.csv')
#dfTimes_small = pd.read_csv(filepath_or_buffer = loc_data+'times_10000sim_'+str(small_size)+'nodes_BA.csv')
#
#dfModIndex_large = pd.read_csv(filepath_or_buffer = loc_data+'modIndex_10000sim_'+str(large_size)+'nodes_BA.csv')
#dfSummaries_large = pd.read_csv(filepath_or_buffer = loc_data+'summaries_10000sim_'+str(large_size)+'nodes_BA.csv')
#dfIsDisc_large = pd.read_csv(filepath_or_buffer = loc_data+'isDisc_10000sim_'+str(large_size)+'nodes_BA.csv')
#dfTimes_large = pd.read_csv(filepath_or_buffer = loc_data+'times_10000sim_'+str(large_size)+'nodes_BA.csv')


##### Preprocessing steps

# Drop the redundant features if any
dfModIndex_small, dfSummaries_small, dfIsDisc_small, dfTimes_small = preprocessing_utils.drop_redundant_features(dfModIndex_small, dfSummaries_small, dfIsDisc_small, dfTimes_small)

# Reorder the features by average computation time to compute each summary statistic
dfModIndex_small, dfSummaries_small, dfIsDisc_small, dfTimes_small = preprocessing_utils.data_reordering_by_avg_cost(dfModIndex_small, dfSummaries_small, dfIsDisc_small, dfTimes_small)

# Drop the redundant features if any
dfModIndex_large, dfSummaries_large, dfIsDisc_large, dfTimes_large = preprocessing_utils.drop_redundant_features(dfModIndex_large, dfSummaries_large, dfIsDisc_large, dfTimes_large)

# Reorder according to the order of the reference table with small networks
dfModIndex_large, dfSummaries_large, dfIsDisc_large, dfTimes_large = preprocessing_utils.data_reordering_identical(dfModIndex_large, dfSummaries_large, dfIsDisc_large,
                                                                                                                   dfTimes_large, dfSummaries_small)

##### Analyses: 1 replication

# We provide a function to compare the feature rankings obtained with the two 
# network sizes, see  the documentation of the function
# cost_based_selection.network_size_analysis.common_features_plot.
#
# We also provide a function to analyze the decrease of accuracy
# resulting after obtaining the rankings, see the documentation of the function
# cost_based_selection.network_size_analysis.difference_accuracy_small_large.
#
# Both these functions require to manually determine the different rankings for
# the different feature selection methods described in
# cost_based_selection.cost_based_methods (to use with cost_param = 0).
#
# However, to facilitate the analyses we merge all these steps into the function
# cost_based_selection.network_size_analysis.common_features_difference_accuracy,
# and we now demonstrate its use.

# Specify the subset sizes for which the decrease of accuracy will be evaluated
subset_size_vec = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

is_disc = dfIsDisc_small.iloc[0,:].tolist() # Useful for certain selection methods
# and common when using small or large networks.

dict_common, dict_areas, dict_SVM_acc_dec, dict_knn_acc_dec = network_size_analysis.common_features_difference_accuracy(dfSummaries_small = dfSummaries_small,
                                                                                                                        dfSummaries_large = dfSummaries_large,
                                                                                                                        dfModIndex_small = dfModIndex_small,
                                                                                                                        dfModIndex_large = dfModIndex_large,
                                                                                                                        is_disc = is_disc,
                                                                                                                        subset_size_vec = subset_size_vec,
                                                                                                                        val_size = 0.5,
                                                                                                                        random_seed = 123)


##### Analyses: multiple replications

# Finally we can use this function to perform num_rep replicate analyses,
# where each replicate relies on different partitioning of the reference tables,
# and use the function network_size_analysis.analyze_replication_res to compute 
# quantities of interest over the replicates and plot related graphs.

num_rep = 50

# Run the replicate analyses
replicate_analysis_res = network_size_analysis.replication_common_features_difference_accuracy(dfSummaries_small = dfSummaries_small,
                                                                                               dfSummaries_large = dfSummaries_large,
                                                                                               dfModIndex_small = dfModIndex_small, 
                                                                                               dfModIndex_large = dfModIndex_large, 
                                                                                               is_disc = is_disc,
                                                                                               subset_size_vec = subset_size_vec,
                                                                                               val_size = 0.5,
                                                                                               num_rep = num_rep,
                                                                                               num_cores = num_cores)

# Analyze the results
df_avg_common, df_std_common, df_avg_areas, df_std_areas = network_size_analysis.analyze_replication_res(replication_res = replicate_analysis_res, 
                                                                                                         subset_size_vec = subset_size_vec,
                                                                                                         showfliers = False, save = True,
                                                                                                         plot_reliefF = True)