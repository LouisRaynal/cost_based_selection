# -*- coding: utf-8 -*-
"""
We here illustrate on a very simple example, the general use of our package to
generate data with two different number of nodes, quantify the ability of the
selection methods to identify identical best features in these two settings
(i.e. with small or large networks), and compute the decrease of classification
accuracy to classify networks with the higher number of nodes, when selecting
networks using small rather than large networks.
"""

import multiprocessing
from cost_based_selection import data_generation
from cost_based_selection import preprocessing_utils
from cost_based_selection import network_size_analysis

### Data generation

# Define the number of simulated data per model and the number of nodes per
# network for small and large networks
num_sim_model = 100
small_size = 20
large_size = 50

# Simulate the reference tables
dfModIndex_small, dfSummaries_small, \
dfIsDisc_small, dfTimes_small = data_generation.BA_ref_table(num_sim_model = num_sim_model,
                                                             num_nodes = small_size)
dfModIndex_large, dfSummaries_large, \
dfIsDisc_large, dfTimes_large = data_generation.BA_ref_table(num_sim_model = num_sim_model,
                                                             num_nodes = large_size)

### Preprocessing steps

# Drop the redundant features when using small networks if any
dfModIndex_small, dfSummaries_small, \
dfIsDisc_small, dfTimes_small = preprocessing_utils.drop_redundant_features(dfModIndex_small, 
                                                                            dfSummaries_small, 
                                                                            dfIsDisc_small, 
                                                                            dfTimes_small)

# Reorder the features by average computation time to compute each summary statistic
dfModIndex_small, dfSummaries_small, \
dfIsDisc_small, dfTimes_small = preprocessing_utils.data_reordering_by_avg_cost(dfModIndex_small,
                                                                                dfSummaries_small,
                                                                                dfIsDisc_small,
                                                                                dfTimes_small)

# Drop the redundant features when using large networks if any
dfModIndex_large, dfSummaries_large, \
dfIsDisc_large, dfTimes_large = preprocessing_utils.drop_redundant_features(dfModIndex_large,
                                                                            dfSummaries_large,
                                                                            dfIsDisc_large,
                                                                            dfTimes_large)

# Reorder according to the order of the reference table with small networks
dfModIndex_large, dfSummaries_large, \
dfIsDisc_large, dfTimes_large = preprocessing_utils.data_reordering_identical(dfModIndex_large,
                                                                              dfSummaries_large,
                                                                              dfIsDisc_large,
                                                                              dfTimes_large,
                                                                              dfSummaries_small)

##### Analyses: 1 replication

# We provide a function to compare the feature rankings obtained with the two 
# network sizes, see the documentation of the function
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

# Specify the subset sizes for which the decrease of accuracy will be evaluated.
subset_size_vec = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

is_disc = dfIsDisc_small.iloc[0,:].tolist() # Useful for certain selection methods
# and common when using small or large networks.

dict_common, dict_areas, \
dict_SVM_acc_dec, dict_knn_acc_dec = network_size_analysis.common_features_difference_accuracy(dfSummaries_small = dfSummaries_small,
                                                                                               dfSummaries_large = dfSummaries_large,
                                                                                               dfModIndex_small = dfModIndex_small,
                                                                                               dfModIndex_large = dfModIndex_large,
                                                                                               is_disc = is_disc,
                                                                                               subset_size_vec = subset_size_vec,
                                                                                               val_size = 0.5)

##### Analyses: multiple replications

# Finally, we can use this function to perform num_rep replicate analyses,
# where each replicate relies on different partitioning of the reference tables,
# and use the function network_size_analysis.analyze_replication_res to compute 
# quantities of interest over the replicates and plot related graphs.

num_rep = 5

# Run the replicate analyses
num_cores = max(1, multiprocessing.cpu_count() - 1)

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
                                                                                                         showfliers = False, save = False,
                                                                                                         plot_reliefF = True)