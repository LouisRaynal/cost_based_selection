# -*- coding: utf-8 -*-
"""
This file runs the cost-based selection methods presented in our paper using a 
precomputed reference table related to the four BA models to classify.
Each simulated network contains num_nodes = 1000 nodes.
The performance of each methods is evaluated thanks to the classification
accuracy of an k-nearest-neighbors classifier, and a Support Vector Machine
classifier.
"""

import multiprocessing
import numpy as np
import pandas as pd

from cost_based_selection import preprocessing_utils
from cost_based_selection import cost_based_methods
from cost_based_selection import cost_based_analysis
from pkg_resources import resource_filename
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Specify the number of CPU cores to use when parallelism can be adopted,
# to change if needed
num_cores = max(1, multiprocessing.cpu_count() - 1)

##### Load the data

# We here load the data available in the data folder of the package
dfModIndex = pd.read_csv(filepath_or_buffer = resource_filename('cost_based_selection', 'data/modIndex_10000sim_1000nodes_BA.csv'))
dfSummaries = pd.read_csv(filepath_or_buffer = resource_filename('cost_based_selection', 'data/summaries_10000sim_1000nodes_BA.csv'))
dfIsDisc = pd.read_csv(filepath_or_buffer = resource_filename('cost_based_selection', 'data/isDisc_10000sim_1000nodes_BA.csv'))
dfTimes = pd.read_csv(filepath_or_buffer = resource_filename('cost_based_selection', 'data/times_10000sim_1000nodes_BA.csv'))


# We loaded the data that we used in our paper, if you would like to use another
# reference table, please uncomment the corresponding lines below and specify
# the folder directory where the data are saved in loc_data

#loc_data = ''
#dfModIndex = pd.read_csv(filepath_or_buffer = loc_data+'modIndex_10000sim_1000nodes_BA.csv')
#dfSummaries = pd.read_csv(filepath_or_buffer = loc_data+'summaries_10000sim_1000nodes_BA.csv')
#dfIsDisc = pd.read_csv(filepath_or_buffer = loc_data+'isDisc_10000sim_1000nodes_BA.csv')
#dfTimes = pd.read_csv(filepath_or_buffer = loc_data+'times_10000sim_1000nodes_BA.csv')


##### Preprocessing steps

# Drop the redundant features if any
dfModIndex, dfSummaries, dfIsDisc, dfTimes = preprocessing_utils.drop_redundant_features(dfModIndex,
                                                                                         dfSummaries,
                                                                                         dfIsDisc,
                                                                                         dfTimes)

# Reorder the features by average computation time to compute each summary statistic
dfModIndex, dfSummaries, dfIsDisc, dfTimes = preprocessing_utils.data_reordering_by_avg_cost(dfModIndex,
                                                                                             dfSummaries,
                                                                                             dfIsDisc,
                                                                                             dfTimes)

# Extract the indices of the noise features
noise_idx = preprocessing_utils.noise_position(dfSummaries)


##### Compute the feature cost vector, i.e. the average computational times, then normalized.

avg_cost_vec = preprocessing_utils.compute_avg_cost(dfTimes)


##### Train the different cost-based feature selection methods, for different penalization parameters
##### to determine the full summary statistic rankings

# Convert the data to the correct format
X = np.array(dfSummaries)
y = dfModIndex.modIndex.tolist()
is_disc = dfIsDisc.iloc[0,:].tolist()

# Split the reference table in training and testing/validation set
val_size = 0.50 # 1-val_size is the proportion of data used for feature selection
random_seed = 123
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=val_size, random_state=random_seed, stratify=y)

# Grid of penalization parameters for which the rankings will be computed
grid_cost_param = list(np.arange(0, 100, 0.02))
# Grid for the weighted random forest methods
grid_cost_param_wRF = list(np.arange(0, 2, 0.002))


# For the mRMR method
dfRank_mRMR = cost_based_methods.multi_mRMR(X = X_train, y = y_train, is_disc = is_disc, 
                                            cost_vec = avg_cost_vec, cost_param_vec = grid_cost_param,
                                            num_features_to_select = None, random_seed = 123,
                                            num_cores = num_cores)

# For the JMI method
dfRank_JMI = cost_based_methods.multi_JMI(X = X_train, y = y_train, is_disc = is_disc,
                                          cost_vec = avg_cost_vec, cost_param_vec = grid_cost_param,
                                          num_features_to_select = None, random_seed = 123,
                                          num_cores = num_cores)

# For the JMIM method
dfRank_JMIM = cost_based_methods.multi_JMIM(X = X_train, y = y_train, is_disc = is_disc,
                                            cost_vec = avg_cost_vec, cost_param_vec = grid_cost_param,
                                            num_features_to_select = None, random_seed = 123,
                                            num_cores = num_cores)

# For the classic reliefF method
dfRank_reliefF_classic = cost_based_methods.multi_reliefF(X = X_train, y = y_train, 
                                                          cost_vec = avg_cost_vec, cost_param_vec = grid_cost_param,
                                                          num_neighbors = 10, num_features_to_select = None,
                                                          proximity = "distance")

# For the reliefF method when neighbors are deduced with the Breiman's similarity matrix
dfRank_reliefF_rfprox = cost_based_methods.multi_reliefF(X = X_train, y = y_train, 
                                                         cost_vec = avg_cost_vec, cost_param_vec = grid_cost_param,
                                                         num_neighbors = 10, num_features_to_select = None,
                                                         proximity = "rf prox", min_samples_leaf = 100, n_estimators = 500)

# For the weighted random forest method, with importance based on the mean decrease of impurity (MDI)
dfRank_wRF_impurity = cost_based_methods.multi_weighted_rf_importance(X = X_train, y = y_train,
                                                                      cost_vec = avg_cost_vec, cost_param_vec = grid_cost_param_wRF,
                                                                      num_features_to_select = None, imp_type = "impurity",
                                                                      min_samples_leaf = 1, n_estimators = 500)

# For the weighted random forest method, with importance based on the mean decrease of accuracy after permutation (MDA)
dfRank_wRF_permutation = cost_based_methods.multi_weighted_rf_importance(X = X_train, y = y_train,
                                                                         cost_vec = avg_cost_vec, cost_param_vec = grid_cost_param_wRF,
                                                                         num_features_to_select = None, imp_type = "permutation",
                                                                         min_samples_leaf = 1, n_estimators = 500)

# For the penalized random forest importance, with importance based on MDI
dfRank_penRF_impurity = cost_based_methods.multi_pen_rf_importance(X = X_train, y = y_train,
                                                                   cost_vec = avg_cost_vec, cost_param_vec = grid_cost_param,
                                                                   num_features_to_select = None, imp_type = "impurity",
                                                                   min_samples_leaf = 1, n_estimators = 500)

# For the penalized random forest importance, with importance based on MDA
dfRank_penRF_permutation = cost_based_methods.multi_pen_rf_importance(X = X_train, y = y_train,
                                                                      cost_vec = avg_cost_vec, cost_param_vec = grid_cost_param,
                                                                      num_features_to_select = None, imp_type = "permutation",
                                                                      min_samples_leaf = 1, n_estimators = 500)


##### Plot the evolution of classification accuracy, total cost and proportion of noise (out of 4)
##### depending on the penalization parameter used. We select the 15 best summary statistics.

# Number of best summary statistics to use:
subset_size = 15

### For the mRMR method

# Using a 10-nearest-neighbors classifier
classifier = KNeighborsClassifier
dict_args = {'n_neighbors':10, 'n_jobs':num_cores}
avg_accuracy_mRMR_knn, std_accuracy_mRMR_knn, total_cost_mRMR_knn, prop_noise_mRMR_knn = cost_based_analysis.accuracy_classifier_plot(dfPen_Ranking = dfRank_mRMR, 
                                                                                                                                      X_val = X_val, 
                                                                                                                                      y_val = y_val,
                                                                                                                                      cost_vec = avg_cost_vec,
                                                                                                                                      noise_idx = noise_idx,
                                                                                                                                      subset_size = subset_size,
                                                                                                                                      classifier_func = classifier, 
                                                                                                                                      args_classifier = dict_args,
                                                                                                                                      num_fold = 3, save_name = "evol_mRMR_kNN_BA.pdf",
                                                                                                                                      random_seed = 123)

# Using an SVM classifier
classifier = SVC
avg_accuracy_mRMR_SVM, std_accuracy_mRMR_SVM, total_cost_mRMR_SVM, prop_noise_mRMR_SVM = cost_based_analysis.accuracy_classifier_plot(dfPen_Ranking = dfRank_mRMR,
                                                                                                                                      X_val = X_val,
                                                                                                                                      y_val = y_val,
                                                                                                                                      cost_vec = avg_cost_vec,
                                                                                                                                      noise_idx = noise_idx,
                                                                                                                                      subset_size = subset_size,
                                                                                                                                      classifier_func = classifier, 
                                                                                                                                      args_classifier = dict(),
                                                                                                                                      num_fold = 3, save_name = "evol_mRMR_SVM_BA.pdf",
                                                                                                                                      random_seed = 123)


### For the JMI method

# Using a 10-nearest-neighbors classifier
classifier = KNeighborsClassifier
dict_args = {'n_neighbors':10, 'n_jobs':num_cores}
avg_accuracy_JMI_knn, std_accuracy_JMI_knn, total_cost_JMI_knn, prop_noise_JMI_knn = cost_based_analysis.accuracy_classifier_plot(dfPen_Ranking = dfRank_JMI,
                                                                                                                                  X_val = X_val, 
                                                                                                                                  y_val = y_val,
                                                                                                                                  cost_vec = avg_cost_vec,
                                                                                                                                  noise_idx = noise_idx,
                                                                                                                                  subset_size = subset_size,
                                                                                                                                  classifier_func = classifier, 
                                                                                                                                  args_classifier = dict_args,
                                                                                                                                  num_fold = 3, save_name = "evol_JMI_kNN_BA.pdf",
                                                                                                                                  random_seed = 123)

# Using an SVM classifier
classifier = SVC
avg_accuracy_JMI_SVM, std_accuracy_JMI_SVM, total_cost_JMI_SVM, prop_noise_JMI_SVM = cost_based_analysis.accuracy_classifier_plot(dfPen_Ranking = dfRank_JMI,
                                                                                                                                  X_val = X_val,
                                                                                                                                  y_val = y_val,
                                                                                                                                  cost_vec = avg_cost_vec,
                                                                                                                                  noise_idx = noise_idx,
                                                                                                                                  subset_size = subset_size,
                                                                                                                                  classifier_func = classifier, 
                                                                                                                                  args_classifier = dict(),
                                                                                                                                  num_fold = 3, save_name = "evol_JMI_SVM_BA.pdf",
                                                                                                                                  random_seed = 123)

### For the JMIM method

# Using a 10-nearest-neighbors classifier
classifier = KNeighborsClassifier
dict_args = {'n_neighbors':10, 'n_jobs':num_cores}
avg_accuracy_JMIM_knn, std_accuracy_JMIM_knn, total_cost_JMIM_knn, prop_noise_JMIM_knn = cost_based_analysis.accuracy_classifier_plot(dfPen_Ranking = dfRank_JMIM,
                                                                                                                                      X_val = X_val, 
                                                                                                                                      y_val = y_val,
                                                                                                                                      cost_vec = avg_cost_vec,
                                                                                                                                      noise_idx = noise_idx,
                                                                                                                                      subset_size = subset_size,
                                                                                                                                      classifier_func = classifier, 
                                                                                                                                      args_classifier = dict_args,
                                                                                                                                      num_fold = 3, save_name = "evol_JMIM_kNN_BA.pdf",
                                                                                                                                      random_seed = 123)

# Using an SVM classifier
classifier = SVC
avg_accuracy_JMIM_SVM, std_accuracy_JMIM_SVM, total_cost_JMIM_SVM, prop_noise_JMIM_SVM = cost_based_analysis.accuracy_classifier_plot(dfPen_Ranking = dfRank_JMIM,
                                                                                                                                      X_val = X_val,
                                                                                                                                      y_val = y_val,
                                                                                                                                      cost_vec = avg_cost_vec,
                                                                                                                                      noise_idx = noise_idx,
                                                                                                                                      subset_size = subset_size,
                                                                                                                                      classifier_func = classifier, 
                                                                                                                                      args_classifier = dict(),
                                                                                                                                      num_fold = 3, save_name = "evol_JMIM_SVM_BA.pdf",
                                                                                                                                      random_seed = 123)

### For the classic reliefF method

# Using a 10-nearest-neighbors classifier
classifier = KNeighborsClassifier
dict_args = {'n_neighbors':10, 'n_jobs':num_cores}
avg_accuracy_reliefF_classic_knn, std_accuracy_reliefF_classic_knn, total_cost_reliefF_classic_knn, prop_noise_reliefF_classic_knn = cost_based_analysis.accuracy_classifier_plot(dfPen_Ranking = dfRank_reliefF_classic,
                                                                                                                                                                                  X_val = X_val, 
                                                                                                                                                                                  y_val = y_val,
                                                                                                                                                                                  cost_vec = avg_cost_vec,
                                                                                                                                                                                  noise_idx = noise_idx,
                                                                                                                                                                                  subset_size = subset_size,
                                                                                                                                                                                  classifier_func = classifier, 
                                                                                                                                                                                  args_classifier = dict_args,
                                                                                                                                                                                  num_fold = 3, save_name = "evol_reliefF_classic_kNN_BA.pdf",
                                                                                                                                                                                  random_seed = 123)

# Using an SVM classifier
classifier = SVC
avg_accuracy_reliefF_classic_SVM, std_accuracy_reliefF_classic_SVM, total_cost_reliefF_classic_SVM, prop_noise_reliefF_classic_SVM = cost_based_analysis.accuracy_classifier_plot(dfPen_Ranking = dfRank_reliefF_classic,
                                                                                                                                                                                  X_val = X_val,
                                                                                                                                                                                  y_val = y_val,
                                                                                                                                                                                  cost_vec = avg_cost_vec,
                                                                                                                                                                                  noise_idx = noise_idx,
                                                                                                                                                                                  subset_size = subset_size,
                                                                                                                                                                                  classifier_func = classifier, 
                                                                                                                                                                                  args_classifier = dict(),
                                                                                                                                                                                  num_fold = 3, save_name = "evol_reliefF_classic_SVM_BA.pdf",
                                                                                                                                                                                  random_seed = 123)

### For the reliefF method using random forest proximity

# Using a 10-nearest-neighbors classifier
classifier = KNeighborsClassifier
dict_args = {'n_neighbors':10, 'n_jobs':num_cores}
avg_accuracy_reliefF_classic_knn, std_accuracy_reliefF_rf_knn, total_cost_reliefF_rf_knn, prop_noise_reliefF_rf_knn = cost_based_analysis.accuracy_classifier_plot(dfPen_Ranking = dfRank_reliefF_rfprox,
                                                                                                                                                                   X_val = X_val,
                                                                                                                                                                   y_val = y_val,
                                                                                                                                                                   cost_vec = avg_cost_vec,
                                                                                                                                                                   noise_idx = noise_idx,
                                                                                                                                                                   subset_size = subset_size,
                                                                                                                                                                   classifier_func = classifier, 
                                                                                                                                                                   args_classifier = dict_args,
                                                                                                                                                                   num_fold = 3, save_name = "evol_reliefF_rf_kNN_BA.pdf",
                                                                                                                                                                   random_seed = 123)

# Using an SVM classifier
classifier = SVC
avg_accuracy_reliefF_rf_SVM, std_accuracy_reliefF_rf_SVM, total_cost_reliefF_rf_SVM, prop_noise_reliefF_rf_SVM = cost_based_analysis.accuracy_classifier_plot(dfPen_Ranking = dfRank_reliefF_rfprox,
                                                                                                                                                              X_val = X_val,
                                                                                                                                                              y_val = y_val,
                                                                                                                                                              cost_vec = avg_cost_vec,
                                                                                                                                                              noise_idx = noise_idx,
                                                                                                                                                              subset_size = subset_size,
                                                                                                                                                              classifier_func = classifier, 
                                                                                                                                                              args_classifier = dict(),
                                                                                                                                                              num_fold = 3, save_name = "evol_reliefF_rf_SVM_BA.pdf",
                                                                                                                                                              random_seed = 123)

### For the weighted random forest using the MDI

# Using a 10-nearest-neighbors classifier
classifier = KNeighborsClassifier
dict_args = {'n_neighbors':10, 'n_jobs':num_cores}
avg_accuracy_wRF_impurity_knn, std_accuracy_wRF_impurity_knn, total_cost_wRF_impurity_knn, prop_noise_wRF_impurity_knn = cost_based_analysis.accuracy_classifier_plot(dfPen_Ranking = dfRank_wRF_impurity,
                                                                                                                                                                      X_val = X_val,
                                                                                                                                                                      y_val = y_val,
                                                                                                                                                                      cost_vec = avg_cost_vec,
                                                                                                                                                                      noise_idx = noise_idx,
                                                                                                                                                                      subset_size = subset_size,
                                                                                                                                                                      classifier_func = classifier, 
                                                                                                                                                                      args_classifier = dict_args,
                                                                                                                                                                      num_fold = 3, save_name = "evol_wRF_impurity_kNN_BA.pdf",
                                                                                                                                                                      random_seed = 123)

# Using an SVM classifier
classifier = SVC
avg_accuracy_wRF_impurity_SVM, std_accuracy_wRF_impurity_SVM, total_cost_wRF_impurity_SVM, prop_noise_wRF_impurity_SVM = cost_based_analysis.accuracy_classifier_plot(dfPen_Ranking = dfRank_wRF_impurity,
                                                                                                                                                                      X_val = X_val,
                                                                                                                                                                      y_val = y_val,
                                                                                                                                                                      cost_vec = avg_cost_vec,
                                                                                                                                                                      noise_idx = noise_idx,
                                                                                                                                                                      subset_size = subset_size,
                                                                                                                                                                      classifier_func = classifier, 
                                                                                                                                                                      args_classifier = dict(),
                                                                                                                                                                      num_fold = 3, save_name = "evol_wRF_impurity_SVM_BA.pdf",
                                                                                                                                                                      random_seed = 123)

### For the weighted random forest using the MDA

# Using a 10-nearest-neighbors classifier
classifier = KNeighborsClassifier
dict_args = {'n_neighbors':10, 'n_jobs':num_cores}
avg_accuracy_wRF_permutation_knn, std_accuracy_wRF_permutation_knn, total_cost_wRF_permutation_knn, prop_noise_wRF_permutation_knn = cost_based_analysis.accuracy_classifier_plot(dfPen_Ranking = dfRank_wRF_permutation,
                                                                                                                                                                                  X_val = X_val,
                                                                                                                                                                                  y_val = y_val,
                                                                                                                                                                                  cost_vec = avg_cost_vec,
                                                                                                                                                                                  noise_idx = noise_idx,
                                                                                                                                                                                  subset_size = subset_size,
                                                                                                                                                                                  classifier_func = classifier, 
                                                                                                                                                                                  args_classifier = dict_args,
                                                                                                                                                                                  num_fold = 3, save_name = "evol_wRF_permutation_kNN_BA.pdf",
                                                                                                                                                                                  random_seed = 123)

# Using an SVM classifier
classifier = SVC
avg_accuracy_wRF_permutation_SVM, std_accuracy_wRF_permutation_SVM, total_cost_wRF_permutation_SVM, prop_noise_wRF_permutation_SVM = cost_based_analysis.accuracy_classifier_plot(dfPen_Ranking = dfRank_wRF_permutation,
                                                                                                                                                                                  X_val = X_val,
                                                                                                                                                                                  y_val = y_val,
                                                                                                                                                                                  cost_vec = avg_cost_vec,
                                                                                                                                                                                  noise_idx = noise_idx,
                                                                                                                                                                                  subset_size = subset_size,
                                                                                                                                                                                  classifier_func = classifier, 
                                                                                                                                                                                  args_classifier = dict(),
                                                                                                                                                                                  num_fold = 3, save_name = "evol_wRF_permutation_SVM_BA.pdf",
                                                                                                                                                                                  random_seed = 123)

### For the penalized random forest importance based on the MDI

# Using a 10-nearest-neighbors classifier
classifier = KNeighborsClassifier
dict_args = {'n_neighbors':10, 'n_jobs':num_cores}
avg_accuracy_penrf_impurity_knn, std_accuracy_penrf_impurity_knn, total_cost_penrf_impurity_knn, prop_noise_penrf_impurity_knn = cost_based_analysis.accuracy_classifier_plot(dfPen_Ranking = dfRank_penRF_impurity,
                                                                                                                                                                              X_val = X_val,
                                                                                                                                                                              y_val = y_val,
                                                                                                                                                                              cost_vec = avg_cost_vec,
                                                                                                                                                                              noise_idx = noise_idx,
                                                                                                                                                                              subset_size = subset_size,
                                                                                                                                                                              classifier_func = classifier, 
                                                                                                                                                                              args_classifier = dict_args,
                                                                                                                                                                              num_fold = 3, save_name = "evol_penRF_impurity_kNN_BA.pdf",
                                                                                                                                                                              random_seed = 123)

# Using an SVM classifier
classifier = SVC
avg_accuracy_penrf_impurity_SVM, std_accuracy_penrf_impurity_SVM, total_cost_penrf_impurity_SVM, prop_noise_penrf_impurity_SVM = cost_based_analysis.accuracy_classifier_plot(dfPen_Ranking = dfRank_penRF_impurity,
                                                                                                                                                                              X_val = X_val,
                                                                                                                                                                              y_val = y_val,
                                                                                                                                                                              cost_vec = avg_cost_vec,
                                                                                                                                                                              noise_idx = noise_idx,
                                                                                                                                                                              subset_size = subset_size,
                                                                                                                                                                              classifier_func = classifier, 
                                                                                                                                                                              args_classifier = dict(),
                                                                                                                                                                              num_fold = 3, save_name = "evol_penRF_impurity_SVM_BA.pdf",
                                                                                                                                                                              random_seed = 123)


### For the penalized random forest importance based on the MDA

# Using a 10-nearest-neighbors classifier
classifier = KNeighborsClassifier
dict_args = {'n_neighbors':10, 'n_jobs':num_cores}
avg_accuracy_penrf_permutation_knn, std_accuracy_penrf_permutation_knn, total_cost_penrf_permutation_knn, prop_noise_penrf_permutation_knn = cost_based_analysis.accuracy_classifier_plot(dfPen_Ranking = dfRank_penRF_permutation,
                                                                                                                                                                                          X_val = X_val,
                                                                                                                                                                                          y_val = y_val,
                                                                                                                                                                                          cost_vec = avg_cost_vec,
                                                                                                                                                                                          noise_idx = noise_idx,
                                                                                                                                                                                          subset_size = subset_size,
                                                                                                                                                                                          classifier_func = classifier, 
                                                                                                                                                                                          args_classifier = dict_args,
                                                                                                                                                                                          num_fold = 3, save_name = "evol_penRF_permutation_kNN_BA.pdf",
                                                                                                                                                                                          random_seed = 123)

# Using an SVM classifier
classifier = SVC
avg_accuracy_penrf_permutation_SVM, std_accuracy_penrf_permutation_SVM, total_cost_penrf_permutation_SVM, prop_noise_penrf_permutation_SVM = cost_based_analysis.accuracy_classifier_plot(dfPen_Ranking = dfRank_penRF_permutation,
                                                                                                                                                                                          X_val = X_val,
                                                                                                                                                                                          y_val = y_val,
                                                                                                                                                                                          cost_vec = avg_cost_vec,
                                                                                                                                                                                          noise_idx = noise_idx,
                                                                                                                                                                                          subset_size = subset_size,
                                                                                                                                                                                          classifier_func = classifier, 
                                                                                                                                                                                          args_classifier = dict(),
                                                                                                                                                                                          num_fold = 3, save_name = "evol_penRF_permutation_SVM_BA.pdf",
                                                                                                                                                                                          random_seed = 123)

