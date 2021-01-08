# -*- coding: utf-8 -*-
"""
We below illustrate on a very simple example, the general use of our package to
generate data, run cost-based selection methods and analyze the results.
We use the Barabási-Albert model with four different settings as described in
the paper associated to this package.
"""

from cost_based_selection import data_generation
from cost_based_selection import preprocessing_utils
from cost_based_selection import cost_based_methods
from cost_based_selection import cost_based_analysis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

### Data generation

# Define the number simulated data per model and the number of nodes per network
num_sim_model = 50
num_nodes = 20

# Generate the reference tables
dfModIndex, dfSummaries, \
dfIsDisc, dfTimes = data_generation.BA_ref_table(num_sim_model = num_sim_model,
                                                 num_nodes = num_nodes)

# The outputs generated are pandas DataFrames containing the model indexes, 
# the summary statistics values, the nature of the summaries (i.e. discrete or
# continuous numerical features), the times in seconds to generate each entry
# and each summary.

### Preprocessing steps

# We remove redundant features if any, and reorder the features by average time
dfModIndex, dfSummaries, \
dfIsDisc, dfTimes = preprocessing_utils.drop_redundant_features(dfModIndex,
                                                                dfSummaries,
                                                                dfIsDisc,
                                                                dfTimes)

dfModIndex, dfSummaries, \
dfIsDisc, dfTimes = preprocessing_utils.data_reordering_by_avg_cost(dfModIndex,
                                                                    dfSummaries,
                                                                    dfIsDisc,
                                                                    dfTimes)

# Extract the indexes of the noise features
noise_idx = preprocessing_utils.noise_position(dfSummaries)

### Compute the average computational times and normalize 
### (between 0 and 1 and to sum to 1)

avg_cost_vec = preprocessing_utils.compute_avg_cost(dfTimes)

### Launch a unique cost-based selection method with a given penalization parameter.
### For illustrative purposes we only use the penalized Joint Mutual Information strategy.

# Convert the data to the correct format
X = np.array(dfSummaries)
y = dfModIndex.modIndex.tolist()
is_disc = dfIsDisc.iloc[0,:].tolist()

# Split the reference table in training and validation set
(X_train, X_val, \
 y_train, y_val) = train_test_split(X, y, test_size=0.5, random_state=123, stratify=y)

# Select the best network features on the training set
ranking, *rest = cost_based_methods.JMI(X = X_train, y = y_train, is_disc = is_disc, 
                                        cost_vec = avg_cost_vec, cost_param = 1)

# ranking contains the ranked feature indexes in decreasing order of importance
dfSummaries.columns[ranking]

### When using multiple penalization parameters, we advice using the multi_* version of the methods

grid_cost_param = [0, 1, 2, 3, 4]

dfRank = cost_based_methods.multi_JMI(X = X_train, y = y_train, is_disc = is_disc,
                                      cost_vec = avg_cost_vec, cost_param_vec = grid_cost_param)

### Given a classifier, we can plot the k-fold cross-validation accuracy
### on a validation set and visualize the trade-off between total cost reduction
### and classification accuracy. This is useful to determine the penalization parameter.

# Number of best features to keep
subset_size = 15

### For the JMI method

# Using a 10-nearest-neighbors classifier with arguments specified in dict_args
classifier = KNeighborsClassifier
dict_args = {'n_neighbors':10}

avg_accuracy, std_accuracy, \
total_cost, prop_noise = \
cost_based_analysis.accuracy_classifier_plot(dfPen_Ranking = dfRank, X_val = X_val,
                                             y_val = y_val, cost_vec = avg_cost_vec,
                                             noise_idx = noise_idx, subset_size = subset_size,
                                             classifier_func = classifier, args_classifier = dict_args,
                                             num_fold = 3, save_name = None)