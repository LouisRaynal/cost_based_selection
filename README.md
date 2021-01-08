# Selection of summary statistics for network model choice with approximate Bayesian computation
We here provide a Python package named `cost_based_selection` associated to our manuscript **Selection of summary statistics for network model choice with approximate Bayesian computation** *by L. Raynal and J.-P. Onnela*.

## Table of contents
* [Description](#description)
* [Installation](#installation)
* [Package outline](#outline)
* [Simple examples](#examples)
* [Authors](#authors)

## Description
This project focuses on cost-based selection of network features. Indeed, when solving network inference problems with approximate Bayesian computation (ABC), because ABC requires simulating a large number of data that are summarized thanks to some features, it is critical to select them to avoid the curse of dimensionality. However, while many features can be used to encode the network structure, their computational complexity can be highly variable and this can quickly create a bottleneck, making the use of ABC even more difficult when working with large network data.

Thanks to cost-based filter selection methods, for classification problems, we take into consideration the computational cost associated to each feature during the feature selection process, to create a balance between total feature cost and classification accuracy.

In our paper we also investigate the benefit of using smaller networks (with fewer nodes than the observed data) for feature selection to classify networks with as many nodes as the observed data. Modules related to this  are also provided.

## Installation

Install with `pip` from a terminal
```shell
pip install git+https://github.com/LouisRaynal/cost_based_selection.git
```

## Package outline

The `cost_based_selection\data` folder contains the simulated data employed in our paper. 
The `paper_examples` folder contains the scripts to reproduce the examples and analyses we performed.
The `simple_examples` folder contains very simple examples to illustrate how to use the various functions of the package.

The Python modules of this package are:
- `summaries.py`: to compute network summary statistics;

- `data_generation.py`: to simulate and compute summarized network data as described in our paper;
- `preprocessing_utils.py`: to reorder data, drop redundant features or identify position of noise features;
- `cost_based_methods`: to obtain feature rankings with cost-based feature selection methods;
- `cost_based_analysis`: to compute and represent the evolution of total cost, classification accuracy, proportion of noise features selected with respect to a grid of cost penalization parameters;
- `network_size_analysis`: to analyze the relevance of using small or large networks for the selection process and its impact on the classification of large networks.

## Simple examples

### Cost-based selection of features

We present below a very simple example on how to use the main functions of this package to perform cost-based feature selection. See also `simple_examples\BA_cost_based_selection_analysis.py`.

First, generate some data, here according to the four settings of the Barabási-Albert models presented in our paper.

```python
from cost_based_selection import data_generation

num_sim_model = 50 	# Number of simulations per model
num_nodes = 20 		# Number of nodes in the networks

dfModIndex, dfSummaries, \
dfIsDisc, dfTimes = data_generation.BA_ref_table(num_sim_model, num_nodes)
```
The outputs generated are pandas DataFrames containing the model indices, the summary statistics values, the types of the summaries (i.e. discrete or continuous numerical features), the times in seconds to generate each entry and each summary. We then apply some preprocessing/utility functions.

```Python
from cost_based_selection import preprocessing_utils

# Remove the redundant features, if any
dfModIndex, dfSummaries, \
dfIsDisc, dfTimes = preprocessing_utils.drop_redundant_features(dfModIndex,
                                                                dfSummaries,
                                                                dfIsDisc,
                                                                dfTimes)
# Identify the noise features
noise_idx = preprocessing_utils.noise_position(dfSummaries)
```

Compute the feature cost vector, here the average CPU time of each feature, that we normalize.

```python
avg_cost_vec = preprocessing_utils.compute_avg_cost(dfTimes)
```

We convert the data to the correct format and keep some for the classifier training/evaluation.

```python
import numpy as np
from sklearn.model_selection import train_test_split

X = np.array(dfSummaries)
y = dfModIndex.modIndex.tolist()
is_disc = dfIsDisc.iloc[0,:].tolist()
(X_train, X_val, \
 y_train, y_val) = train_test_split(X, y, test_size=0.5, random_state=123, stratify=y)
```
We now can obtain a feature ranking given a cost penalization parameter value, or use a list of parameters to obtain each ranking in a pandas DataFrame. We below use the method based on Joint Mutual Information (JMI), but other methods can be used similarly. See the module `cost_based_methods`.

```python
from cost_based_selection import cost_based_methods

# Given a unique cost penalization parameter
ranking, *rest = cost_based_methods.JMI(X = X_train, y = y_train, 
                                        is_disc = is_disc, cost_vec = avg_cost_vec, 
                                        cost_param = 1)
dfSummaries.columns[ranking] # Summaries in decreasing order of importance

# Given a list of cost penalization parameters
grid_cost_param = [0, 1, 2, 3, 4]
dfRank = cost_based_methods.multi_JMI(X = X_train, y = y_train, 
                                      is_disc = is_disc, cost_vec = avg_cost_vec, 
                                      cost_param_vec = grid_cost_param)
```
Finally, using the rankings we can plot the k-fold cross-validation accuracy of a classifier and visualize the trade-off between total cost reduction and classification accuracy depending on the penalization parameters. This is useful to determine a correct value for this latter.

```python
from cost_based_selection import cost_based_analysis
from sklearn.neighbors import KNeighborsClassifier

subset_size = 15 					# Number of best features to keep
classifier = KNeighborsClassifier 	# Classifier
dict_args = {'n_neighbors':10}		# Arguments of the classifier

avg_accuracy, std_accuracy, \
total_cost, prop_noise = \
cost_based_analysis.accuracy_classifier_plot(dfPen_Ranking = dfRank, X_val = X_val,
                                             y_val = y_val, cost_vec = avg_cost_vec,
                                             noise_idx = noise_idx, 
                                             subset_size = subset_size,
                                             classifier_func = classifier,
                                             args_classifier = dict_args,
                                             num_fold = 3, save_name = None)
```

### Network size influence to select features

We also provide a simple code (see `simple_examples\BA_network_size_analysis.py`) to illustrate how to analyze the influence of using feature selection with smaller networks, to then classify larger networks.


## Authors
* **Louis Raynal** - *Harvard T.H. Chan School of Public Health*
* **Jukka-Pekka Onnela** - *Harvard T.H. Chan School of Public Health*
