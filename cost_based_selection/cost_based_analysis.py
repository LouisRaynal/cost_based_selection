# -*- coding: utf-8 -*-
"""Functions to analyze the results of the cost-based selection methods."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

def accuracy_classifier_plot(dfPen_Ranking, X_val, y_val, cost_vec,
                             noise_idx, subset_size, classifier_func, 
                             args_classifier = dict(), num_fold = 3,
                             save_name = None, random_seed = 123):
    """ Accuracy plot of a classifier by cross-validation k-fold.
    
    Train a classifier to evaluate the quality of the subsets deduced 
    with the multi_* cost-based functions (see cost_based_methods.py),
    and plot the evolution of the accuracy deduced by cross validation k-fold,
    total subset cost and proportion of noise features (out of the four 
    simulated) depending on the cost penalization parameter. 
    The graph also represents +/- std_deviation/2 of the classifier accuracy 
    across the folds.
    
    Args:
        dfPen_Ranking (pandas.core.frame.DataFrame):
            the pandas DataFrame returned by the multi_* cost-based functions,
            that contains in each row, a penalization parameter value and the
            corresponding ranking indexes, in decreasing order of importance.
        X_val (numpy.ndarray):
            the numerical features to use as validation data, where 
            each row represents an individual, and each column a feature.
            It contains all the features, which are not selected yet.
        y_val (numpy.ndarray):
            a list of integers representing the validation data labels.
        cost_vec (numpy.ndarray):
            the vector of feature costs used to obtain the rankings.
        noise_idx (list):
            a list containing the indices of the noise features.
            Returned by preprocessing_utils.noise_position.
        subset_size (int):
            the size of the best subset of features to keep and analyze.
        classifier_func (sklearn classifier):
            a sklearn classifier, compatible with the function
            sklearn.model_selection.cross_val_score. For examples:
            sklearn.neighbors.KNeighborsClassifier or sklearn.svm.SVC.
        args_classifier (dict):
            a dictionary containing as keys the arguments of the classifier_func
            function, and as values the argument values.
        num_fold (int):
            the number of folds to use for the cross-validation.
        save_name (str):
            the path/name of the file in which the resulting plot will be saved.
            If None, the graph is not saved.
        random_seed (int):
            the random seed to use when determining the different folds, with
            the function StratifiedKFold. This is useful when analyzing different
            classifiers, to ensure that the same data are used.

    Returns:
        avg_accuracy (numpy.ndarray):
            a numpy array containing, for each cost parameter value, the average
            classification accuracy (across the folds).
        std_accuracy (numpy.ndarray):
            a numpy array containing, for each cost parameter value, the 
            standard deviation of the classification accuracy (across the folds). 
        total_cost (numpy.ndarray):
            a numpy array containing, for each cost parameter value, the
            total cost of the resulting subset of selected features.
        prop_noise (numpy.ndarray):
            a numpy array containing, for each cost parameter value, the
            proportion of noise included in the resulting subset of features.
        
    """
    
    cross_val = StratifiedKFold(n_splits = num_fold, random_state = random_seed, shuffle=True)
    grid_cost_param = dfPen_Ranking.cost_param

    avg_accuracy = np.zeros(len(grid_cost_param))
    std_accuracy = np.zeros(len(grid_cost_param))
    total_cost = np.zeros(len(grid_cost_param))
    prop_noise = np.zeros(len(grid_cost_param))

    k = 0
    for cost_param in grid_cost_param:
        
        # Extract the ranking
        ranking = dfPen_Ranking.iloc[k,1:]
        # and best features
        top_features = [int(ranking.iloc[i]) for i in range(subset_size)]
        # Compute the total cost and proportion of noise
        total_cost[k] = np.sum(cost_vec[top_features])
        prop_noise[k] = len( _private_common(top_features, noise_idx) )/len(noise_idx)
    
        # Reduce the size of the data using the top subset_size features
        X_val_intermed = X_val[:, top_features]
        
        # Train and run the classifier to determine the CV accuracy
        classifier = classifier_func(**args_classifier)
        scores = cross_val_score(classifier, X_val_intermed, y_val, cv = cross_val)
        avg_accuracy[k] = scores.mean()
        std_accuracy[k] = scores.std()

        k += 1
        
    plt.figure(figsize=(6,6))
    plt.plot(grid_cost_param, total_cost, "r--", label='Cost', linewidth=2)
    plt.step(grid_cost_param, prop_noise, 'm:', label='Number of noise features (out of 4)', where='post', linewidth=2)
    plt.plot(grid_cost_param, avg_accuracy, "b-.", label='Accuracy', linewidth=2)
    
    high_yval = avg_accuracy + std_accuracy/2
    low_yval = avg_accuracy - std_accuracy/2
    
    plt.fill_between(grid_cost_param, low_yval, high_yval, alpha=0.2, color="blue")
    plt.legend()
    plt.xlabel("Cost penalization parameter" + " $\lambda$", fontsize=12)
    plt.ylabel("Cost / Accuracy / Number of noise", fontsize=12)
    plt.ylim(-0.2,1.4)
    plt.subplots_adjust(top=0.95, right=0.95)
    if save_name is not None:
        plt.savefig(save_name, pad_inches = 0)
    #plt.show()
    
    return avg_accuracy, std_accuracy, total_cost, prop_noise

def _private_common(lst1, lst2):
    """ Utility function to determine the common features between two sets. """
    return list(set(lst1) & set(lst2))