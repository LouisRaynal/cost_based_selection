# -*- coding: utf-8 -*-
""" Functions to analyze the relevance of features selected with smaller
networks when classifying larger networks.
"""

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from cost_based_selection import cost_based_methods

def common_features_plot(dict_small, dict_large, dict_fmt = dict()):
    """ Plot the graph of evolution of common features selected with different reference tables.
    
    This function returns and plots the number of common features selected when 
    using larger and smaller networks, depending on the number of features 
    selected. The best case scenario is represented by the black curve, i.e. 
    when for each number of selected features, the same features are selected 
    when using small or large networks. This function also returns the areas 
    under the curves of the evolution of the number of common selected 
    features, relatively to the best curve. These areas are computed 
    for all intervals [1,p] where p takes as value all possible feature subset 
    sizes, thus p = 1, ..., number of features.
    
    Args:
        dict_small (dict):
            a dictionary containing as keys the name of the methods, and as
            values the corresponding rankings (list) obtained when using smaller
            networks.
        dict_large (dict):
            a dictionary containing as keys the name of the methods, and as
            values the corresponding rankings (list) obtained when using larger
            networks.
        dict_fmt (dict):
            a dictionary containing as keys the name of the methods, and as
            values a string in the format strings as employed by
            matplotlib.pyplot.plot: namely '[marker][line][color]'.
        
    Returns:
        dict_common (dict):
            a dictionary containing as keys the name of the methods, and as
            values the number of common features selected in dict_small and
            dict_large. All possible subset sizes p are considered.
        dict_areas (dict):
            a dictionary containing as keys the name of the methods, and as
            values the relative areas between the evolution of 
            the common selected features curve and the best case scenario.
            All possible intervals [1,p] are considered.
    
    """
    
    # Check that the two dictionaries have the same keys
    diff = set(dict_large) - set(dict_small)
    if len(diff) > 0:
        raise Exception("The keys in both dictionaries must be the same.")
    
    # Recover the number of features
    tmpRank = dict_small[random.choice(list(dict_small.keys()))]
    nCov = len(tmpRank)
    
    # Initialize a dictionary to store the number of common features per method
    dict_common = {key: [None]*nCov for key in dict_small.keys()}
    # Initialize a dictionary to store the area under the resulting curves
    dict_areas = {key: [None]*nCov for key in dict_small.keys()}
    
    # For each possible subset size, compute the number of common features selected
    for i in range(nCov):
        for key in dict_small.keys():
            dict_common[key][i] = len(_private_common(dict_small[key][:i+1], dict_large[key][:i+1]))
            dict_areas[key][i] = np.sum(dict_common[key][:i+1]) / np.sum(list(range(1,i+2,1)))
    
    subset_grid = list(range(1,nCov+1))
    # Plot the curves of evolution
    fig, ax = plt.subplots(figsize = (13, 8))
    ax.step(subset_grid, list(range(1,len(subset_grid)+1,1)), 'k-', label="Best", where='post')
    for key in dict_small.keys():
        if ( len(dict_fmt) == 0 ) | ( len(dict_fmt) != len(dict_small) ):
            ax.step(subset_grid, dict_common[key], label = key, where='post')
        else:
            ax.step(subset_grid, dict_common[key], dict_fmt[key], label = key, where = 'post')
    ax.legend(prop={'size': 12})
    ax.set_xlabel("Size of feature subset", fontsize=12)
    ax.set_ylabel("Number of common features", fontsize=12)
    plt.subplots_adjust(top=0.95, right=0.95)
    #fig.show()

    return dict_common, dict_areas

    
def _private_common(lst1, lst2):
    """ Function to determine the common elements in two lists. """
    return list(set(lst1) & set(lst2))


def difference_accuracy_small_large(dict_small, dict_large, X_val, y_val, subset_size_vec,
                                    classifier_func, args_classifier = dict(), num_fold = 3):
    """ Compute the decrease of classification accuracy when using small networks instead of large ones.
    
    This function compute the difference of accuracy to classify networks with
    a large number of nodes, when using networks with that many nodes to select
    the features, or a smaller number of nodes, in other terms the difference
    Acc(large) - Acc(small) is computed. This is done for a sklearn classifier,
    and cross-validation is used. The individual accuracies are also returned.
    
    Args:
        dict_small (dict):
            a dictionary containing as keys the name of the methods, and as
            values the corresponding rankings (list) obtained when using smaller
            networks.
        dict_large (dict):
            a dictionary containing as keys the name of the methods, and as
            values the corresponding rankings (list) obtained when using larger
            networks.
        X_val (numpy.ndarray):
            the numerical features to use as validation data, where 
            each row represents an individual, and each column a feature.
            These contains all the features, which are not selected yet.
        y_val (numpy.ndarray):
            a list of integers representing the validation data labels.
        subset_size_vec (list):
            a list of difference feature subset sizes on which the difference of
            accuracy will be computed.
        classifier_func (sklearn classifier):
            a sklearn classifier, compatible with the function
            sklearn.model_selection.cross_val_score. For examples:
            sklearn.neighbors.KNeighborsClassifier or sklearn.svm.SVC.
        args_classifier (dict):
            a dictionary containing as keys the arguments of the classifier_func
            function, and as values the argument values.            
        num_fold (int):
            the number of folds to use for the cross-validation.
            
    Returns:
        dict_accuracy_decrease (dict):
            a dictionary containing as keys, the name of the method used in
            dict_small and dict_large, and as values, for each size of feature
            subsets, the decrease of classification accuracy observed when
            using smaller networks instead of larger networks for the feature 
            selection step: Acc(large) - Acc(small).
        dict_accuracy_large_using_large (dict):
            a dictionary containing as keys, the name of the method used in
            dict_small and dict_large, and as values, for each size of feature 
            subsets, the average classification accuracy (across the folds) 
            when classifying large networks and selecting features with 
            large networks.
        dict_std_large_using_large (dict):
            a dictionary containing as keys, the name of the method used in
            dict_small and dict_large, and as values, for each size of feature 
            subsets, the standard deviation of classification accuracy (across the
            folds) when classifying large networks and selecting features with 
            large networks.
        dict_accuracy_large_using_small (dict):
            a dictionary containing as keys, the name of the method used in
            dict_small and dict_large, and as values, for each size of feature 
            subsets, the average classification accuracy (across the folds) 
            when classifying large networks and selecting features with 
            small networks.
        dict_std_large_using_small (dict):
            a dictionary containing as keys, the name of the method used in
            dict_small and dict_large, and as values, for each size of feature 
            subsets, the standard deviation of classification accuracy (across the
            folds) when classifying large networks and selecting features with 
            small networks.
    
    """
    
    cross_val = StratifiedKFold(n_splits = num_fold)
    num_subsets = len(subset_size_vec)

    # Check that the two dictionary have the same keys
    diff = set(dict_large) - set(dict_small)
    if len(diff) > 0:
        raise Exception("The keys in both dictionaries must be the same.")
    
    # Initialize the outputs
    dict_accuracy_decrease = {key: [None]*num_subsets for key in dict_small.keys()}
    dict_accuracy_large_using_large = {key: [None]*num_subsets for key in dict_small.keys()}
    dict_std_large_using_large = {key: [None]*num_subsets for key in dict_small.keys()}
    dict_accuracy_large_using_small = {key: [None]*num_subsets for key in dict_small.keys()}
    dict_std_large_using_small = {key: [None]*num_subsets for key in dict_small.keys()}
    
    # For each subset size
    idx = 0
    for subset_size in subset_size_vec:
        # For each method
        for key in dict_small.keys():
            # Deduce the feature to keep using small or large networks
            top_small = dict_small[key][:subset_size]
            top_large = dict_large[key][:subset_size]
            # Reduce X_val accordingly
            X_val_using_small = X_val[:,top_small]
            X_val_using_large = X_val[:,top_large]
            # Train and run the classifier to determine the CV accuracy
            classifier = classifier_func(**args_classifier)
            scores_using_small = cross_val_score(classifier, 
                                                 X_val_using_small,
                                                 y_val, cv = cross_val)
            scores_using_large = cross_val_score(classifier, 
                                                 X_val_using_large,
                                                 y_val, cv = cross_val)
            # Fill the outputs
            dict_accuracy_large_using_small[key][idx] = scores_using_small.mean()
            dict_std_large_using_small[key][idx] = scores_using_small.std()
            dict_accuracy_large_using_large[key][idx] = scores_using_large.mean()
            dict_std_large_using_large[key][idx] = scores_using_large.std()
            dict_accuracy_decrease[key][idx] = dict_accuracy_large_using_large[key][idx] - dict_accuracy_large_using_small[key][idx]
        idx += 1
    
    return dict_accuracy_decrease, dict_accuracy_large_using_large, dict_std_large_using_large, dict_accuracy_large_using_small, dict_std_large_using_small


def common_features_difference_accuracy(dfSummaries_small, dfSummaries_large,
                                        dfModIndex_small, dfModIndex_large,
                                        is_disc, subset_size_vec, val_size = 0.5,
                                        random_seed = 123, num_fold = 3,
                                        args_mRMR = dict(),
                                        args_JMI = dict(), args_JMIM = dict(),
                                        args_reliefF_classic = dict(),
                                        args_reliefF_rf = dict(),
                                        args_rf_impurity = dict(),
                                        args_rf_permutation = dict(),
                                        args_svm = dict(),
                                        args_knn = {'n_neighbors':10, 'n_jobs':1}):
    """ Display the analyses on common features and decrease of accuracy induced by the use of smaller networks.

    This is a wrapper to run the functions common_feature_plot
    and difference_accuracy_small_large jointly and easily.
    This function might be useful to run multiple replicate analyses.
    
    For every possible subset size of features, it returns the 
    common number of features selected by the different selection methods, when
    using small and large networks, as well as areas under the curves of the
    evolution of these numbers depending on the subset sizes.
    It also computes the decrease of classification accuracy to classify
    large networks, when selecting features using the table formed from
    small networks, rather than large networks: Acc(large) - Acc(small).
    The different reference tables are divided in train and 
    validation sets with a proportion of validation data defined by 
    val_size. The decrease of accuracy is computed on subset sizes
    described in subset_size_vec, and the classification accuracy is
    determined by cross-validation with num_fold-folds on the proportion 
    val_size of data. We are using all the filter selection methods
    described in our paper and implemented in the module cost_based_methods.py,
    so additional arguments for the selection methods must be specified in
    dictionaries if the default values are not satisfying. Note also that each
    penalization parameter value is set to 0.
    Finally, two classifiers are used, an SVM and a k-nearest-neighbors classifier,
    for which arguments can be specified by the arguments args_svc and args_knn.
        
    Args:
        dfSummaries_small (pandas.core.frame.DataFrame):
            the pandas DataFrame containing, for each simulated data (in rows),
            the summary statistic values (in columns) computed using small
            networks.
        dfSummaries_large (pandas.core.frame.DataFrame):
            the pandas DataFrame containing, for each simulated data (in rows),
            the summary statistic values (in columns) computed using large
            networks.
        dfModIndex_small (pandas.core.frame.DataFrame):
            the panda DataFrame containing the model indexes associated to
            the table obtained from small networks.
        dfModIndex_large (pandas.core.frame.DataFrame):
            the panda DataFrame containing the model indexes associated to
            the table obtained from large networks.
        is_disc (list):
            a list of Booleans, common to dfSummaries_small and 
            dfSummaries_large, indicating with True if the feature is discrete 
            and False if continuous.
        subset_size_vec (list):
            a list containing sizes of feature subsets for which the decrease
            of accuracy will be computed.
        random_seed (int):
            the random seed to use when partitioning the data in train, 
            validation sets.
        num_fold (int):
            the number of folds to use for the cross-validation.
        args_mRMR (dict):
            a dictionary containing as keys the optional arguments for the
            cost_based_methods.mRMR function, and as values the argument values.
            If unspecified, the default values of mRMR are used.
        args_JMI (dict):
            a dictionary containing as keys the optional arguments for the
            cost_based_methods.JMI function, and as values the argument values.
            If unspecified, the default values of JMI are used.
        args_JMIM (dict):
            a dictionary containing as keys the optional arguments for the
            cost_based_methods.JMIM function, and as values the argument values.
            If unspecified, the default values of JMIM are used.
        args_reliefF_classic (dict):
            a dictionary containing as keys the optional arguments for the
            cost_based_methods.reliefF function when using 
            proximity = "distance", and as values the argument values. 
            If unspecified, the default values of reliefF are used.
        args_reliefF_rf (dict):
            a dictionary containing as keys the optional arguments for the
            cost_based_methods.reliefF function when using 
            proximity = "rf prox", and as values the argument values.
            If unspecified, the default values of reliefF are used.
        args_rf_impurity (dict):
            a dictionary containing as keys the optional arguments for the
            cost_based_methods.pen_rf_importance function when
            importance = "impurity", and as values the argument values. 
            If unspecified, the default values of pen_rf_importance are used.
        args_rf_permutation (dict):
            a dictionary containing as keys the optional arguments for the
            cost_based_methods.pen_rf_importance function when
            importance = "permutation", and as values the argument values. 
            If unspecified, the default values of pen_rf_importance are used.
        args_svc (dict):
            a dictionary containing as keys the arguments of the SVM classifier
            as described in the sklearn.svm.SVC function, and as values the 
            argument values.
        args_knn (dict):
            a dictionary containing as keys the arguments of the k-nearest-
            neighbor algorithm as described in the sklearn.neighbors.KNeighborsClassifier
            function, and as values the argument values.            
    
    Returns:
        dict_common (dict):
            a dictionary containing as keys the name of the methods, and as
            values the number of common features selected when using small and
            large networks. All possible subset sizes p are considered.
        dict_areas (dict):
            a dictionary containing as keys the name of the methods, and as
            values the relative areas between the evolution of 
            the common selected features curve and the best case scenario curve.
            All possible intervals [1,p] are considered.
        decrease_acc_SVM (dict):
            for the SVM classifier, a dictionary containing as keys, the name of
            the different feature selection methods, and as values, for each 
            size of feature subsets in subset_size_vec, the decrease of 
            classification accuracy observed when using smaller networks instead
            of larger networks for the selection step: Acc(large) - Acc(small).    
        decrease_acc_knn (dict):
            for the k-nearest-neighbors classifier, a dictionary containing as keys, the name of
            the different feature selection methods, and as values, for each 
            size of feature subsets in subset_size_vec, the decrease of 
            classification accuracy observed when using smaller networks instead
            of larger networks for the selection step: Acc(large) - Acc(small).

    """
    
    ### Select features with small networks
    
    X_small = np.array(dfSummaries_small)
    y_small = dfModIndex_small.modIndex.tolist()
    
    # For one replicate
    
    # Split the reference table in training and validation set
    (X_train_small, X_val_small, y_train_small, y_val_small) = train_test_split(X_small, y_small, test_size=val_size, random_state=random_seed, stratify=y_small)
    
    # For each method, recover the full ranking of the features, without penalization
    ranking_mRMR_small, *other = cost_based_methods.mRMR(X = X_train_small, y = y_train_small,
                                                           is_disc = is_disc, **args_mRMR)
    
    ranking_JMI_small, *other = cost_based_methods.JMI(X = X_train_small, y = y_train_small,
                                                         is_disc = is_disc, **args_JMI)
    
    ranking_JMIM_small, *other = cost_based_methods.JMIM(X = X_train_small, y=y_train_small, 
                                                           is_disc=is_disc, **args_JMIM)
    
    ranking_reliefF_c_small, *other = cost_based_methods.reliefF(X = X_train_small, y=y_train_small, 
                                                                   proximity = "distance",
                                                                   **args_reliefF_classic)
    
    ranking_reliefF_rf_small, *other = cost_based_methods.reliefF(X = X_train_small, y=y_train_small, 
                                                                  proximity = "rf prox",
                                                                  **args_reliefF_classic)
    
    ranking_impurity_small, *other = cost_based_methods.pen_rf_importance(X = X_train_small, y = y_train_small,
                                                                          imp_type = "impurity",
                                                                          **args_rf_impurity)
    
    ranking_permut_small, *other = cost_based_methods.pen_rf_importance(X = X_train_small, y = y_train_small,
                                                                          imp_type = "permutation",
                                                                          **args_rf_permutation)
    
    ### Select features with large networks
    
    X_large = np.array(dfSummaries_large)
    y_large = dfModIndex_large.modIndex.tolist()
        
    # Split the reference table in training and validation set
    (X_train_large, X_val_large, y_train_large, y_val_large) = train_test_split(X_large, y_large, test_size=val_size, random_state=random_seed, stratify=y_large)
    
    # For each method, recover the full ranking of the features, without penalization
    ranking_mRMR_large, *other = cost_based_methods.mRMR(X = X_train_large, y = y_train_large,
                                                           is_disc = is_disc, **args_mRMR)
    
    ranking_JMI_large, *other = cost_based_methods.JMI(X = X_train_large, y = y_train_large,
                                                         is_disc = is_disc, **args_JMI)
    
    ranking_JMIM_large, *other = cost_based_methods.JMIM(X = X_train_large, y=y_train_large, 
                                                           is_disc=is_disc, **args_JMIM)
    
    ranking_reliefF_c_large, *other = cost_based_methods.reliefF(X = X_train_large, y=y_train_large, 
                                                                   proximity = "distance",
                                                                   **args_reliefF_classic)
    
    ranking_reliefF_rf_large, *other = cost_based_methods.reliefF(X = X_train_large, y=y_train_large, 
                                                                  proximity = "rf prox",
                                                                  **args_reliefF_classic)
    
    ranking_impurity_large, *other = cost_based_methods.pen_rf_importance(X = X_train_large, y = y_train_large,
                                                                          imp_type = "impurity",
                                                                          **args_rf_impurity)
    
    ranking_permut_large, *other = cost_based_methods.pen_rf_importance(X = X_train_large, y = y_train_large,
                                                                          imp_type = "permutation",
                                                                          **args_rf_permutation)
    
    ### To compare the rankings with small and large networks,
    ### we build one dictionary per reference table, with keys the name of the methods
    ### and value the corresponding rankings
    
    dict_small_rankings = {"mRMR": ranking_mRMR_small, "JMI": ranking_JMI_small, "JMIM": ranking_JMIM_small,
                           "reliefF classic": ranking_reliefF_c_small, "reliefF RF prox.": ranking_reliefF_rf_small,
                           "RF MDI": ranking_impurity_small, "RF MDA": ranking_permut_small}
    
    dict_large_rankings = {"mRMR": ranking_mRMR_large, "JMI": ranking_JMI_large, "JMIM": ranking_JMIM_large,
                           "reliefF classic": ranking_reliefF_c_large, "reliefF RF prox.": ranking_reliefF_rf_large,
                           "RF MDI": ranking_impurity_large, "RF MDA": ranking_permut_large}
    
    dict_fmt = {"mRMR": 'b--', "JMI": 'b-.', "JMIM": 'b:', "reliefF classic": 'r--',
                "reliefF RF prox.": 'r:', "RF MDI": 'g--', "RF MDA": 'g:'}
    
    dict_common, dict_areas = common_features_plot(dict_small_rankings,
                                                   dict_large_rankings,
                                                   dict_fmt)
    
    ### Now, we want to check the classification accuracy to predict the network classes
    ### of large networks, when using the features selected with small or large networks
    
    decrease_acc_SVM, *other = difference_accuracy_small_large(dict_small = dict_small_rankings,
                                                               dict_large = dict_large_rankings,
                                                               X_val = X_val_large,
                                                               y_val = y_val_large,
                                                               subset_size_vec = subset_size_vec,
                                                               classifier_func = SVC,
                                                               args_classifier = args_svm,
                                                               num_fold = num_fold)
    
    decrease_acc_knn, *other =  difference_accuracy_small_large(dict_small = dict_small_rankings,
                                                                dict_large = dict_large_rankings,
                                                                X_val = X_val_large,
                                                                y_val = y_val_large,
                                                                subset_size_vec = subset_size_vec,
                                                                classifier_func = KNeighborsClassifier,
                                                                args_classifier = args_knn,
                                                                num_fold = num_fold)

    return dict_common, dict_areas, decrease_acc_SVM, decrease_acc_knn


def replication_common_features_difference_accuracy(dfSummaries_small, dfSummaries_large,
                                                    dfModIndex_small, dfModIndex_large,
                                                    is_disc, subset_size_vec, 
                                                    val_size = 0.5, num_fold = 3,
                                                    args_mRMR = dict(),
                                                    args_JMI = dict(), 
                                                    args_JMIM = dict(),
                                                    args_reliefF_classic = dict(),
                                                    args_reliefF_rf = dict(),
                                                    args_rf_impurity = dict(),
                                                    args_rf_permutation = dict(),
                                                    args_svm = dict(),
                                                    args_knn = {'n_neighbors':10, 'n_jobs':1},
                                                    num_rep = 50, num_cores = 1):
    """ Launch with replication the function common_features_difference_accuracy

    This function launch num_rep times the function 
    common_features_difference_accuracy, possibly in parallel.
    The results must then be analyzed with the function analyze_replication_res.
    
    Args:
        The arguments are almost identical to the ones in the function
        common_features_difference_accuracy, (see its documentation), excepted
        for random_seed that is used to provide different partitioning of the
        reference table, depending on the index of replication.
        Below are the two new arguments.
        num_rep (int):
            the number of replications to perform. Each run will be performed
            on a different partitioning of the reference table into a training
            and validation set. 50 by default.
        num_cores (int):
            the number of CPU cores to perform parallel computing.

    Returns:
        replication_res (list):
            the resulting list containing the results over multiple runs.
            This output must be given to the function analyze_replication_res.

    """
    
    replication_res = Parallel(n_jobs = num_cores)(delayed(common_features_difference_accuracy)(
            dfSummaries_small = dfSummaries_small,
            dfSummaries_large = dfSummaries_large,
            dfModIndex_small = dfModIndex_small,
            dfModIndex_large = dfModIndex_large,
            is_disc = is_disc, 
            subset_size_vec = subset_size_vec,
            val_size = val_size,
            random_seed = seed,
            num_fold = num_fold,
            args_mRMR = args_mRMR,
            args_JMI = args_JMI,
            args_JMIM = args_JMIM,
            args_reliefF_classic = args_reliefF_classic,
            args_reliefF_rf = args_reliefF_rf,
            args_rf_impurity = args_rf_impurity,
            args_rf_permutation = args_rf_permutation,
            args_svm = args_svm,
            args_knn = args_knn) for seed in list(range(1,num_rep+1,1)) )

    return replication_res


def analyze_replication_res(replication_res, subset_size_vec,
                            showfliers = True, save = True, plot_reliefF = True):
    """ Analyze replicated results returned by common_features_difference_accuracy
    
    This function analyze the results returned when launching the function
    common_features_difference_accuracy multiple times. Namely, it returns over
    the replicated analysis, the mean and standard deviation of the common 
    number of features selected with the two different network sizes, the mean 
    and standard deviation of the associated relative areas, and plots, for the
    k-nearest-neighbors (top graph) and SVM (bottom graph) classifiers, the decrease
    of accuracy boxplots.
    The common_features_difference_accuracy function must be launched on n_jobs
    CPU cores, num_rep times in the following way:
    replicate_res = Parallel(n_jobs = n_jobs)(delayed(common_features_difference_accuracy)(dfSummaries_small, dfSummaries_large, dfModIndex_small, dfModIndex_large, is_disc, subset_size_vec, val_size, random_seed = seed) for seed in list(range(1,num_rep+1,1)) )
    The replicate analysis are thus performed over the same reference table,
    but divided in different training - validation sets depending on each random_seed 
    value.
    
    Args:
        replication_res (list):
            the resulting list when launching multiple times the function
            common_features_difference_accuracy as stated above.
        subset_size_vec (list):
            a list containing sizes of feature subsets for which the decrease
            of accuracy were computed.
        showfliers (bool):
            a Boolean specifying whether or not the outliers must be included
            in the boxplots. True by default.
        save (bool):
            a Boolean specifying whether or not the resulting graphs need to be
            saved in the current file location. True by default.
        plot_reliefF (bool):
            a Boolean to indicate if the reliefF methods must be included in the
            boxplots or not. True by default.
        
    Returns:
        dict_avg_common (pandas.core.frame.DataFrame):
            a pandas DataFrame containing for each method (in rows) the average
            number of common features selected when using small and large
            networks, for each subset size (in columns).
        dict_std_common (pandas.core.frame.DataFrame):
            a pandas DataFrame containing for each method (in rows) the standard
            deviation of the number of common features selected when using small
            and large networks, for each subset size (in columns).
        dict_avg_areas (pandas.core.frame.DataFrame):
            a pandas DataFrame containing for each method (in rows) the average
            relative areas between the evolution of the common selected features
            curve and the best case scenario, for each interval [1,p] with p
            in columns.
        dict_std_areas (pandas.core.frame.DataFrame):
            a pandas DataFrame containing for each method (in rows) the standard
            deviation of the relative areas between the evolution of the common 
            selected features curve and the best case scenario, for each 
            interval [1,p] with p in columns.
    
    """

    num_rep = len(replication_res)
    num_features = len(replication_res[0][0]["mRMR"])

    keys_used = replication_res[0][0].keys()

    ### Compute the average number of common features selected over replicates
    ### and the average area under the curves of common features evolution,
    ### and corresponding standard deviations
    dict_common_tmp_rep = {}
    dict_avg_common = {}
    dict_std_common = {}    
    dict_areas_tmp_rep = {}
    dict_avg_areas = {}
    dict_std_areas = {}

    # Initialization
    for key in keys_used:
        dict_common_tmp_rep[key] = [[]] * num_features
        dict_areas_tmp_rep[key] = [[]] * num_features
        dict_avg_common[key] = np.zeros(num_features)
        dict_std_common[key] = np.zeros(num_features)
        dict_avg_areas[key] = np.zeros(num_features)
        dict_std_areas[key] = np.zeros(num_features)
    
    # Store the individual values for each replicate
    for key in keys_used:
        for feat in range(num_features):
            for rep in range(num_rep):
                dict_common_tmp_rep[key][feat] = dict_common_tmp_rep[key][feat] + [replication_res[rep][0][key][feat]]
                dict_areas_tmp_rep[key][feat] = dict_areas_tmp_rep[key][feat] + [replication_res[rep][1][key][feat]]
    
    # Compute the average and standard deviation over replicated runs
    for key in keys_used:
        for feat in range(num_features):
            dict_avg_common[key][feat] = np.mean(dict_common_tmp_rep[key][feat])
            dict_std_common[key][feat] = np.std(dict_common_tmp_rep[key][feat])
            dict_avg_areas[key][feat] = np.mean(dict_areas_tmp_rep[key][feat])
            dict_std_areas[key][feat] = np.std(dict_areas_tmp_rep[key][feat])
    
    ### Recover the decrease of accuracy for the K-NN and SVM classifiers
    ### stored in a dictionary to plot the boxplots
    dict_precision_subset_method_SVM = {'Precision': [], 'Size of feature subset': [], 'Methods': []}
    dict_precision_subset_method_KNN = {'Precision': [], 'Size of feature subset': [], 'Methods': []}
    
    for key in keys_used:
        k = 0
        for subset in subset_size_vec:
            for rep in range(num_rep):
                dict_precision_subset_method_SVM['Precision'] = dict_precision_subset_method_SVM['Precision'] + [replication_res[rep][2][key][k]]
                dict_precision_subset_method_SVM['Size of feature subset'] = dict_precision_subset_method_SVM['Size of feature subset'] + [subset]
                dict_precision_subset_method_SVM['Methods'] = dict_precision_subset_method_SVM['Methods'] + [key]
                dict_precision_subset_method_KNN['Precision'] = dict_precision_subset_method_KNN['Precision'] + [replication_res[rep][3][key][k]]
                dict_precision_subset_method_KNN['Size of feature subset'] = dict_precision_subset_method_KNN['Size of feature subset'] + [subset]
                dict_precision_subset_method_KNN['Methods'] = dict_precision_subset_method_KNN['Methods'] + [key]
            k = k + 1
                
    df_res_SVM = pd.DataFrame(dict_precision_subset_method_SVM)
    df_res_KNN = pd.DataFrame(dict_precision_subset_method_KNN)
    
    if not plot_reliefF:
        df_res_KNN_sub =  df_res_KNN[(df_res_KNN.Methods != "reliefF classic") & (df_res_KNN.Methods != "reliefF RF prox.")]
        df_res_SVM_sub =  df_res_SVM[(df_res_SVM.Methods != "reliefF classic") & (df_res_SVM.Methods != "reliefF RF prox.")]

    # For K-NN
    fig, ax = plt.subplots(figsize=(12,8))
    if plot_reliefF:
        bb = sns.boxplot(ax=ax, x="Size of feature subset", y="Precision", hue="Methods",
                         data=df_res_KNN, palette="Set1", showfliers = showfliers)
    else:
        bb = sns.boxplot(ax=ax, x="Size of feature subset", y="Precision", hue="Methods",
                         data=df_res_KNN_sub, palette="Set1", showfliers = showfliers)
    #ax.set(ylim=(-0.06, 0.06))
    bb.set_xlabel("Size of feature subset", fontsize=12)
    bb.set_ylabel("Precision", fontsize=12)
    plt.subplots_adjust(top=0.95, right=0.95)
    if save and plot_reliefF:
        plt.savefig('Boxplot_diff_withReliefF_KNN.pdf', pad_inches=0)
    elif save and not plot_reliefF:
        plt.savefig('Boxplot_diff_withoutReliefF_KNN.pdf', pad_inches=0)
    
    # For SVM
    fig, ax = plt.subplots(figsize=(12,8))
    if plot_reliefF:
        bb = sns.boxplot(ax=ax, x="Size of feature subset", y="Precision", hue="Methods",
                         data=df_res_SVM, palette="Set1", showfliers = showfliers)
    else:
        bb = sns.boxplot(ax=ax, x="Size of feature subset", y="Precision", hue="Methods",
                         data=df_res_SVM_sub, palette="Set1", showfliers = showfliers)    
    #ax.set(ylim=(-0.06, 0.06))
    bb.set_xlabel("Size of feature subset", fontsize=12)
    bb.set_ylabel("Precision", fontsize=12)
    plt.subplots_adjust(top=0.95, right=0.95)
    if save and plot_reliefF:
        plt.savefig('Boxplot_diff_withReliefF_SVM.pdf', pad_inches=0)
    elif save and not plot_reliefF:
        plt.savefig('Boxplot_diff_withoutReliefF_SVM.pdf', pad_inches=0)
        
    # TO DO: Add graphical arguments
    df_avg_common = pd.DataFrame.from_dict(dict_avg_common, orient = 'index',
                                           columns = list(range(1,num_features+1,1)))
    df_std_common = pd.DataFrame.from_dict(dict_std_common, orient = 'index',
                                           columns = list(range(1,num_features+1,1)))
    df_avg_areas = pd.DataFrame.from_dict(dict_avg_areas, orient = 'index',
                                           columns = list(range(1,num_features+1,1)))
    df_std_areas = pd.DataFrame.from_dict(dict_std_areas, orient = 'index',
                                           columns = list(range(1,num_features+1,1)))

    return df_avg_common, df_std_common, df_avg_areas, df_std_areas
