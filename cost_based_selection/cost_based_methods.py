# -*- coding: utf-8 -*-
""" Implementation of cost-based feature selection/ranking algorithms.

Implementation of the cost-based version of the filter feature selection method
based on Maximal-Relevance-Minimal-Redundancy (mRMR), Joint Mutual Information
(JMI), Joint Mutual Information Maximization (JMIM), a version of 
ReliefF that can compute nearest neighbors either with random forests, or with
an L1 distance. A cost-based ranking is also available by penalization of the 
random forest feature importance, or by using the feature importance of
a random forest where the sampling of features at each internal node
is proportional to the inverse of their cost.
Moreover, to analyze the rankings for different penalization parameter values,
we also implement corresponding functions that return the different rankings
for each penalization value.

"""

import collections
import copy
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import  RandomForestClassifier

# To use the R package ranger for RF importance computation
import rpy2.robjects


def mRMR(X, y, is_disc, cost_vec = None, cost_param = 0, 
         num_features_to_select = None, random_seed = 123, num_cores = 1,
         MI_matrix = None):
    """ Cost-based feature ranking with maximum relevance minimum redundancy.
    
    Cost-based adaptation of the filter feature selection algorithm Maximal-
    Relevance-Minimal-Redundancy (mRMR, Peng et al. (2005)).

    H. Peng, F. Long, and C. Ding.  Feature Selection Based on Mutual 
    Information: Criteria of Max-Dependency, Max-Relevance, and Min-Redundancy.
    IEEE Transactions on pattern analysis and machine intelligence, 
    27:1226–1238, 2005.
    
    Args:
        X (numpy.ndarray):
            the numerical features to use as training data, where 
            each row represents an individual, and each column a feature.
        y (list):
            a list of integers representing the training data labels.
        is_disc (list):
            a list of booleans indicating with True if the feature is discrete
            and False if continuous.
        cost_vec (numpy.ndarray):
            the vector of costs represented by a numpy.ndarray with shape
            (1, X.shape[1]). If None, the cost is set to zero for each feature.
        cost_param (float): 
            the positive cost penalization parameter. 0 by default.
        num_features_to_select (int):
            the number of best features to select. If unspecified, does not
            select a subset of features but keep all of them.
        random_seed (int):
            the random seed to use with the mutual_information function
            (when computing the Mutual Information (MI) involving one or more
            continuous features).
        num_cores (int):
            the number of CPU cores to use in parallel to compute the MI.
        MI_matrix (numpy.ndarray):
            the matrix of precomputed pairwise MI between pairs of features to
            save times when wanting to use multiple cost values. 
            By default this matrix is computed in the function.
            
    Returns:
        ranking (list):
            list containing the indices of the ranked features as specified in
            X, in decreasing order of importance.
        matrix_MI (numpy.ndarray):
            the matrix of precomputed MI between pairs of features.
            
    """
        
    num_features = X.shape[1]
    
    if (cost_vec is None):
        # If no cost is specified, then all costs are set as equal to zero
        cost_vec = np.zeros(num_features)
    #else:
    #    cost_vec=np.array(cost_vec)
    
    # Check on num_features_to_select
    if (num_features_to_select is not None):
        num_selected_features = min(num_features, num_features_to_select)
    else:
        num_selected_features = num_features
    
    # unRanked contains the feature indices unranked
    unRanked = list(range(num_features))
    
    # If a feature is discrete but with always different values, then
    # convert it into a continuous one
    # (to handle errors with the MI computation function)
    for featIdx in range(num_features):
        if is_disc[featIdx]==True and len(np.unique(X[:,featIdx])) == X.shape[0]:
            is_disc[featIdx] = False
    
    # Computing all the MIs I(X_j; y)
    initial_scores = mutual_info_classif(X, y, discrete_features=is_disc, random_state=random_seed)
    # The cost based will substract lambda*cost for each item of initial_scores
    initial_scores_mcost = initial_scores - cost_param*cost_vec

    if MI_matrix is None:        
        # Compute all the pairwise mutual info depending on if the feature
        # is discrete or continuous
        matrix_MI = np.zeros( (num_features, num_features), dtype=float)
    
        for ii in range(num_features):
            if num_cores==1:
                if is_disc[ii] == True: # If the ii-th feature is discrete
                    # we use the classif version
                    matrix_MI[ii,:] = mutual_info_classif(X, X[:,ii], discrete_features=is_disc, random_state=random_seed)
                elif is_disc[ii] == False:
                    # otherwise we use the continuous (regression) version
                    matrix_MI[ii,:] = mutual_info_regression(X, X[:,ii], discrete_features=is_disc, random_state=random_seed)
             
            else:
                 if is_disc[ii] == True:
                     matrix_MI[ii,:] = Parallel(n_jobs=num_cores)(delayed(mutual_info_classif)(X[:,k].reshape(-1,1), X[:,ii], discrete_features=is_disc[k], random_state=random_seed) for k in range(num_features) )
                 elif is_disc[ii] == False:
                     matrix_MI[ii,:] = Parallel(n_jobs=num_cores)(delayed(mutual_info_regression)(X[:,k].reshape(-1,1), X[:,ii], discrete_features=is_disc[k], random_state=random_seed) for k in range(num_features) )

    else:
        matrix_MI = MI_matrix

    # ranking contains the indices of the final ranking in decreasing order of importance
    ranking = []

    ### The first selected feature is the one with the maximal penalized I(X_j, Y) value
    selected = np.argmax(initial_scores_mcost)
    ranking.append(selected)
    unRanked.pop(selected)

    # Until we have the desired number of selected_features, we apply the selection criterion
    for k in range(1,num_selected_features):
    
        featureRel = []
        # Compute the criterion to maximize for each unranked covariate
        for idx in unRanked:
                featureRel.append( initial_scores_mcost[idx] - np.mean(matrix_MI[ranking,idx]) )
    
        tmp_idx = np.argmax(featureRel)
        ranking.append(unRanked[tmp_idx])
        unRanked.pop(tmp_idx)
        
    return ranking, matrix_MI


def JMI(X, y, is_disc, cost_vec = None, cost_param = 0,
        num_features_to_select = None, random_seed = 123, num_cores = 1,
        MI_matrix = None, MI_conditional = None):
    """ Cost-based feature ranking based on Joint Mutual Information.
    
    Cost-based adaptation of the filter feature selection algorithm based on 
    Joint Mutual Information (Yang and Moody (1999)).
    
    H. H. Yang and J. Moody. Feature selection based on joint mutual information. 
    In Advances in intelligent data analysis, proceedings of international 
    ICSC symposium, pages 22—-25, 1999.

    Args:
        X (numpy.ndarray):
            the numerical features to use as training data, where 
            each row represents an individual, and each column a feature.
        y (list):
            a list of integers representing the training data labels.
        is_disc (list):
            a list of booleans indicating with True if the feature is discrete
            and False if continuous.
        cost_vec (numpy.ndarray):
            the vector of costs represented by a numpy.ndarray with shape
            (1, X.shape[1]). If None, the cost is set to zero for each feature.
        cost_param (float): 
            the positive cost penalization parameter. 0 by default.
        num_features_to_select (int):
            the number of best features to select. If unspecified, does not
            select a subset of features but keep all of them.
        random_seed (int):
            the random seed to use with the mutual_information function
            (when computing the Mutual Information (MI) involving one or more
            continuous features).
        num_cores (int):
            the number of CPU cores to use in parallel to compute the MI and JMI.
        MI_matrix (numpy.ndarray):
            the matrix of precomputed pairwise MI between pairs of features to
            save times when wanting to use multiple cost values. 
            By default this matrix is computed in the function.
        MI_conditional (dict):
            a dictionary that contains the precomputed numpy.ndarray of conditional
            pairwise MI between features, conditioned to the response values.
            Each key is a response modality, and each value is a conditional
            MI matrix between features I(X_i,X_j | y=key). Useful to save
            computational times when wanting to use multiple cost values, but 
            by default it is computed in the function.
            
    Returns:
        ranking (list):
            list containing the indices of the ranked features as specified in
            X, in decreasing order of importance.
        matrix_MI_Xk_Xj (numpy.ndarray):
            the matrix of precomputed MI between pairs of features.
        MI_condY (dict):
            a dictionary that contains the precomputed numpy.ndarray of conditional
            pairwise MI between features, conditioned to the response values.
            Each key is a response modality, and each value is a conditional
            MI matrix between features I(X_i,X_j | y=key).

    """
        
    num_features = X.shape[1]

    if (cost_vec is None):
        # If no cost is specified, then all costs are set as equal to zero
        cost_vec = np.zeros(num_features)
    #else:
    #    cost_vec=np.array(cost_vec)
    
    # Check on num_features_to_select
    if (num_features_to_select is not None):
        num_selected_features = min(num_features, num_features_to_select)
    else:
        num_selected_features = num_features
    
    # unRanked contains the feature indices unranked
    unRanked = list(range(num_features))
    
    # If a feature is discrete but with always different values, then
    # convert it into a continuous one
    # (to handle errors with the MI computation function)
    for featIdx in range(num_features):
        if is_disc[featIdx]==True and len(np.unique(X[:,featIdx])) == X.shape[0]:
            is_disc[featIdx] = False
    
    # Computing all the MIs I(X_j; y)
    initial_scores = mutual_info_classif(X, y, discrete_features=is_disc, random_state=random_seed)

    # The cost based will substract lambda*cost for each item of initial_scores
    initial_scores_mcost = initial_scores - cost_param*cost_vec

    if MI_matrix is None:
        # Compute all the pairwise mutual info depending on if the feature
        # is discrete or continuous
        matrix_MI_Xk_Xj = np.zeros( (num_features, num_features), dtype=float)
    
        for ii in range(num_features):
            if num_cores == 1:
                if is_disc[ii] == True: # If the ii-th feature is discrete
                    # we use the classif version
                    matrix_MI_Xk_Xj[ii,:] = mutual_info_classif(X, X[:,ii], discrete_features=is_disc, random_state = random_seed)
                elif is_disc[ii] == False:
                    # otherwise we use the continuous (regression) version
                    matrix_MI_Xk_Xj[ii,:] = mutual_info_regression(X, X[:,ii], discrete_features = is_disc, random_state = random_seed)
            
            else:
                 if is_disc[ii] == True:
                     matrix_MI_Xk_Xj[ii,:] = Parallel(n_jobs = num_cores)( delayed(mutual_info_classif)(X[:,k].reshape(-1,1), X[:,ii], discrete_features = is_disc[k], random_state = random_seed) for k in range(num_features) )
                 elif is_disc[ii] == False:
                     matrix_MI_Xk_Xj[ii,:] = Parallel(n_jobs = num_cores)( delayed(mutual_info_regression)(X[:,k].reshape(-1,1), X[:,ii], discrete_features = is_disc[k], random_state = random_seed) for k in range(num_features) )

    else:
        matrix_MI_Xk_Xj = MI_matrix


    # For the Joint mutual information, we also need to compute the matrices
    # I(Xk, Xj | Y=y) for y in Y

    # Extract the modalities in y
    yModalities = np.unique(y)

    # Create a dictionary that will contains the corresponding MI matrices
    # conditionally on the different unique values of y
    MI_condY = dict()

    # If not given, we compute it
    if MI_conditional is None:

        # For each modality of y
        for valY in yModalities:
                    
            # Initialize a new matrix
            matTmp = np.zeros( (num_features, num_features), dtype=float)
            # Extract the rows of X with this modality of Y
            subX = X[y==valY,]
            
            # proportion of this modality
            proValY = np.mean(y==valY)
            
            is_discForSubX = copy.deepcopy(is_disc)
            for featIdx in range(num_features):
                if is_disc[featIdx] == True and len(np.unique(subX[:,featIdx])) == subX.shape[0]:
                    is_discForSubX[featIdx] = False
                                
            # Fill the matrix
            for ii in range(num_features):
                if num_cores==1:
                    if is_discForSubX[ii] == True:
                        matTmp[ii,:] = proValY * mutual_info_classif(subX, subX[:,ii], discrete_features = is_discForSubX, random_state = random_seed)
                    elif is_discForSubX[ii] == False:
                        matTmp[ii,:] = proValY * mutual_info_regression(subX, subX[:,ii], discrete_features = is_discForSubX, random_state = random_seed)

                else:
                     if is_discForSubX[ii] == True:
                         vecToMultiply = Parallel(n_jobs=num_cores)( delayed(mutual_info_classif)(subX[:,k].reshape(-1,1), subX[:,ii], discrete_features = is_discForSubX[k], random_state = random_seed) for k in range(num_features) )
                         matTmp[ii,:] = [proValY * val for val in vecToMultiply]
                     elif is_discForSubX[ii] == False:
                         vecToMultiply = Parallel(n_jobs=num_cores)( delayed(mutual_info_regression)(subX[:,k].reshape(-1,1), subX[:,ii], discrete_features = is_discForSubX[k], random_state = random_seed) for k in range(num_features) )
                         matTmp[ii,:] = [proValY * val for val in vecToMultiply]
    
            MI_condY[valY] = matTmp

    else:
        MI_condY = MI_conditional

    # ranking contains the indices of the final ranking in decreasing order of importance
    ranking = []

    ### The first selected feature is the one with the maximal penalized I(X_j, Y) value
    selected = np.argmax(initial_scores_mcost)
    ranking.append(selected)
    unRanked.pop(selected)

    # Until we have the desired number of selected_features, we apply the selection criterion
    for k in range(1,num_selected_features):
    
        featureRel = []
        # Compute the criterion to maximize for each unranked covariate
        for idx in unRanked:
            vecSummed = np.zeros(len(ranking))
            for valY in yModalities:
                # Compute I(Xk; Xj | Y)
                vecSummed += MI_condY[valY][ranking,idx]
            
            criterionVal = initial_scores_mcost[idx] - np.mean(matrix_MI_Xk_Xj[ranking,idx]) + np.mean(vecSummed)
            
            featureRel.append(criterionVal)
    
        tmp_idx = np.argmax(featureRel)
        ranking.append(unRanked[tmp_idx])
        unRanked.pop(tmp_idx)
        
    return ranking, matrix_MI_Xk_Xj, MI_condY


def JMIM(X, y, is_disc, cost_vec = None, cost_param = 0,
         num_features_to_select = None, random_seed = 123, num_cores = 1,
         MI_matrix = None, MI_conditional = None):
    """ Cost-based feature ranking based on Joint Mutual Information Maximization.
    
    Cost-based adaptation of the filter feature selection algorithm based on 
    Joint Mutual Information Maximization (Bennasar et al. (2015)).
    
    M. Bennasar, Y. Hicks, and R. Setchi. Feature selection using Joint Mutual 
    Information Maximisation. Expert Systems With Applications, 42:8520–8532, 
    2015.
    
    Args:
        X (numpy.ndarray):
            the numerical features to use as training data, where 
            each row represents an individual, and each column a feature.
        y (list):
            a list of integers representing the training data labels.
        is_disc (list):
            a list of booleans indicating with True if the feature is discrete
            and False if continuous.
        cost_vec (numpy.ndarray):
            the vector of costs represented by a numpy.ndarray with shape
            (1, X.shape[1]). If None, the cost is set to zero for each feature.
        cost_param (float): 
            the positive cost penalization parameter. 0 by default.
        num_features_to_select (int):
            the number of best features to select. If unspecified, does not
            select a subset of features but keep all of them.
        random_seed (int):
            the random seed to use with the mutual_information function
            (when computing the Mutual Information (MI) involving one or more
            continuous features).
        num_cores (int):
            the number of CPU cores to use in parallel to compute the MI and JMI.
        MI_matrix (numpy.ndarray):
            the matrix of precomputed pairwise MI between pairs of features to
            save times when wanting to use multiple cost values. 
            By default this matrix is computed in the function.
        MI_conditional (dict):
            a dictionary that contains the precomputed numpy.ndarray of conditional
            pairwise MI between features, conditioned to the response values.
            Each key is a response modality, and each value is a conditional
            MI matrix between features I(X_i,X_j | y=key). Useful to save
            computational times when wanting to use multiple cost values, but 
            by default it is computed in the function.
            
    Returns:
        ranking (list):
            list containing the indices of the ranked features as specified in
            X, in decreasing order of importance.
        matrix_MI_Xk_Xj (numpy.ndarray):
            the matrix of precomputed MI between pairs of features.
        MI_condY (dict):
            a dictionary that contains the precomputed numpy.ndarray of conditional
            pairwise MI between features, conditioned to the response values.
            Each key is a response modality, and each value is a conditional
            MI matrix between features I(X_i,X_j | y=key).
            
    """
        
    num_features = X.shape[1]

    if (cost_vec is None):
        # If no cost is specified, then all costs are set as equal to zero
        cost_vec = np.zeros(num_features)
    #else:
    #    cost_vec=np.array(cost_vec)
    
    # Check on num_features_to_select
    if (num_features_to_select is not None):
        num_selected_features = min(num_features, num_features_to_select)
    else:
        num_selected_features = num_features
    
    # unRanked contains the feature indices unranked
    unRanked = list(range(num_features))
    
    for featIdx in range(num_features):
        if is_disc[featIdx]==True and len(np.unique(X[:,featIdx])) == X.shape[0]:
            is_disc[featIdx] = False
    
    initial_scores = mutual_info_classif(X, y, discrete_features=is_disc, random_state=random_seed)
    initial_scores_mcost = initial_scores - cost_param*cost_vec

    if MI_matrix is None:
        # Compute all the pairwise mutual info depending on if the feature
        # is discrete or continuous
        matrix_MI_Xk_Xj = np.zeros( (num_features, num_features), dtype=float)
    
        for ii in range(num_features):
            if num_cores==1:
                if is_disc[ii] == True: # If the ii-th feature is discrete
                    # we use the classif version
                    matrix_MI_Xk_Xj[ii,:] = mutual_info_classif(X, X[:,ii], discrete_features=is_disc, random_state=random_seed)
                elif is_disc[ii] == False:
                    # otherwise we use the continuous (regression) version
                    matrix_MI_Xk_Xj[ii,:] = mutual_info_regression(X, X[:,ii], discrete_features=is_disc, random_state=random_seed)
            else:
                 if is_disc[ii] == True:
                     matrix_MI_Xk_Xj[ii,:] = Parallel(n_jobs = num_cores)( delayed(mutual_info_classif)(X[:,k].reshape(-1,1), X[:,ii], discrete_features = is_disc[k], random_state = random_seed) for k in range(num_features) )
                 elif is_disc[ii] == False:
                     matrix_MI_Xk_Xj[ii,:] = Parallel(n_jobs = num_cores)( delayed(mutual_info_regression)(X[:,k].reshape(-1,1), X[:,ii], discrete_features = is_disc[k], random_state = random_seed) for k in range(num_features) )

    else:
        matrix_MI_Xk_Xj = MI_matrix


    # For the Joint mutual information, we also need to compute the matrices
    # I(Xk, Xj | Y=y) for y in Y

    # Extract the modalities in y
    yModalities = np.unique(y)

    # Create a dictionary that will contains the corresponding MI matrices
    # conditionally on the different unique values of y
    MI_condY = dict()

    # If not given, we compute it
    if MI_conditional is None:
        
        # For each modality of y
        for valY in yModalities:
                    
            # Initialize a new matrix
            matTmp = np.zeros( (num_features, num_features), dtype=float)
            # Extract the rows of X with this modality of Y
            subX = X[y==valY,]
            
            # proportion of this modality
            proValY = np.mean(y==valY)
            
            is_discForSubX = copy.deepcopy(is_disc)
            for featIdx in range(num_features):
                if is_disc[featIdx]==True and len(np.unique(subX[:,featIdx])) == subX.shape[0]:
                    is_discForSubX[featIdx] = False
            
            # Fill the matrix
            for ii in range(num_features):
                if num_cores==1:
                    if is_discForSubX[ii] == True: # If the ii-th feature is discrete
                        # we use the classif version
                        matTmp[ii,:] = proValY * mutual_info_classif(subX, subX[:,ii], discrete_features=is_discForSubX, random_state=random_seed)
                    elif is_discForSubX[ii] == False:
                        # otherwise we use the continuous (regression) version
                        matTmp[ii,:] = proValY * mutual_info_regression(subX, subX[:,ii], discrete_features=is_discForSubX, random_state=random_seed)
                
                else:
                     if is_discForSubX[ii] == True:
                         vecToMultiply = Parallel(n_jobs=num_cores)( delayed(mutual_info_classif)(subX[:,k].reshape(-1,1), subX[:,ii], discrete_features = is_discForSubX[k], random_state = random_seed) for k in range(num_features) )
                         matTmp[ii,:] = [proValY * val for val in vecToMultiply]
                     elif is_discForSubX[ii] == False:
                         vecToMultiply = Parallel(n_jobs=num_cores)( delayed(mutual_info_regression)(subX[:,k].reshape(-1,1), subX[:,ii], discrete_features = is_discForSubX[k], random_state = random_seed) for k in range(num_features) )
                         matTmp[ii,:] = [proValY * val for val in vecToMultiply]
                        
            MI_condY[valY] = matTmp
    else:
        MI_condY = MI_conditional
    
    # ranking contains the indices of the final ranking in decreasing order of importance
    ranking = []

    ### The first selected feature is the one with the maximal penalized I(X_j, Y) value    
    selected = np.argmax(initial_scores_mcost)
    ranking.append(selected)
    unRanked.pop(selected)

    # Until we have the desired number of selected_features, we apply the selection criterion
    for k in range(1,num_selected_features):

        featureRel = []
        # Compute the criterion to maximize for each unranked covariate
        for idx in unRanked:
            vecSummed = np.zeros(len(ranking))
            for valY in yModalities:
                vecSummed += MI_condY[valY][ranking,idx]
            
            criterionVal = np.min( initial_scores[ranking] - matrix_MI_Xk_Xj[ranking,idx] + vecSummed ) + initial_scores_mcost[idx]
            # J(Xk) = min_j [ I(Xj;Y) - I(Xk;Xj) + I(Xk;Xj|Y) ] + (I(Xk;Y) - lambda * costk)
            
            featureRel.append( criterionVal )
        
        tmp_idx = np.argmax(featureRel)
        ranking.append(unRanked[tmp_idx])
        unRanked.pop(tmp_idx)
        
    return ranking, matrix_MI_Xk_Xj, MI_condY


def reliefF(X, y, cost_vec = None, cost_param = 0, num_neighbors = 10, num_features_to_select = None,
            proximity = "distance", min_samples_leaf = 100, n_estimators = 500,
            sim_matrix = None):
    """ Cost-based feature ranking adaptation of the ReliefF algorithm.
    
    Cost-based adaptation of the ReliefF algorithm, where the nearest neighbors
    of each data can be identified either using a classic L1 distance, or a
    random forest proximity matrix.
    
    I. Kononenko. Estimating attributes: Analysis and extensions of relief.
    In F. Bergadano and L. De Raedt, editors, Machine Learning: ECML-94, 
    pages 171–182, Berlin, Heidelberg, 1994. Springer Berlin Heidelberg.
        
    Args:
        X (numpy.ndarray):
            the numerical features to use as training data, where 
            each row represents an individual, and each column a feature.
        y (list):
            a list of integers representing the training data labels.
        cost_vec (numpy.ndarray):
            the vector of costs represented by a numpy.ndarray with shape
            (1, X.shape[1]). If None, the cost is set to zero for each feature.
        cost_param (float):
            the positive cost penalization parameter. 0 by default.
        num_neighbors (int):
            the number of nearest neighbors. 10 by default.
        num_features_to_select (int):
            the number of best features to select. If unspecified, does not
            select a subset of features but keep all of them.
        proximity (str):
            a string that is either "distance" to use the classic version of
            reliefF, or "rf prox" to use the random forest proximity between
            data to deduce the neighbors. "distance" by default.
        min_samples_leaf (int):
            when using proximity = "rf prox", the minimum number of samples 
            required to split an internal node. 100 by default.
        n_estimators (int):
            the number of trees in the random forest. Only relevant when
            proximity = "rf prox". 500 by default.
        sim_matrix (numpy.ndarray):
            the precomputed matrix of pairwise similarity between data,
            either distance or random forest proximity. This argument is
            returned to speed up the analysis when working with multiple
            cost_param values.
            
    Returns:
        ranking (list):
            a list containing the indices of the ranked features as
            specified in X, in decreasing order of importance.
        weights (dict):
            a dictionary with as keys the covariate index, and as values the
            corresponding scores used to obtain the ranking.
        sim_matrix (numpy.ndarray):
            the pairwise distance/proximity matrix used.
            
    """

    y = np.array(y)
    nTrain = X.shape[0]
    nCov = X.shape[1]

    if proximity not in ['distance', 'rf prox']:
        raise ValueError("The argument proximity must be either 'distance' or 'rf prox'.")
    
    if (cost_vec is None):
        # If no cost is specified, then all costs are set as equal to zero
        cost_vec = np.zeros(nCov)
    
    # Check on num_features_to_select
    if (num_features_to_select is not None):
        num_features_to_select = min(nCov, num_features_to_select)
    else:
        num_features_to_select = nCov
    
    # Data standardization
    X_std = copy.deepcopy(X)
    cov_means = np.mean(X, axis=0)
    cov_std = np.std(X, axis=0)
    for i in range(X.shape[1]):
        X_std[:,i] = (X[:,i] - cov_means[i])/cov_std[i]
    
    # Determine the number/proportion of classes in y
    classes = np.unique(y)
    nClasses = len(classes)
    pClasses = collections.Counter(y)
    nbrData = np.sum(list(pClasses.values()))
    for cLab in pClasses:
        pClasses[cLab] = pClasses[cLab]/nbrData
    
    # Compute for each covariate the max and min values. Useful for L1 dist.
    maxXVal = np.max(X_std, axis=0)
    minXVal = np.min(X_std, axis=0)

    # If we use the classic (Manhattan) distance:
    if proximity == "distance":
        if sim_matrix is None:
            # Compute all pairs of distance between training data
            distMat = np.zeros(shape = (nTrain,nTrain), dtype=float)
            
            for i in range(nTrain-1):
                for j in range(i+1,nTrain,1):
                    distMat[i,j] = _private_man_dist(X_std[i,:], X_std[j,:], minXVal, maxXVal)
                    distMat[j,i] = distMat[i,j]

        else:
            distMat = sim_matrix
        
    # If we use the RF proximity matrix instead of classic distance:
    if proximity == "rf prox":
        if sim_matrix is None:
            # Train a random forest and deduce the proximity matrix
            model = RandomForestClassifier(n_estimators = n_estimators, 
                                           min_samples_leaf = min_samples_leaf)
            model.fit(X_std, y)
            proxMatRF = _private_proximity_matrix(model, X_std, normalize=True)
            proxMat = proxMatRF
        else:
            proxMat = sim_matrix
                
    # For each training data R_i:
    # Search for k nearest hits
    # Search, for each class different than R_i's, the k nearest misses
    
    # To store the indices of the nearest hits
    kNearHits = np.zeros(num_neighbors, dtype=int)
    # To store the indices of the misses for all class different than R_i
    kNearMisses = np.zeros( (nClasses-1,num_neighbors), dtype=int)

    # Initialize the weights to zero
    weightsDic = dict()
    for cov in range(nCov):
        weightsDic[cov] = 0
    
    m = nTrain # Here we compute the score using all the training data
    for i in range(m):
        # For the same class that R_i, keep the indices achieving the k lower distances
        if proximity == "distance":
            argSorted = np.argsort(distMat[i,y==y[i]]) # We withdraw the i-th element
            kNearHits = argSorted[argSorted != i][0:num_neighbors]
            classDifRi = classes[classes!=y[i]]
            for c in range(len(classDifRi)):
                tmp = classDifRi[c]
                kNearMisses[c,:] = np.argsort(distMat[i,y==tmp])[0:num_neighbors]
                
        if proximity == "rf prox":
            argSorted = np.argsort(-proxMat[i,y==y[i]]) # We withdraw the i-th element
            kNearHits = argSorted[argSorted != i][0:num_neighbors]
            classDifRi = classes[classes!=y[i]]
            for c in range(len(classDifRi)):
                tmp = classDifRi[c]
                kNearMisses[c,:] = np.argsort(-proxMat[i,y==tmp])[0:num_neighbors]

        # Compute the elements diff(A, R_i, H_j) for j in 1:k, per feature A
        for cov in range(nCov):
            compDistRiFromHits = [ np.abs(X_std[i,cov] - X_std[hit,cov])/(maxXVal[cov] - minXVal[cov])
                                    for hit in kNearHits ]
            weightsDic[cov] -= np.mean(compDistRiFromHits)/m
            
            # For each class different from the one of R_i, do the same with 
            # weight by prior proba ratio
            for c in range(len(classDifRi)): 
                tmp = classDifRi[c]
                compDistRiFromMisses = [ np.abs(X_std[i,cov] - X_std[miss,cov])/(maxXVal[cov] - minXVal[cov])
                                        for miss in kNearMisses[c] ]
                
                # Reminder: pClasses is a dictionary
                weightsDic[cov] += ( pClasses[tmp]/(1-pClasses[y[i]]) ) * np.mean(compDistRiFromMisses)/m
 
            # Finally also update with the penalization (cost)
            # I do not use the /(m*k) term but only /m to be more consistent 
            # with the other criteria of this module.
            weightsDic[cov] -= cost_param*cost_vec[cov]/(m)
            
    # Return the number of feature requested, in decreasing order, plus weights
    ranking = np.argsort(-np.array(list(weightsDic.values())))[:num_features_to_select]      
    ranking = ranking.tolist()
    
    if proximity == "distance":
        return ranking, weightsDic, distMat
    elif proximity == "rf prox":
        return ranking, weightsDic, proxMat

    
def _private_man_dist(inst1, inst2, minXVal, maxXVal):
    """ Compute the Manhattan distance between two set of covariates. """
    
    nCov = len(inst1)
    dist_comp = [ np.abs(inst1[i] - inst2[i])/(maxXVal[i] - minXVal[i])
                for i in range(nCov) ]
    dist = np.sum(dist_comp)
    return dist

def _private_proximity_matrix(model, X, normalize=True):
    """ Compute the random forest proximity matrix. """
    
    terminals = model.apply(X)
    nTrees = terminals.shape[1]
    a = terminals[:,0]
    proxMat = 1*np.equal.outer(a, a)

    for i in range(1, nTrees):
        a = terminals[:,i]
        proxMat += 1*np.equal.outer(a, a)

    if normalize:
        proxMat = proxMat / nTrees

    return proxMat


def pen_rf_importance(X, y, cost_vec = None, cost_param = 0, num_features_to_select = None,
                      imp_type = "impurity", min_samples_leaf = 1,
                      n_estimators = 500, rf_importance_vec = None):
    """ Cost-based feature ranking with penalized random forest importance.
    
    The cost-based ranking of the features are deduced by penalizing the
    random forest importance by the feature costs.
    
    Args:
        X (numpy.ndarray):
            the numerical features to use as training data, where 
            each row represents an individual, and each column a feature.
        y (list):
            a list of integers representing the training data labels.
        cost_vec (numpy.ndarray):
            the vector of costs represented by a numpy.ndarray with shape
            (1, X.shape[1]). If None, the cost is set to zero for each feature.
        cost_param (float): 
            the positive cost penalization parameter. 0 by default.
        num_features_to_select (int):
            the number of best features to select. If unspecified, does not
            select a subset of features but keep all of them.
        imp_type (str):
            a string, either "impurity" or "permutation", to use the 
            random forest importance based on the decrease of the impurity
            measure (MDI), or based on the decrease of accuracy due to random permutation
            of the covariate values (MDA). "impurity" by default.
        min_samples_leaf (int):
            the minimum number of samples required to split an internal node.
            1 by default.
        n_estimators (int):
            the number of trees in the random forest. 500 by default.
        rf_importance_vec (numpy.ndarray):
            an array that contains the precomputed unpenalized random forest 
            importance. Useful when analyzing the rankings for different 
            cost_parameter value, to reduce the computational time.
            
    Returns:
        ranking (list):
            a list containing the indices of the ranked features as
            specified in X, in decreasing order of importance.
        unpenalized_rf_importance (numpy.ndarray):
            an array that contains the computed UNPENALIZED random forest
            importance. This might be used to reduce the computational time
            when implementing a version with multiple cost_parameter values.
            
    """

    nCov = X.shape[1]

    if imp_type not in ['impurity', 'permutation']:
        raise ValueError("The argument imp_type must be either 'impurity' or 'permutation'.")
        
    if (cost_vec is None):
        # If no cost is specified, then all costs are set as equal to zero
        cost_vec = np.zeros(nCov)

    # Check on num_features_to_select
    if (num_features_to_select is not None):
        num_features_to_select = min(nCov, num_features_to_select)
    else:
        num_features_to_select = nCov

    if(rf_importance_vec is None):
        # For format compatibility between python and R (rpy2)
        from rpy2.robjects import numpy2ri
        numpy2ri.activate()
    
        rpy2.robjects.globalenv["X_train"] = X
        rpy2.robjects.globalenv["y_train"] = y
        rpy2.robjects.globalenv["imp_type"] = imp_type
        rpy2.robjects.globalenv["min_samples_leaf"] = min_samples_leaf
        rpy2.robjects.globalenv["n_estimators"] = n_estimators
            
        unpenalized_rf_importance = rpy2.robjects.r('''                            
        # Check if ranger is installed
        packages = c("ranger")
        package.check <- lapply(
                packages,
                FUN = function(x) {
                        if (!require(x, character.only = TRUE)) {
                                install.packages(x, dependencies = TRUE)
                                library(x, character.only = TRUE)
                        }
                      })
        # Determine the importance
        library(ranger)
        trainedRF <- ranger(x=as.data.frame(X_train), y = as.numeric(y_train),
                            classification = TRUE, importance = imp_type, 
                            num.trees = n_estimators, min.node.size = min_samples_leaf,
                            num.threads = 1)
        trainedRF$variable.importance
        ''')
        
        numpy2ri.deactivate()

    else:
        unpenalized_rf_importance = copy.deepcopy(rf_importance_vec)
            
    rf_importance_copy = copy.deepcopy(unpenalized_rf_importance)
    
    # To facilitate the comparison between different types of importance,
    # we set values between 0 and 1, and to sum to 1.
    rf_importance_copy = (np.array(rf_importance_copy)-np.min(rf_importance_copy))/(np.max(rf_importance_copy) - np.min(rf_importance_copy))
    rf_importance_copy = rf_importance_copy/np.sum(rf_importance_copy)
    
    for cov in range(nCov):
        rf_importance_copy[cov] -= cost_param * cost_vec[cov]    
    
    ranking = np.argsort(-rf_importance_copy)[:num_features_to_select]      
    ranking = ranking.tolist()
        
    return ranking, unpenalized_rf_importance


def weighted_rf_importance(X, y, cost_vec = None, cost_param = 0, num_features_to_select = None,
                           imp_type = "impurity", min_samples_leaf = 1,
                           n_estimators = 500):
    """ Cost-based feature ranking using weighted random forest importance.
    
    The cost-based ranking of the features are deduced using the feature
    importance of a weighted random forest, where the probability of sampling
    a covariate at a given node is proportional to 1/(cost)^cost_param.
            
    Args:
        X (numpy.ndarray):
            the numerical features to use as training data, where 
            each row represents an individual, and each column a feature.
        y (list):
            a list of integers representing the training data labels.
        cost_vec (numpy.ndarray):
            the vector of costs represented by a numpy.ndarray with shape
            (1, X.shape[1]). If None, the cost is set to zero for each feature.
        cost_param (float): 
            the positive cost penalization parameter. 0 by default.
        num_features_to_select (int):
            the number of best features to select. If unspecified, does not
            select a subset of features but keep all of them.
        imp_type (str):
            a string, either "impurity" or "permutation", to use the 
            random forest importance based on the decrease of the impurity
            measure (MDI), or based on the decrease of accuracy due to random permutation
            of the covariate values (MDA). "impurity" by default.
        min_samples_leaf (int):
            the minimum number of samples required to split an internal node.
            1 by default.
        n_estimators (int):
            the number of trees in the random forest. 500 by default.
    
    Returns:
        ranking (list):
            a list containing the indices of the ranked features as
            specified in X, in decreasing order of importance.
            
    """

    nCov = X.shape[1]

    if imp_type not in ['impurity', 'permutation']:
        raise ValueError("The argument imp_type must be either 'impurity' or 'permutation'.")
    
    if (cost_vec is None):
        # If no cost is specified, then all costs are set as equal to zero
        cost_vec = np.zeros(nCov)
    
    # Check on num_features_to_select
    if (num_features_to_select is not None):
        num_features_to_select = min(nCov, num_features_to_select)
    else:
        num_features_to_select = nCov    

    # Compute the rf weights for sampling the covariates
    # Note, a base importance of 0.01 is added to all features to avoid num. errors
    sampling_weights = ( 1/(cost_vec+0.01)**cost_param) / (np.sum(1/(cost_vec+0.01)**cost_param))
    
    # For format compatibility between python and R (rpy2)
    from rpy2.robjects import numpy2ri
    numpy2ri.activate()
    
    rpy2.robjects.globalenv["X_train"] = X
    rpy2.robjects.globalenv["y_train"] = y
    rpy2.robjects.globalenv["imp_type"] = imp_type
    rpy2.robjects.globalenv["min_samples_leaf"] = min_samples_leaf
    rpy2.robjects.globalenv["n_estimators"] = n_estimators
    rpy2.robjects.globalenv["sampling_weights"] = sampling_weights
            
    weighted_rf_importance = rpy2.robjects.r('''                             
    # Check if ranger is installed
    packages = c("ranger")
    package.check <- lapply(
            packages,
            FUN = function(x) {
                    if (!require(x, character.only = TRUE)) {
                            install.packages(x, dependencies = TRUE)
                            library(x, character.only = TRUE)}
            }
            )
    # Determine the importance
    library(ranger)
    trainedWeightedRF <- ranger(x=as.data.frame(X_train), y = as.numeric(y_train),
                        classification = TRUE, importance = imp_type, 
                        num.trees = n_estimators, min.node.size = min_samples_leaf,
                        num.threads = 1, split.select.weights = as.numeric(sampling_weights))
    trainedWeightedRF$variable.importance
    ''')

    numpy2ri.deactivate()
          
    ranking = np.argsort(-weighted_rf_importance)[:num_features_to_select]      
    ranking = ranking.tolist()
    
    return ranking

    
def multi_mRMR(X, y, is_disc, cost_vec, cost_param_vec, 
               num_features_to_select = None, random_seed = 123, num_cores = 1):
    """ Cost-based mRMR feature ranking with multiple penalization parameters.
    
    Function to obtain the rankings associated to different cost parameters,
    with the Maximal-Relevance-Minimal-Redundancy (mRMR) method.

    H. Peng, F. Long, and C. Ding.  Feature Selection Based on Mutual 
    Information: Criteria of Max-Dependency, Max-Relevance, and Min-Redundancy.
    IEEE Transactions on pattern analysis and machine intelligence, 
    27:1226–1238, 2005.
    
    Args:
        X (numpy.ndarray):
            the numerical features to use as training data, where 
            each row represents an individual, and each column a feature.
        y (list):
            a list of integers representing the training data labels.
        is_disc (list):
            a list of booleans indicating with True if the feature is discrete
            and False if continuous.
        cost_vec (numpy.ndarray):
            the vector of costs represented by a numpy.ndarray with shape
            (1, X.shape[1]). If None, the cost is set to zero for each feature.
        cost_param (float): 
            the positive cost penalization parameter. 0 by default.
        num_features_to_select (int):
            the number of best features to select. If unspecified, does not
            select a subset of features but keep all of them.
        random_seed (int):
            the random seed to use with the mutual_information function
            (when computing the Mutual Information (MI) involving one or more
            continuous features).
        num_cores (int):
            the number of CPU cores to use in parallel to compute the MI.

    Returns:
        dfPen_Ranking (pandas.core.frame.DataFrame):
            a pandas DataFrame that contains in each row, a penalization
            parameter value and the corresponding ranking indexes, in decreasing
            order of importance.
            
    """
    
    MI_matrix = None
    grid_size = len(cost_param_vec)
    nCov = X.shape[1]

    # Check on num_features_to_select
    if (num_features_to_select is not None):
        num_features_to_select = min(nCov, num_features_to_select)
    else:
        num_features_to_select = nCov
    
    matRanking = np.zeros( (grid_size, num_features_to_select) )
    
    k = 0
    for cost_param in cost_param_vec:
        rankedIdx, MI_matrix = mRMR(X = X, y = y, is_disc = is_disc, 
                                        cost_vec = cost_vec, cost_param = cost_param,
                                        num_features_to_select = num_features_to_select, 
                                        random_seed = random_seed, num_cores = num_cores, MI_matrix = MI_matrix)
        matRanking[k,:] = rankedIdx
        k += 1
    
    output = {"cost_param": cost_param_vec}
    dfPen_Ranking = pd.DataFrame(output)
    
    dfRanking = pd.DataFrame(matRanking, columns=range(1, num_features_to_select+1))
    dfPen_Ranking = pd.concat([dfPen_Ranking, dfRanking], axis=1)
    
    return dfPen_Ranking


def multi_JMI(X, y, is_disc, cost_vec, cost_param_vec,
              num_features_to_select = None, random_seed = 123, num_cores = 1):
    """ Cost-based JMI feature ranking with multiple penalization parameters.
    
    Function to obtain the rankings associated to different cost parameters,
    with the filter feature selection algorithm based on Joint Mutual 
    Information (Yang and Moody (1999)).
    
    H. H. Yang and J. Moody. Feature selection based on joint mutual information. 
    In Advances in intelligent data analysis, proceedings of international 
    ICSC symposium, pages 22—-25, 1999.
    
    Args:
        X (numpy.ndarray):
            the numerical features to use as training data, where 
            each row represents an individual, and each column a feature.
        y (list):
            a list of integers representing the training data labels.
        is_disc (list):
            a list of booleans indicating with True if the feature is discrete
            and False if continuous.
        cost_vec (numpy.ndarray):
            the vector of costs represented by a numpy.ndarray with shape
            (1, X.shape[1]). If None, the cost is set to zero for each feature.
        cost_param (float): 
            the positive cost penalization parameter. 0 by default.
        num_features_to_select (int):
            the number of best features to select. If unspecified, does not
            select a subset of features but keep all of them.
        random_seed (int):
            the random seed to use with the mutual_information function
            (when computing the Mutual Information (MI) involving one or more
            continuous features).
        num_cores (int):
            the number of CPU cores to use in parallel to compute the MI and JMI.
            
    Returns:
        dfPen_Ranking (pandas.core.frame.DataFrame):
            a pandas DataFrame that contains in each row, a penalization
            parameter value and the corresponding ranking indexes, in decreasing
            order of importance.
            
    """
    
    MI_matrix = None
    MI_conditional = None
    grid_size = len(cost_param_vec)
    nCov = X.shape[1]

    # Check on num_features_to_select
    if (num_features_to_select is not None):
        num_features_to_select = min(nCov, num_features_to_select)
    else:
        num_features_to_select = nCov    
    
    matRanking = np.zeros( (grid_size, num_features_to_select) )
    
    k = 0
    for cost_param in cost_param_vec:
        rankedIdx, MI_matrix, MI_conditional = JMI(X = X, y = y, is_disc = is_disc, 
                                                       cost_vec = cost_vec, cost_param = cost_param,
                                                       num_features_to_select = num_features_to_select, 
                                                       random_seed = random_seed, num_cores = num_cores,
                                                       MI_matrix = MI_matrix, MI_conditional = MI_conditional)
        matRanking[k,:] = rankedIdx
        k += 1
    
    output = {"cost_param": cost_param_vec}
    dfPen_Ranking = pd.DataFrame(output)
    
    dfRanking = pd.DataFrame(matRanking, columns=range(1, num_features_to_select+1))
    dfPen_Ranking = pd.concat([dfPen_Ranking, dfRanking], axis=1)
    
    return dfPen_Ranking


def multi_JMIM(X, y, is_disc, cost_vec, cost_param_vec, 
               num_features_to_select = None, random_seed = 123, num_cores = 1):
    """ Cost-based JMIM feature ranking with multiple penalization parameters
    
    Function to obtain the rankings associated to different cost parameters,
    with the filter feature selection algorithm based on 
    Joint Mutual Information Maximization (Bennasar et al. (2015)).
    
    M. Bennasar, Y. Hicks, and R. Setchi. Feature selection using Joint Mutual 
    Information Maximisation. Expert Systems With Applications, 42:8520–8532, 
    2015.
    
    Args:
        X (numpy.ndarray):
            the numerical features to use as training data, where 
            each row represents an individual, and each column a feature.
        y (list):
            a list of integers representing the training data labels.
        is_disc (list):
            a list of booleans indicating with True if the feature is discrete
            and False if continuous.
        cost_vec (numpy.ndarray):
            the vector of costs represented by a numpy.ndarray with shape
            (1, X.shape[1]). If None, the cost is set to zero for each feature.
        cost_param (float): 
            the positive cost penalization parameter. 0 by default.
        num_features_to_select (int):
            the number of best features to select. If unspecified, does not
            select a subset of features but keep all of them.
        random_seed (int):
            the random seed to use with the mutual_information function
            (when computing the Mutual Information (MI) involving one or more
            continuous features).
        num_cores (int):
            the number of CPU cores to use in parallel to compute the MI and JMI.
            
    Returns:
        dfPen_Ranking (pandas.core.frame.DataFrame):
            a pandas DataFrame that contains in each row, a penalization
            parameter value and the corresponding ranking indexes, in decreasing
            order of importance.
            
    """
    
    MI_matrix = None
    MI_conditional = None
    grid_size = len(cost_param_vec)
    nCov = X.shape[1]

    # Check on num_features_to_select
    if (num_features_to_select is not None):
        num_features_to_select = min(nCov, num_features_to_select)
    else:
        num_features_to_select = nCov    
    
    matRanking = np.zeros( (grid_size, num_features_to_select) )
    
    k = 0
    for cost_param in cost_param_vec:
        rankedIdx, MI_matrix, MI_conditional = JMIM(X = X, y = y, is_disc = is_disc, 
                                                    cost_vec = cost_vec, cost_param = cost_param,
                                                    num_features_to_select = num_features_to_select, 
                                                    random_seed = random_seed, num_cores = num_cores,
                                                    MI_matrix = MI_matrix, MI_conditional = MI_conditional)
        matRanking[k,:] = rankedIdx
        k += 1
        
    output = {"cost_param": cost_param_vec}
    dfPen_Ranking = pd.DataFrame(output)
    
    dfRanking = pd.DataFrame(matRanking, columns=range(1, num_features_to_select+1))
    dfPen_Ranking = pd.concat([dfPen_Ranking, dfRanking], axis=1)
    
    return dfPen_Ranking


def multi_reliefF(X, y, cost_vec, cost_param_vec, num_neighbors = 10, num_features_to_select = None,
                 proximity = "distance", min_samples_leaf = 100, n_estimators = 500):
    """ Cost-based ReliefF algorithm with multiple penalization parameters.
    
    Cost-based adaptation of the ReliefF algorithm, where the nearest neighbors
    of each data can be identified either using a classic L1 distance, or a
    random forest proximity matrix.
    
    I. Kononenko. Estimating attributes: Analysis and extensions of relief.
    In F. Bergadano and L. De Raedt, editors, Machine Learning: ECML-94, 
    pages 171–182, Berlin, Heidelberg, 1994. Springer Berlin Heidelberg.
        
    Args:
        X (numpy.ndarray):
            the numerical features to use as training data, where 
            each row represents an individual, and each column a feature.
        y (list):
            a list of integers representing the training data labels.
        cost_vec (numpy.ndarray):
            the vector of costs represented by a numpy.ndarray with shape
            (1, X.shape[1]). If None, the cost is set to zero for each feature.
        cost_param_vec (numpy.ndarray): 
            the vector of the cost penalization parameters for which the 
            corresponding rankings will be computed.
        num_neighbors (int):
            the number of nearest neighbors. 10 by default.
        num_features_to_select (int):
            the number of best features to select. If unspecified, does not
            select a subset of features but keep all of them.
        proximity (str):
            a string that is either "distance" to use the classic version of
            reliefF, or "rf prox" to use the random forest proximity between
            data to deduce the neighbors. "distance" by default.
        min_samples_leaf (int):
            when using proximity = "rf prox", the minimum number of samples 
            required to split an internal node. 100 by default.
        n_estimators (int):
            the number of trees in the random forest. Only relevant when
            proximity = "rf prox". 500 by default.
                        
    Returns:
        dfPen_Ranking (pandas.core.frame.DataFrame):
            a pandas DataFrame that contains in each row, a penalization
            parameter value and the corresponding ranking indexes, in decreasing
            order of importance.
            
    """
    
    sim_matrix = None
    grid_size = len(cost_param_vec)
    nCov = X.shape[1]

    # Check on num_features_to_select
    if (num_features_to_select is not None):
        num_features_to_select = min(nCov, num_features_to_select)
    else:
        num_features_to_select = nCov    
    
    matRanking = np.zeros( (grid_size, num_features_to_select) )
    
    k = 0
    for cost_param in cost_param_vec:
        rankedIdx, scores, sim_matrix  = reliefF(X = X, y = y, cost_vec = cost_vec, cost_param = cost_param, num_neighbors = num_neighbors,
                                                 num_features_to_select = num_features_to_select, proximity = proximity, 
                                                 min_samples_leaf = min_samples_leaf, n_estimators = n_estimators,
                                                 sim_matrix = sim_matrix)
        matRanking[k,:] = rankedIdx
        k += 1
    
    output = {"cost_param": cost_param_vec}
    dfPen_Ranking = pd.DataFrame(output)
    
    dfRanking = pd.DataFrame(matRanking, columns=range(1, num_features_to_select+1))
    dfPen_Ranking = pd.concat([dfPen_Ranking, dfRanking], axis=1)
    
    return dfPen_Ranking


def multi_pen_rf_importance(X, y, cost_vec, cost_param_vec, num_features_to_select = None,
                            imp_type = "impurity", min_samples_leaf = 1,
                            n_estimators = 500):
    """ Cost-based feature ranking with penalized random forest importance, with multiple penalization values.
    
    The cost-based ranking of the features are deduced by penalizing the
    random forest importance by the feature costs. This function allows the use
    of multiple penalization parameter values.
            
    Args:
        X (numpy.ndarray):
            the numerical features to use as training data, where 
            each row represents an individual, and each column a feature.
        y (list):
            a list of integers representing the training data labels.
        cost_vec (numpy.ndarray):
            the vector of costs represented by a numpy.ndarray with shape
            (1, X.shape[1]). If None, the cost is set to zero for each feature.
        cost_param_vec (numpy.ndarray):
            the vector of the cost penalization parameters for which the 
            corresponding rankings will be computed.
        num_features_to_select (int):
            the number of best features to select. If unspecified, does not
            select a subset of features but keep all of them.
        imp_type (str):
            a string, either "impurity" or "permutation", to use the 
            random forest importance based on the decrease of the impurity
            measure (MDI), or based on the decrease of accuracy due to random permutation
            of the covariate values (MDA). "impurity" by default.
        min_samples_leaf (int):
            the minimum number of samples required to split an internal node.
            1 by default.
        n_estimators (int):
            the number of trees in the random forest. 500 by default.

    Returns:
        dfPen_Ranking (pandas.core.frame.DataFrame):
            a pandas DataFrame that contains in each row, a penalization
            parameter value and the corresponding ranking indexes, in decreasing
            order of importance.
            
    """
    
    unpenalized_rf_importance = None
    grid_size = len(cost_param_vec)
    nCov = X.shape[1]

    # Check on num_features_to_select
    if (num_features_to_select is not None):
        num_features_to_select = min(nCov, num_features_to_select)
    else:
        num_features_to_select = nCov    
    
    matRanking = np.zeros( (grid_size, num_features_to_select) )
    
    k = 0
    for cost_param in cost_param_vec:
        rankedIdx, unpenalized_rf_importance  = pen_rf_importance(X = X, y = y, cost_vec = cost_vec, cost_param = cost_param,
                                                                  num_features_to_select = num_features_to_select, imp_type = imp_type,
                                                                  min_samples_leaf = min_samples_leaf, n_estimators = n_estimators,
                                                                  rf_importance_vec = unpenalized_rf_importance)
        matRanking[k,:] = rankedIdx
        k += 1
    
    output = {"cost_param": cost_param_vec}
    dfPen_Ranking = pd.DataFrame(output)
    
    dfRanking = pd.DataFrame(matRanking, columns=range(1, num_features_to_select+1))
    dfPen_Ranking = pd.concat([dfPen_Ranking, dfRanking], axis=1)
    
    return dfPen_Ranking


def multi_weighted_rf_importance(X, y, cost_vec, cost_param_vec, num_features_to_select = None,
                                 imp_type = "impurity", min_samples_leaf = 1, n_estimators = 500):
    """ Cost-based ranking using weighted random forest importance, with multiple penalization values.
    
    The cost-based ranking of the features are deduced using the feature
    importance of a weighted random forest, where the probability of sampling
    a covariate at a given node is proportional to 1/(cost)^cost_param.
    This function allows the use of multiple penalization parameters.
            
    Args:
        X (numpy.ndarray):
            the numerical features to use as training data, where 
            each row represents an individual, and each column a feature.
        y (list):
            a list of integers representing the training data labels.
        cost_vec (numpy.ndarray):
            the vector of costs represented by a numpy.ndarray with shape
            (1, X.shape[1]). If None, the cost is set to zero for each feature.
        cost_param (float): 
            the positive cost penalization parameter. 0 by default.
        num_features_to_select (int):
            the number of best features to select. If unspecified, does not
            select a subset of features but keep all of them.
        imp_type (str):
            a string, either "impurity" or "permutation", to use the 
            random forest importance based on the decrease of the impurity
            measure (MDI), or based on the decrease of accuracy due to random permutation
            of the covariate values (MDA). "impurity" by default.
        min_samples_leaf (int):
            the minimum number of samples required to split an internal node.
            1 by default.
        n_estimators (int):
            the number of trees in the random forests. 500 by default.
            
    Returns:
        dfPen_Ranking (pandas.core.frame.DataFrame):
            a pandas DataFrame that contains in each row, a penalization
            parameter value and the corresponding ranking indexes, in decreasing
            order of importance.
    """
    
    grid_size = len(cost_param_vec)
    nCov = X.shape[1]

    # Check on num_features_to_select
    if (num_features_to_select is not None):
        num_features_to_select = min(nCov, num_features_to_select)
    else:
        num_features_to_select = nCov
    
    matRanking = np.zeros( (grid_size, num_features_to_select) )
    
    k = 0
    for cost_param in cost_param_vec:
        rankedIdx  = weighted_rf_importance(X = X, y = y, cost_vec = cost_vec, cost_param = cost_param,
                                            num_features_to_select = num_features_to_select, imp_type = imp_type,
                                            min_samples_leaf = min_samples_leaf, n_estimators = n_estimators)
        matRanking[k,:] = rankedIdx
        k += 1
    
    output = {"cost_param": cost_param_vec}
    dfPen_Ranking = pd.DataFrame(output)
    
    dfRanking = pd.DataFrame(matRanking, columns=range(1, num_features_to_select+1))
    dfPen_Ranking = pd.concat([dfPen_Ranking, dfRanking], axis=1)
    
    return dfPen_Ranking