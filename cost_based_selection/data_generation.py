# -*- coding: utf-8 -*-
""" Functions related to the generation of the data (reference tables) employed
in our paper.

The models used are either four settings of the Barabási-Albert model,
or the Duplication Mutation Complementation / Duplication with Random Mutation models.
"""

import networkx as nx
import pandas as pd
import random
import scipy.stats as ss
from cost_based_selection import summaries
from joblib import Parallel, delayed

def BA_ref_table(num_sim_model, num_nodes):
    """ Generation of a reference table under the four Barabási-Albert (BA) models.
    
    Function to generate a reference table with num_sim_model simulated networks per
    model, where each model is the Barabási-Albert BA(n_1, n_2) model, with n_1 
    the number of nodes num_nodes and n_2 equal to 1, 2, 3 or 4.
    The summary statistics computed are the ones defined in summaries.py.
    
    Args:
        num_sim_model (int):
            the number of simulated networks per BA model.
        num_nodes (int):
            the number of nodes in a simulated network.
            
    Returns:
        dfModIndex (pandas.core.frame.DataFrame): 
            a pandas DataFrame with one column labeled ``modIndex'' containing the
            different model indexes, 1, 2, 3 or 4 corresponding to the
            parameter value n_2 used.
        dfSummaries (pandas.core.frame.DataFrame):
            a pandas DataFrame containing, for each simulated data (in rows), the
            summary statistic values (in columns).
        dfIsDisc (pandas.core.frame.DataFrame):
            a pandas DataFrame containing in each column the nature of the
            corresponding summary statistic: True for a discrete summary,
            False for a continuous one.
        dfTimes (pandas.core.frame.DataFrame):
            a pandas DataFrame containing, for each simulated data (in rows),
            the time in seconds to compute each summary statistic values
            (in columns).
            
    """
    
    resList1 = []
    resList2 = []
    resList3 = []
    resList4 = []
    
    for i in range(num_sim_model):
        simuGraph1 = nx.barabasi_albert_graph(num_nodes, 1)
        simuGraph2 = nx.barabasi_albert_graph(num_nodes, 2)
        simuGraph3 = nx.barabasi_albert_graph(num_nodes, 3)
        simuGraph4 = nx.barabasi_albert_graph(num_nodes, 4)
    
        dictSums1, dictTimes1, dictIsDisc1 = summaries.compute_summaries(simuGraph1)
        dictSums2, dictTimes2, dictIsDisc2 = summaries.compute_summaries(simuGraph2)
        dictSums3, dictTimes3, dictIsDisc3 = summaries.compute_summaries(simuGraph3)
        dictSums4, dictTimes4, dictIsDisc4 = summaries.compute_summaries(simuGraph4)
    
        resList1.append( [1, dictSums1, dictTimes1, dictIsDisc1] )
        resList2.append( [2, dictSums2, dictTimes2, dictIsDisc2] )
        resList3.append( [3, dictSums3, dictTimes3, dictIsDisc3] )
        resList4.append( [4, dictSums4, dictTimes4, dictIsDisc4] )

    modIndex = [1]*num_sim_model + [2]*num_sim_model + [3]*num_sim_model + [4]*num_sim_model

    listOfSummaries = []
    listOfIsDisc = []
    listOfTimes = []
    
    # Model 1
    for simIdx in range(num_sim_model):
        listOfSummaries += [resList1[simIdx][1]]
        listOfTimes += [resList1[simIdx][2]]
    
    # Model 2
    for simIdx in range(num_sim_model):
        listOfSummaries += [resList2[simIdx][1]]
        listOfTimes += [resList2[simIdx][2]]
    
    # Model 3
    for simIdx in range(num_sim_model):
        listOfSummaries += [resList3[simIdx][1]]
        listOfTimes += [resList3[simIdx][2]]
    
    # Model 4
    for simIdx in range(num_sim_model):
        listOfSummaries += [resList4[simIdx][1]]
        listOfTimes += [resList4[simIdx][2]]
        
        
    listOfIsDisc = [resList1[0][3]]
    
    dfModIndex = pd.DataFrame(modIndex, columns=["modIndex"])
    dfSummaries = pd.DataFrame(listOfSummaries)
    dfIsDisc = pd.DataFrame(listOfIsDisc)
    dfTimes = pd.DataFrame(listOfTimes)
    
    return dfModIndex, dfSummaries, dfIsDisc, dfTimes


def BA_ref_table_parallel(num_sim_model, num_nodes, num_cores):
    """ Parallel generation of a reference table under the four Barabási-Albert (BA) models.
    
    Function to generate a reference table with num_sim_model simulated networks per
    model, where each model is the Barabási-Albert BA(n_1, n_2) model, with n_1 
    the number of nodes num_nodes and n_2 equal to 1, 2, 3 or 4.
    The summary statistics computed are the ones defined in summaries.py.
    
    Args:
        num_sim_model (int):
            the number of simulated networks per BA model.
        num_nodes (int):
            the number of nodes in a simulated network.
        num_cores (int):
            the number of cores to use for the table generation in parallel.
            
    Returns:
        dfModIndex (pandas.core.frame.DataFrame): 
            a pandas DataFrame with one column labeled ``modIndex'' containing the
            different model indexes, 1, 2, 3 or 4 corresponding to the
            parameter value n_2 used.
        dfSummaries (pandas.core.frame.DataFrame):
            a pandas DataFrame containing, for each simulated data (in rows), the
            summary statistic values (in columns).
        dfIsDisc (pandas.core.frame.DataFrame):
            a pandas DataFrame containing in each column the nature of the
            corresponding summary statistic: True for a discrete summary,
            False for a continuous one.
        dfTimes (pandas.core.frame.DataFrame):
            a pandas DataFrame containing, for each simulated data (in rows),
            the time in seconds to compute each summary statistic values
            (in columns).
            
    """
    
    parallel_data = Parallel(n_jobs = num_cores)(delayed(_private_BA_indiv_simulation)(num_sim_model = num_sim_model, num_nodes = num_nodes, sim_index = sim) for sim in list(range(4*num_sim_model)) )

    modIndex = []
    listOfSummaries = []
    listOfIsDisc = []
    listOfTimes = []
    
    num_sim = len(parallel_data)
    
    for simIdx in range(num_sim):
        modIndex += [parallel_data[simIdx][0]]
        listOfSummaries += [parallel_data[simIdx][1]]
        listOfTimes += [parallel_data[simIdx][2]]

    listOfIsDisc = [parallel_data[0][3]]
    
    dfModIndex = pd.DataFrame(modIndex, columns=["modIndex"])
    dfSummaries = pd.DataFrame(listOfSummaries)
    dfIsDisc = pd.DataFrame(listOfIsDisc)
    dfTimes = pd.DataFrame(listOfTimes)

    return dfModIndex, dfSummaries, dfIsDisc, dfTimes


def _private_BA_indiv_simulation(num_sim_model, num_nodes, sim_index):
    """ Generation of one line of the reference table under the four Barabási-Albert (BA) models.
        
    Args:
        num_sim_model (int):
            the number of simulated networks per BA model.
        num_nodes (int):
            the number of nodes in a simulated network.
        sim_index (int):
            the simulation index, will specify according to which model to simulate.
            
    Returns:
        modIndex (int): 
            an integer representing the model index used for simulation.
        dictSums (dict):
            a dictionary with the name of the summaries as keys and the summary 
            statistic values as values.
        dictTimes (pandas.core.frame.DataFrame):
            dictionary with the name of the summaries as keys and the time to 
            compute each one as values.
        dictIsDisc (pandas.core.frame.DataFrame):
            a dictionary indicating if the summary is discrete (True) or not 
            (False).
            
    """

    if (0 <= sim_index) & (sim_index < num_sim_model):
        simuGraph = nx.barabasi_albert_graph(num_nodes, 1)
        dictSums, dictTimes, dictIsDisc = summaries.compute_summaries(simuGraph)
        return 1, dictSums, dictTimes, dictIsDisc
        
    elif (num_sim_model <= sim_index) & (sim_index < 2*num_sim_model):
        simuGraph = nx.barabasi_albert_graph(num_nodes, 2)
        dictSums, dictTimes, dictIsDisc = summaries.compute_summaries(simuGraph)
        return 2, dictSums, dictTimes, dictIsDisc
        
    elif (num_sim_model <= 2*sim_index) & (sim_index < 3*num_sim_model):
        simuGraph = nx.barabasi_albert_graph(num_nodes, 3)
        dictSums, dictTimes, dictIsDisc = summaries.compute_summaries(simuGraph)
        return 3, dictSums, dictTimes, dictIsDisc
        
    elif (num_sim_model <= 3*sim_index) & (sim_index < 4*num_sim_model):
        simuGraph = nx.barabasi_albert_graph(num_nodes, 4)
        dictSums, dictTimes, dictIsDisc = summaries.compute_summaries(simuGraph)
        return 4, dictSums, dictTimes, dictIsDisc
 

def DMC(seed_network, num_nodes, q_mod, q_con):
    """ Simulate one network according to the DMC model.
    
    Simulate one network according to the Duplication Mutation Complementation
    model (Vázquez et al. 2003).
    
    A. Vázquez, A. Flammini, A. Maritan, and A. Vespignani. 
    Modeling of protein interaction networks. Complexus, 1:38–44, 2003.
    
    Args:
        seed_network (networkx.classes.graph.Graph):
            a networkx graph which is the seed network used to simulate the
            network.
        num_nodes (int):
            the final number of nodes in the simulated network.
        q_mod (float):
            the probability (between 0 and 1) of edge removal during the 
            mutation step.
        q_con (float):
            the probability (between 0 and 1) of connecting the new node and 
            the duplicated node during the complementation step.
            
    Returns:
        sim_network (networkx.classes.graph.Graph):
            the simulated DMC network.
            
    """
    
    G = seed_network.copy()
    seed_num_nodes = G.number_of_nodes()
    for v in range(seed_num_nodes, num_nodes):
        # Select a random node u in the graph for duplication
        u = random.choice( list(G.nodes()) )
        # Add a new node to the graph 
        G.add_node(v)
        # and duplicate the relationships of u to v
        G.add_edges_from([(v,w) for w in G.neighbors(u)])
        # For each neighbors of u
        for w in list(G.neighbors(u)):
            # We generate a Bernoulli random variable with parameter q_mod
            # if it's 1 we remove at random the relationship (v,w) or (u,w)
            if ss.bernoulli.rvs(q_mod):
                edge = random.choice([(v,w), (u,w)])
                G.remove_edge(*edge) # * to unpack the tuple edge
        # Finally, with probability q_con, add an edge between 
        # the duplicated and duplicate nodes
        if ss.bernoulli.rvs(q_con):
            G.add_edge(u,v)
    return G


def DMR(seed_network, num_nodes, q_del, q_new):
    """ Simulate one network according to the DMR model.
    
    Simulate one network according to the Duplication with Random Mutation
    model (Solé et al. 2002).
    
    R. V. Solé,  R. Pastor-Satorras, E. Smith, and T. B. Kepler.
    A model of large-scale proteome evolution. Advances in Complex Systems,
    5(1):43–54, 2002. 
    
    Args:
        seed_network (networkx.classes.graph.Graph):
            a networkx graph which is the seed network used to simulate the
            network.
        num_nodes (int):
            the final number of nodes in the simulated network.
        q_del (float):
            the probability (between 0 and 1) of edge removal during the 
            mutation step.
        q_new (float):
            the probability (between 0 and 1) of connecting the new node and 
            the duplicated node is q_new divided by the initial number of nodes
            at the start of the network construction step.
            
    Returns:
        sim_network (networkx.classes.graph.Graph):
            the simulated DMC network.
            
    """
    
    G = seed_network.copy()
    seed_num_nodes = G.number_of_nodes()
    for v in range(seed_num_nodes, num_nodes):
        node_list = list(G.nodes())
        # Select a random node u in the graph for duplication
        u = random.choice(node_list)
        # Add a new node to the graph 
        G.add_node(v)
        # and duplicate the relationships of u to v
        G.add_edges_from([(v,w) for w in G.neighbors(u)])
        # For each neighbor of v
        for w in list(G.neighbors(v)):
            # with probability q_del we remove the link (v,w)
            if ss.bernoulli.rvs(q_del):
                G.remove_edge(v,w)
        # And for all other node, we establish a link with v with 
        # proba q_new/number_of_nodes_before_duplication
        nodes_to_link = random.sample(node_list, ss.binom.rvs(v, q_new/v))
        G.add_edges_from([(v,x) for x in nodes_to_link])
        
    return G


def DMC_DMR_ref_table(seed_network, num_sim_model, num_nodes):
    """ Generation of a reference table under the DMC and DMR models.
    
    Function to generate a reference table with num_sim_model simulated data per
    model, where the models are the Duplication Mutation Complementation model
    and the Duplication with Random Mutation model. The DMC model carries the
    index 1, while DMR carries the index 2. The priors on parameters are
    Uniform [0.25, 0.75] for all model parameters.
    The summary statistics computed are the ones defined in summaries.py.
    
    Args:
        seed_network (networkx.classes.graph.Graph):
            a networkx graph which is the seed network used to simulate each
            network.
        num_sim_model (int):
            the number of simulated networks per model.
        num_nodes (int):
            the number of nodes in a simulated network.
            
    Returns:
        dfModIndex (pandas.core.frame.DataFrame): 
            a pandas DataFrame with one column labeled ``modIndex'' containing the
            different model indexes, 1 for DMC, 2 for DMR.
        dfSummaries (pandas.core.frame.DataFrame):
            a pandas DataFrame containing, for each simulated data (in rows), the
            summary statistic values (in columns).
        dfIsDisc (pandas.core.frame.DataFrame):
            a pandas DataFrame containing in each column the nature of the
            corresponding summary statistic: True for a discrete summary,
            False for a continuous one.
        dfTimes (pandas.core.frame.DataFrame):
            a pandas DataFrame containing, for each simulated data (in rows),
            the time in seconds to compute each summary statistic values
            (in columns).
            
    """
    
    resList1 = []
    resList2 = []
    
    for i in range(num_sim_model):
        
        # Parameter generation
        q_mod_sim = ss.uniform.rvs(0.25, 0.5, size=1)[0]
        q_con_sim = ss.uniform.rvs(0.25, 0.5, size=1)[0]

        q_del_sim = ss.uniform.rvs(0.25, 0.5, size=1)[0]
        q_new_sim = ss.uniform.rvs(0.25, 0.5, size=1)[0]

        simuGraphDMC = DMC(seed_network = seed_network, num_nodes = num_nodes, 
                           q_mod = q_mod_sim, q_con = q_con_sim)
        simuGraphDMR = DMR(seed_network = seed_network, num_nodes = num_nodes, 
                           q_del = q_del_sim, q_new = q_new_sim)
    
        dictSums1, dictTimes1, dictIsDisc1 = summaries.compute_summaries(simuGraphDMC)
        dictSums2, dictTimes2, dictIsDisc2 = summaries.compute_summaries(simuGraphDMR)
    
        resList1.append( [1, dictSums1, dictTimes1, dictIsDisc1] )
        resList2.append( [2, dictSums2, dictTimes2, dictIsDisc2] )

    modIndex = [1]*num_sim_model + [2]*num_sim_model
    
    listOfSummaries = []
    listOfIsDisc = []
    listOfTimes = []
    
    # Model DMC (indexed 1)
    for simIdx in range(num_sim_model):
        listOfSummaries += [resList1[simIdx][1]]
        listOfTimes += [resList1[simIdx][2]]
    
    # Model DMR (indexed 2)
    for simIdx in range(num_sim_model):
        listOfSummaries += [resList2[simIdx][1]]
        listOfTimes += [resList2[simIdx][2]]
                    
    listOfIsDisc = [resList1[0][3]]
    
    dfModIndex = pd.DataFrame(modIndex, columns=["modIndex"])
    dfSummaries = pd.DataFrame(listOfSummaries)
    dfIsDisc = pd.DataFrame(listOfIsDisc)
    dfTimes = pd.DataFrame(listOfTimes)
    
    return dfModIndex, dfSummaries, dfIsDisc, dfTimes

def DMC_DMR_ref_table_parallel(seed_network, num_sim_model, num_nodes, num_cores):
    """ Generation of a reference table under the DMC and DMR models in parallel.
    
    Function to generate a reference table with num_sim_model simulated data per
    model, where the models are the Duplication Mutation Complementation model
    and the Duplication with Random Mutation model. The DMC model carries the
    index 1, while DMR carries the index 2. The priors on parameters are
    Uniform [0.25, 0.75] for all model parameters.
    The summary statistics computed are the ones defined in summaries.py.
    
    Args:
        seed_network (networkx.classes.graph.Graph):
            a networkx graph which is the seed network used to simulate each
            network.
        num_sim_model (int):
            the number of simulated networks per model.
        num_nodes (int):
            the number of nodes in a simulated network.
        num_cores (int):
            the number of cores to use for the table generation in parallel.
            
    Returns:
        dfModIndex (pandas.core.frame.DataFrame): 
            a pandas DataFrame with one column labeled ``modIndex'' containing the
            different model indexes, 1 for DMC, 2 for DMR.
        dfSummaries (pandas.core.frame.DataFrame):
            a pandas DataFrame containing, for each simulated data (in rows), the
            summary statistic values (in columns).
        dfIsDisc (pandas.core.frame.DataFrame):
            a pandas DataFrame containing in each column the nature of the
            corresponding summary statistic: True for a discrete summary,
            False for a continuous one.
        dfTimes (pandas.core.frame.DataFrame):
            a pandas DataFrame containing, for each simulated data (in rows),
            the time in seconds to compute each summary statistic values
            (in columns).
            
    """
    
    parallel_data = Parallel(n_jobs = num_cores)(delayed(_private_DMC_DMR_indiv_simulation)(seed_network = seed_network, num_sim_model = num_sim_model, num_nodes = num_nodes, sim_index = sim) for sim in list(range(2*num_sim_model)) )

    modIndex = []
    listOfSummaries = []
    listOfIsDisc = []
    listOfTimes = []
    
    num_sim = len(parallel_data)
    
    for simIdx in range(num_sim):
        modIndex += [parallel_data[simIdx][0]]
        listOfSummaries += [parallel_data[simIdx][1]]
        listOfTimes += [parallel_data[simIdx][2]]

    listOfIsDisc = [parallel_data[0][3]]
            
    dfModIndex = pd.DataFrame(modIndex, columns=["modIndex"])
    dfSummaries = pd.DataFrame(listOfSummaries)
    dfIsDisc = pd.DataFrame(listOfIsDisc)
    dfTimes = pd.DataFrame(listOfTimes)

    return dfModIndex, dfSummaries, dfIsDisc, dfTimes


def _private_DMC_DMR_indiv_simulation(seed_network, num_sim_model, num_nodes, sim_index):
    """ Generation of one line of the reference table under the DMC or DMR models.
        
    Args:
        seed_network (networkx.classes.graph.Graph):
            a networkx graph which is the seed network used to simulate each
            network.
        num_sim_model (int):
            the number of simulated networks per BA model.
        num_nodes (int):
            the number of nodes in a simulated network.
        sim_index (int):
            the simulation index, will specify according to which model to simulate.
            
    Returns:
        modIndex (int): 
            an integer representing the model index used for simulation.
        dictSums (dict):
            a dictionary with the name of the summaries as keys and the summary 
            statistic values as values.
        dictTimes (pandas.core.frame.DataFrame):
             dictionary with the name of the summaries as keys and the time to 
             compute each one as values.
        dictIsDisc (pandas.core.frame.DataFrame):
            a dictionary indicating if the summary is discrete (True) or not 
            (False).
            
    """
    
    if (0 <= sim_index) & (sim_index < num_sim_model):
        q_mod_sim = ss.uniform.rvs(0.25, 0.5, size=1)[0]
        q_con_sim = ss.uniform.rvs(0.25, 0.5, size=1)[0]
        simuGraph = DMC(seed_network = seed_network, num_nodes = num_nodes,
                        q_mod = q_mod_sim, q_con = q_con_sim)
        dictSums, dictTimes, dictIsDisc = summaries.compute_summaries(simuGraph)
        return 1, dictSums, dictTimes, dictIsDisc
        
    elif (num_sim_model <= sim_index) & (sim_index < 2*num_sim_model):
        q_del_sim = ss.uniform.rvs(0.25, 0.5, size=1)[0]
        q_new_sim = ss.uniform.rvs(0.25, 0.5, size=1)[0]
        simuGraph = DMR(seed_network = seed_network, num_nodes = num_nodes, 
                        q_del = q_del_sim, q_new = q_new_sim)
        dictSums, dictTimes, dictIsDisc = summaries.compute_summaries(simuGraph)
        return 2, dictSums, dictTimes, dictIsDisc