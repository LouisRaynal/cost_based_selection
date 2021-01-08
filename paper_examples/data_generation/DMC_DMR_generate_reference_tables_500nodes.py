# -*- coding: utf-8 -*-
"""
This file generates the reference tables associated to the DMC and DMR models as
presented in our paper.

Note: When num_cores = 1, the elements of the reference tables are sequentially 
generated, so we advise relying on parallel computing by using a number of CPU 
cores (num_cores) higher than one, using a cluster, or directly using the 
reference tables we employed in our paper that are available in the data folder.
"""

from cost_based_selection import data_generation
import multiprocessing
import networkx as nx

# Specify the number of cores to use
num_cores = max(1, multiprocessing.cpu_count() - 1)

# Number of simulated data per model
num_sim_model = 5000

# We use a seed network made of two connected nodes
seed_network = nx.Graph()
seed_network.add_edge(0,1)

# Specify the directory where to save the data
save_directory = ""

if num_cores == 1:
    
    # Networks made of 500 nodes
    num_nodes = 500
    dfModIndex_DMC_DMR500, dfSummaries_DMC_DMR500, dfIsDisc_DMC_DMR500, dfTimes_DMC_DMR500 = data_generation.DMC_DMR_ref_table(seed_network = seed_network,
                                                                                                                               num_sim_model = num_sim_model,
                                                                                                                               num_nodes = num_nodes)
    
    dfModIndex_DMC_DMR500.to_csv(save_directory+'modIndex_'+str(2*num_sim_model)+'sim_'+str(num_nodes)+'nodes_DMC_DMR.csv', index=None, header=True)
    dfSummaries_DMC_DMR500.to_csv(save_directory+'summaries_'+str(2*num_sim_model)+'sim_'+str(num_nodes)+'nodes_DMC_DMR.csv', index=None, header=True)
    dfIsDisc_DMC_DMR500.to_csv(save_directory+'times_'+str(2*num_sim_model)+'sim_'+str(num_nodes)+'nodes_DMC_DMR.csv', index=None, header=True)
    dfTimes_DMC_DMR500.to_csv(save_directory+'isDisc_'+str(2*num_sim_model)+'sim_'+str(num_nodes)+'nodes_DMC_DMR.csv', index=None, header=True)
 
elif num_cores > 1:
    
    # Networks made of 500 nodes
    num_nodes = 500
    dfModIndex_DMC_DMR500, dfSummaries_DMC_DMR500, dfIsDisc_DMC_DMR500, dfTimes_DMC_DMR500 = data_generation.DMC_DMR_ref_table_parallel(seed_network = seed_network,
                                                                                                                                        num_sim_model = num_sim_model,
                                                                                                                                        num_nodes = num_nodes,
                                                                                                                                        num_cores = num_cores)
    
    dfModIndex_DMC_DMR500.to_csv(save_directory+'modIndex_'+str(2*num_sim_model)+'sim_'+str(num_nodes)+'nodes_DMC_DMR.csv', index=None, header=True)
    dfSummaries_DMC_DMR500.to_csv(save_directory+'summaries_'+str(2*num_sim_model)+'sim_'+str(num_nodes)+'nodes_DMC_DMR.csv', index=None, header=True)
    dfIsDisc_DMC_DMR500.to_csv(save_directory+'times_'+str(2*num_sim_model)+'sim_'+str(num_nodes)+'nodes_DMC_DMR.csv', index=None, header=True)
    dfTimes_DMC_DMR500.to_csv(save_directory+'isDisc_'+str(2*num_sim_model)+'sim_'+str(num_nodes)+'nodes_DMC_DMR.csv', index=None, header=True)

    
# Note: the function DMC_DMR_ref_table_parallel can be used with num_cores = 1
# however the function DMC_DMR_ref_table is slightly faster in this situation.