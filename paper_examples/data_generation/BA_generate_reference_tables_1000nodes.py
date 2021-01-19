# -*- coding: utf-8 -*-
"""
This file generates the reference tables associated to the four 
Barabási-Albert models as presented in our paper.

Note: When num_cores = 1, the elements of the reference tables are sequentially 
generated, so we advise relying on parallel computing by using a number of CPU 
cores (num_cores) higher than one, using a cluster, or directly using the 
reference tables we employed in our paper that are available in the data folder.
"""

from cost_based_selection import data_generation
import multiprocessing

# Specify the number of cores to use
num_cores = max(1, multiprocessing.cpu_count() - 1)

# Number of simulated data per model
num_sim_model = 2500

# Specify the directory where to save the data
loc_data = ''

if num_cores == 1:
    
    # Networks made of 1000 nodes
    num_nodes = 1000
    dfModIndex_BA1000, dfSummaries_BA1000, dfIsDisc_BA1000, dfTimes_BA1000 = data_generation.BA_ref_table(num_sim_model = num_sim_model,
                                                                                                      num_nodes = num_nodes)
    
    dfModIndex_BA1000.to_csv(loc_data+'modIndex_'+str(4*num_sim_model)+'sim_'+str(num_nodes)+'nodes_BA.csv', index=None, header=True)
    dfSummaries_BA1000.to_csv(loc_data+'summaries_'+str(4*num_sim_model)+'sim_'+str(num_nodes)+'nodes_BA.csv', index=None, header=True)
    dfIsDisc_BA1000.to_csv(loc_data+'times_'+str(4*num_sim_model)+'sim_'+str(num_nodes)+'nodes_BA.csv', index=None, header=True)
    dfTimes_BA1000.to_csv(loc_data+'isDisc_'+str(4*num_sim_model)+'sim_'+str(num_nodes)+'nodes_BA.csv', index=None, header=True)

elif num_cores > 1:
    
    # Networks made of 1000 nodes
    num_nodes = 1000
    dfModIndex_BA1000, dfSummaries_BA1000, dfIsDisc_BA1000, dfTimes_BA1000 = data_generation.BA_ref_table_parallel(num_sim_model = num_sim_model,
                                                                                                               num_nodes = num_nodes,
                                                                                                               num_cores = num_cores)
    
    dfModIndex_BA1000.to_csv(loc_data+'modIndex_'+str(4*num_sim_model)+'sim_'+str(num_nodes)+'nodes_BA.csv', index=None, header=True)
    dfSummaries_BA1000.to_csv(loc_data+'summaries_'+str(4*num_sim_model)+'sim_'+str(num_nodes)+'nodes_BA.csv', index=None, header=True)
    dfIsDisc_BA1000.to_csv(loc_data+'times_'+str(4*num_sim_model)+'sim_'+str(num_nodes)+'nodes_BA.csv', index=None, header=True)
    dfTimes_BA1000.to_csv(loc_data+'isDisc_'+str(4*num_sim_model)+'sim_'+str(num_nodes)+'nodes_BA.csv', index=None, header=True)
    
    
# Note: the function BA_ref_table_parallel can be used with num_cores = 1
# however the function BA_ref_table is slightly faster in this situation.