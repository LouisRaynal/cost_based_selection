These folders contains the Python scripts to perform the analyses displayed in our paper. Scripts labeled `BA_*.py` are related to the Barabási-Albert models, while scripts labeled `DMC_DMR_*.py` are related to the Duplication Mutation Complementation / Duplication with Random Mutation models. Moreover, codes are using by default the maximum number of CPU cores minus one when possible.

- The folder `data_generation` contains the scripts to generate the different reference tables.
- The folder `cost_based_selection_analysis` contains the scripts to analyze the performance of the cost-based feature selection methods.
- The folder `network_size_analysis` contains the scripts to analyze the influence of smaller networks for the feature selection process.

Notes:
The folder `..\cost_based_selection\data` contains the data that we simulated and used, and by default the different scripts use them.
We recommend launching the different scripts in parallel on a computer cluster as we did.