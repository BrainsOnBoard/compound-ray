Toy Experiment: Apis Melifera Position Estimation

This folder contains the code used to generate the data examined in the section "Example experiment: Apis mellifera visual field comparison" of the [paper](https://elifesciences.org/articles/73893). Each python script (runnable in python 3.9) contains a block comment at the top that explains its function.

The general approach to generating all relevant data is as such:
1. Generate eye views to train the neural networks on. This is done by running `compoundViewGenerator.py`, which will create and save many eye views to the data folder. Note that this can take a long time.
2. Train the neural networks (and save these neural networks and their performance metrics such as loss curves and volumes) using `position-estimator-file-based.py`.
3. Interrogate the data using `all-learning-graphs.py`, `single-learning-graph.py`, `volume-comparator.py`
`all-learning-graphs.py` can be run once data has been generated in `data-out`
