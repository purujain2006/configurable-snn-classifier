"""snnsearch -- hyperparameter search for spiking networks under hardware limits.

The package is layered so that the cheap parts stay cheap:

    config, planning, hardware, cost      no torch. `summary` runs anywhere.
    quantgrid, neuron, model, folding     the network and its deployed form
    quantize, train, synops               training and what it measures
    data/, encoders                       what makes it work on any dataset
    spaces, search, results, report       the search and what it writes

Entry point is `python main.py`, or `python -m snnsearch.cli`.
"""

__version__ = "0.2.0"
