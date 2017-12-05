__author__ = 'Thurston'
import numpy as np
import pomegranate as pg
import pandas as pd

def fit_symbol_model(s, **kwargs):
    """
    used to discretize the system behavior in to a set of "symbols".
    Original signal bounds are determined with kmeans, and state trainsitions
    are estimated via Baum-Welch algorithm.

    :param s: list of pd.Series objects, that contains a given sensor signal
        (optionally for multiple separate observation periods)
    :param n_comp: how many symbols/states the system is assumed to have.
    :return: pg.HiddenMarkovModel(), trained on the observed sensors.
    """

    base_kwargs = {
        'verbose': True,
        'n_jobs': 4,
        'n_components': 3
    }
    base_kwargs.update(kwargs)

    model = pg.HiddenMarkovModel.from_samples(pg.NormalDistribution,
                                              X=[s], **base_kwargs)
    return model


class RollingHMM(object):

    def __init__(self, n_states, anomaly_times, anomaly_window, seq_times):
        self.n_states = n_states
        self.init = False
        self.model = None
        self.ref = anomaly_times
        self.window = anomaly_window

    # TODO: rolling_apply to ADD sequences into a list for later. then fit in one pass.
    # (use a rolling np.any() on the TARGET feature, that's genius. Gives anom/not)

    def fit(self, s):
        x=10

    def train(self, s, **kwargs):
        base_kwargs = {
            'verbose': True,
            'n_jobs': 4,
            'n_components': 3
        }
        base_kwargs.update(kwargs)

        # if self.init is False:
            # self.model = pg.HiddenMarkovModel(pg.NormalDistribution,
            #                                   X=)
