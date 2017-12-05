__author__ = 'Thurston'
import numpy as np
import pomegranate as pg
import pandas as pd
from itertools import compress

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

    def __init__(self, seq, n_states, target, anomaly_window='6h'):
        """

        :param seq: pd.Series : the observation times
        :param n_states: int : how many "symbols" in the system
        :param anomaly_times: pd.Series of pd.DateTime : when anomalies happened
        :param anomaly_window: str : the time-window to track sequences at.
        """
        self.s = seq  #
        self.n_states = n_states  #
        # self.init = False
        self.model = None  # the HMM model (pomegranate)
        # self.ref = anomaly_times  #
        self.window = anomaly_window
        self.sequences = []  # init. the sequences to be parsed.
        # self.target = pd.Series(index=seq.index, data=0)
        # self.target[np.isin(seq.index, anomaly_times)] = 1
        self.target = target
        self.mask = np.logical_not(self.target.rolling(anomaly_window).apply(np.max)).tolist()
        self.mask[0] = 0

    # TODO: rolling_apply to ADD sequences into a list for later. then fit in one pass.
    # (use a rolling np.any() on the TARGET feature, that's genius. Gives anom/not)

    def rolling_store(self, s):
        """
        empty rolling function that stores each chunk into <self.sequences>
        For use later in <self.train()>
        :param s: an ndarray
        :return: np.nan
        """

        self.sequences += [s]
        return np.nan

    def train(self, X, **kwargs):
        base_kwargs = {
            'verbose': True,
            'n_jobs': 4,
            'n_components': self.n_states
        }
        base_kwargs.update(kwargs)
        self.model = pg.HiddenMarkovModel.from_samples(pg.NormalDistribution, X=X, **base_kwargs)
        return self.model

    def fit(self, **kwargs):
        print('Parsing sub-sequences...')
        r = self.s.rolling(self.window)
        r.apply(self.rolling_store)
        print(f'Parsed {len(self.sequences)} state sequences...')
        nans = np.logical_not([np.any(np.isnan(i)) for i in self.sequences])
        self.sequences = list(compress(self.sequences, nans))
        print(f'removed {sum(nans)} NaN sequences...')
        X = list(compress(self.sequences, self.mask))
        print(f'Using {sum(self.mask)} Non-anomalous sequences;\n Training HMM...')
        self.train(X, **kwargs)
        print("Done!")
        return self.model


    def like(self):
        r = self.s.rolling(self.window)
        return r.apply(self.model.log_probability)
