__author__ = 'Thurston'
import numpy as np
import pomegranate as pg
import pandas as pd
from itertools import compress
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from scipy import signal
import pickle


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
        # print('Parsing sub-sequences...')
        r = self.s.rolling(self.window)
        r.apply(self.rolling_store)
        # print(f'Parsed {len(self.sequences)} state sequences...')
        nans = np.logical_not([np.any(np.isnan(i)) for i in self.sequences])
        self.sequences = list(compress(self.sequences, nans))
        # print(f'removed {sum(nans)} NaN sequences...')
        X = list(compress(self.sequences, self.mask))
        # print(f'Using {sum(self.mask)} Non-anomalous sequences;\n Training HMM...')
        self.train(X, **kwargs)
        # print("Done!")
        return self.model


    def like(self):
        r = self.s.rolling(self.window)
        return r.apply(self.model.log_probability)

    def unsupervised_anom(self):
        l = self.like()
        # if np.any(l.isna()):
        #     return 'NaN'

        # p = l.rolling(self.window).min().drop_duplicates()
        # dp = p.diff()
        n_anom = int(self.target.sum())
        # # print(dp)
        # min_idx = np.argpartition(dp, n_anom)[:n_anom]
        # return dp.iloc[min_idx]
        min_idx = np.argpartition(l, n_anom)[:n_anom]
        return l.iloc[min_idx]

    def rmse(self):
        anom = self.unsupervised_anom()
        # if anom is 'NaN':
        #     return 10000.
        pred = anom.sort_index().index.values
        actual = self.target[self.target == 1].sort_index().index.values
        return np.sqrt((np.array([i.item()*1e-13 for i in (pred-actual)])**2).mean())


class TurbineAnomalyHMM(object):

    def __init__(self, data_dir, compress_window='3h'):
        self.dir = data_dir
        self.ref = pd.read_excel(os.path.join(data_dir, 'GroundTruths.xlsx'),
                            names=['file', 'date_time'])
        self.ref.date_time = pd.to_datetime(self.ref.date_time)

        if not os.path.isfile(os.path.join(data_dir, 'completeDB.h5')):
            print('reading in...')
            df = pd.concat([pd.read_csv(os.path.join(data_dir, 'data_6302.csv'),
                                        index_col=0, parse_dates=True),
                            pd.read_csv(os.path.join(data_dir, 'data_7600.csv'),
                                        index_col=0, parse_dates=True),
                            pd.read_csv(os.path.join(data_dir, 'data_7664.csv'),
                                        index_col=0, parse_dates=True)])
            print('creating h5...')
            df.to_hdf(os.path.join(data_dir, 'completeDB.h5'))
            print('done!')
        else:
            print('reading h5...')
            df = pd.read_hdf(os.path.join(data_dir, 'completeDB.h5'))
            print('done!')

        ## normalize the data
        scale = StandardScaler()
        df_n = scale.fit_transform(df.loc[:, 'T_1':'T_27'].values)
        df_n = pd.DataFrame(data=df_n, index=df.index, columns=df.loc[:, 'T_1':'T_27'].columns)

        ## Up-sample the data; remove noise
        # TODO: Swinging Door Alg.
        # compress_window = '3h'
        dfr = df_n.resample(compress_window).median()
        dfr = dfr[dfr.abs() < 3]
        self.dfr = dfr.dropna()

        ## define the anomaly times
        target = pd.Series(index=df_n.index, data=0)
        target[np.isin(df_n.index, self.ref.date_time)] = 1

        ## Up-sample the anomaly times
        target = target.resample(compress_window).max()
        self.target = target[self.dfr.index]

    def signal_plot(self, kind='raw', ax=None):
        
        if ax is None:
            f,ax = plt.subplots(figsize=(15, 15))
        if kind is 'mean':
            for i, (name, sig) in enumerate(self.dfr.iteritems()):
                ax.plot(self.dfr.mean(axis=1) - sig+0.1*i, 'k')
            ax.vlines(self.ref.date_time.values, 0, 2.6, color='r', alpha=.5)
            ax.set_title('signal deviation from mean')
            ax.set_yticks(np.arange(0,2.8, .1))
            ax.set_yticklabels(self.dfr.columns )
        else:
            for i, (name, sig) in enumerate(self.dfr.iteritems()):
                ax.plot(sig + self.dfr.max().max() * i, 'k')
            ax.vlines(self.ref.date_time.values, 0, 27 * self.dfr.max(), color='r', alpha=.5)
            ax.set_title('signal')
            ax.set_yticks(np.arange(0, 27 * self.dfr.max().max(), self.dfr.max().max()))
            ax.set_yticklabels(self.dfr.columns)
            ax.set_ylim(-.1, 27 * self.dfr.max().max() + .1)

    def objective(self, x):
        experiment = {}
        print(x)
        anomaly_window, n_states = int(x[0, 0]), int(x[0, 1])
        for sensor, signal in tqdm(self.dfr.items()):
            # print(sensor)
            experiment[sensor] = RollingHMM(signal, n_states, self.target, anomaly_window='{}h'.format(anomaly_window))
            experiment[sensor].fit(verbose=False)

        # print('HI')
        rmse = {n: i.rmse() for n, i in list(experiment.items())}
        obj = stats.hmean(np.array(list(rmse.values())))
        case_study_fname = 'o{:.1f}w{}s{}_optHMM.pkl'.format(obj, anomaly_window, n_states)
        # print('saving iteration HMM pkl...')
        with open(os.path.join(self.dir, 'results', case_study_fname), 'wb') as f:
            pickle.dump(experiment, f)
        # print('done!')
        return obj

    def optimize(self, domain, max_iter):
        import GPyOpt
        myBopt = GPyOpt.methods.BayesianOptimization(f=self.objective,  # Objective function
                                                     domain=domain,  # Box-constrains of the problem
                                                     exact_feval=False,
                                                     # X=X, Y=Y,
                                                     initial_design_type='latin',
                                                     # num_cores=4,
                                                     model_type='GP',
                                                     acquisition_type='LCB')
        myBopt.run_optimization(max_iter)
        return myBopt


if __name__ == "__main__":
    data_dir = os.path.join('.', 'data')
    study = TurbineAnomalyHMM(data_dir, compress_window='4h')
    domain = [{'name': 'anomaly_window', 'type': 'discrete', 'domain': range(8, 49)},
              {'name': 'n_states', 'type': 'discrete', 'domain': range(3, 11)}]
    myBopt = study.optimize(domain, 20)
    myBopt.plot_acquisition(filename='acq.png')
    myBopt.plot_convergence(filename='conv.png')

    print('done!')
