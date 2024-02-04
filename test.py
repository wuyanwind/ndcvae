import os
import sys

import numpy as np

import pandas as pd
import tensorflow as tf
from scipy import interpolate
import scipy.io as sio
import models
import time
from tqdm import tqdm

# dataset
class Dataset:
    """Object to hold the simulated or empirical data"""
    def __init__(self, name, t, x, y, thetareg, thetasub, w, description=None):
        self.name = name

        assert t.ndim == 1
        assert x.ndim == 4
        assert y.ndim == 4
        assert thetareg.ndim == 3
        assert thetasub.ndim == 2
        assert w.ndim == 3

        nsub, nreg, ns, nt = x.shape
        nobs = y.shape[2]

        assert y.shape[:3] == (nsub, nreg, nobs)
        assert thetareg.shape[0:2] == (nsub, nreg)
        assert thetasub.shape[0] == nsub
        assert w.shape == (nsub, nreg, nreg)

        self.t = t
        self.x = x
        self.y = y
        self.thetareg = thetareg
        self.thetasub = thetasub
        self.w = w

        self.nsub = nsub
        self.nreg = nreg
        self.nk = ns
        self.nobs = nobs

        self.description = description if description is not None else ""

    @classmethod
    def from_file(cls, filename):
        data = np.load(filename)
        return cls(name=data['name'], t=data['t'], x=data['x'], y=data['y'],
                   thetareg=data['thetareg'], thetasub=data['thetasub'],
                   w=data['w'], description=data['description'])


    def save(self, filename):
        np.savez(filename, t=self.t, x=self.x, y=self.y, thetareg=self.thetareg, thetasub=self.thetasub,
                 w=self.w, name=self.name, description=self.description)



def normalize(w):
    w = w/np.max(w)
    return w

def get_ds_nc_sz(TS,SC):
    name = '973'
    w = np.array(normalize(SC))
    y = TS
    nsub, nreg, _ = w.shape
    sampling_period = 2  # seconds
    nsub, nreg, _, nt = y.shape
    x = np.full((nsub, nreg, 1, nt), np.nan)
    t = sampling_period * np.r_[:nt]
    thetareg = np.full((nsub, nreg, 0), np.nan)
    thetasub = np.full((nsub, 0), np.nan)
    ds = Dataset(name, t, x, y, thetareg, thetasub, w)
    return ds


### training dataset
def _prep_training_dataset(ds_nc, ds_sz, mode='region-upsampled', upsample_factor=None, shuffle=1,
                           mask_nc=None, mask_sz=None):
    """
    Take dataset and prepare training dataset for a model.
    Two modes are (or are planned to be) supported:
    - 'region',  where each training sample consist of subject id and the region timeseries
    - 'subject', where each training sample consist of all time series
    """

    if mode == 'region-upsampled':
        nsamples = 200
        subj_ind_nc, yobs_nc, iext_nc, iext_upsampled_nc = get_full_dataset(ds_nc, upsample_factor, mask_nc)
        subj_ind_sz, yobs_sz, iext_sz, iext_upsampled_sz = get_full_dataset(ds_sz, upsample_factor, mask_sz)


        return subj_ind_nc, yobs_nc, iext_nc, iext_upsampled_nc, subj_ind_sz, yobs_sz, iext_sz, iext_upsampled_sz

    elif mode == 'subject':
        raise NotImplementedError("Subject-based dataset preparation not implemented")

    else:
        raise NotImplementedError(f"Dataset preparation mode {mode} not implemented")

def get_full_dataset(ds, upsample_factor=1, mask=None):
    nsub, nreg, nobs, nt = ds.y.shape
    mask = np.reshape(mask, (nsub * nreg))
    subj_ind = np.repeat(np.r_[:nsub], nreg)
    yobs = np.reshape(np.swapaxes(ds.y, 2, 3), (nsub * nreg, nt, nobs))
    iext = np.reshape(get_network_input_obs(ds.w, ds.y, comp=0)[:, :, 0, :], (nsub * nreg, nt))
    finterp = interpolate.interp1d(np.linspace(0, nt - 1, nt), iext, axis=-1,
                                   fill_value=(iext[:, 0], iext[:, -1]), bounds_error=False)
    iext_upsampled = finterp(
        np.linspace(-1. / 2. + 1. / (2 * upsample_factor), nt - 1. / 2. - 1. / (2 * upsample_factor),
                    nt * upsample_factor))
    subj_ind = np.array(subj_ind, dtype=np.int32)[mask]
    yobs = np.array(yobs, dtype=np.float32)[mask]
    iext = np.array(iext, dtype=np.float32)[mask]
    iext_upsampled = np.array(iext_upsampled, dtype=np.float32)[mask]

    return subj_ind, yobs, iext, iext_upsampled

def get_network_input_obs(w, y, comp=0):
    """
    Calculate the network input, using the comp component of the observations
    """

    nsub, nreg, nobs, nt = y.shape
    assert w.shape == (nsub, nreg, nreg)

    yinp = np.zeros((nsub, nreg, 1, nt))
    for i in range(nsub):
        for j in range(nreg):
            yinp[i, j, 0, :] = np.dot(w[i,j,:], y[i, :, comp, :])
    return yinp


def train(model, nc_subj_ind, nc_yobs, nc_u, nc_u_upsampled, sz_subj_ind, sz_yobs, sz_u, sz_u_upsampled,
          nc_subj_ind_test, nc_yobs_test, nc_u_test, nc_u_upsampled_test, sz_subj_ind_test, sz_yobs_test, sz_u_test, sz_u_upsampled_test,
          runner, fh=None, callback=None):

    hist = History(model)
    if fh:
        hist.print_header(fh)
    train_step = get_train_step_fn()

    for i in tqdm(range(1, runner.nbatches)):
        betax = 1
        betap = 1

        nc_index = np.random.randint(low=0, high=nc_subj_ind.shape[0], size=batch_size)
        sz_index = np.random.randint(low=0, high=sz_subj_ind.shape[0], size=batch_size)

        nc_subj_ind_train = nc_subj_ind[nc_index]
        nc_yobs_train     = nc_yobs[nc_index, :, :]
        nc_u_train = nc_u[nc_index, :]
        nc_u_upsampled_train = nc_u_upsampled[nc_index, :]
        sz_subj_ind_train = sz_subj_ind[sz_index]
        sz_yobs_train     = sz_yobs[sz_index, :, :]
        sz_u_train = sz_u[sz_index, :]
        sz_u_upsampled_train = sz_u_upsampled[sz_index, :]


        loss = tf.keras.metrics.Mean()

        loss(train_step(model, nc_subj_ind_train, nc_yobs_train, nc_u_train, nc_u_upsampled_train,
                        sz_subj_ind_train, sz_yobs_train, sz_u_train, sz_u_upsampled_train,
                        runner, betax, betap))

        if (i % 500 == 0):
            nc_index_test = np.random.randint(low=0, high=nc_subj_ind_test.shape[0], size=batch_size)
            sz_index_test = np.random.randint(low=0, high=sz_subj_ind_test.shape[0], size=batch_size)

            nc_subj_ind_train_test = nc_subj_ind_test[nc_index_test]
            nc_yobs_train_test = nc_yobs_test[nc_index_test, :, :]
            nc_u_train_test = nc_u_test[nc_index_test, :]
            nc_u_upsampled_train_test = nc_u_upsampled_test[nc_index_test, :]
            sz_subj_ind_train_test = sz_subj_ind_test[sz_index_test]
            sz_yobs_train_test = sz_yobs_test[sz_index_test, :, :]
            sz_u_train_test = sz_u_test[sz_index_test, :]
            sz_u_upsampled_train_test = sz_u_upsampled_test[sz_index_test, :]
            loss_test = tf.keras.metrics.Mean()
            loss_test(model.loss(nc_subj_ind_train_test, nc_yobs_train_test, nc_u_train_test, nc_u_upsampled_train_test,
                                 sz_subj_ind_train_test, sz_yobs_train_test, sz_u_train_test, sz_u_upsampled_train_test,
                                 nsamples=runner.nsamples, betax=betax, betap=betap))

            hist.add(i, loss.result().numpy(), betax, betap, model, loss_test.result().numpy())
            if fh:
                hist.print_last(fh)
            if callback:
                state = State(i, loss.result().numpy(), betax, betap)
                callback(state, model)
            tf.keras.backend.clear_session()

    return hist

def get_train_step_fn():
    @tf.function
    def train_step(model, nc_subj_ind, nc_yobs, nc_u, nc_u_upsampled, sz_subj_ind, sz_yobs, sz_u, sz_u_upsampled,
                   runner, betax, betap):
        with tf.GradientTape() as tape:
            loss = model.loss(nc_subj_ind, nc_yobs, nc_u, nc_u_upsampled, sz_subj_ind, sz_yobs, sz_u, sz_u_upsampled,
                              nsamples=runner.nsamples, betax=betax, betap=betap)

        gradients = tape.gradient(loss, model.trainable_variables)

        # Remove empty gradients
        trainable_variables = [v for (v,g) in zip(model.trainable_variables, gradients) if g is not None]
        gradients = [g for g in gradients if g is not None]

        if runner.clip_gradients is None:
            runner.optimizer.apply_gradients(zip(gradients, trainable_variables))
        else:
            gradients_clipped = [tf.clip_by_value(grad,
                                                  runner.clip_gradients[0],
                                                  runner.clip_gradients[1])
                                 for grad in gradients]
            runner.optimizer.apply_gradients(zip(gradients_clipped, trainable_variables))

        return loss

    return train_step


class History:
    def __init__(self, model):
        model_names, model_fmts = model.tracked_variables()
        self.names = ["nbatch", "loss", "loss_test", "betax", "betap"] + model_names
        self.fmts = ["%6d", "%14.2f", "%14.2f", "%6.3f", "%6.3f"] + model_fmts
        self._hist = []

    def print_header(self, fh):
        fh.write(" ".join(self.names) + "\n")
        fh.flush()

    def add(self, nbatch, loss, betax, betap, model, loss_test=0.):
        model_values = model.tracked_variables_values()
        self._hist.append([nbatch, loss, loss_test, betax, betap] + model_values)

    def print_last(self, fh):
        line = " ".join(self.fmts) % tuple(self._hist[-1])
        fh.write(line + "\n")
        fh.flush()

    def as_dataframe(self):
        df = pd.DataFrame(self._hist, columns=self.names)
        return df

class State:
    def __init__(self, nbatch, loss, betax, betap):
        self.nbatch = nbatch
        self.loss = loss
        self.betax = betax
        self.betap = betap


def linclip(i1, i2, v1, v2):
    """
    Return function to calculate beta for clipped linear growth
    """

    def beta(i):
        a = (v2 - v1)/(i2 - i1)
        b = (i2*v1 - i1*v2)/(i2 - i1)
        return np.clip(a*i+b, min(v1, v2), max(v1, v2))

    return beta

def irregsaw(lst, default=1.0):
    def beta(i):
        for (ifr, ito, valfr, valto) in lst:
            if (i >= ifr) and (i <= ito):
                a = (valto - valfr)/(ito - ifr)
                b = (ito*valfr - ifr*valto)/(ito - ifr)
                return a*i + b
        return default

    return beta


class Runner:
    """
    Class to hold the info about the optimization process
    """

    def __init__(self, batch_size, optimizer, nbatches, clip_gradients=None, nsamples=8, betax=1.0, betap=1.0):
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.nbatches = nbatches
        self.clip_gradients = clip_gradients
        self.nsamples = nsamples
        self.betax = betax if callable(betax) else (lambda e: betax)
        self.betap = betap if callable(betap) else (lambda e: betap)

if __name__ == "__main__":
    ds_all = sio.loadmat(r'ds_SC100.mat')
    ts_nc = ds_all['ts_nc']
    ts_sz = ds_all['ts_sz']
    SC_nc = ds_all['SC_nc']
    SC_sz = ds_all['SC_sz']
    upsample_factor = 1
    train_ratio = 0.8
    batch_size = 64
    output_dir = ''

    ds_nc = get_ds_nc_sz(ts_nc, SC_nc)
    ds_sz = get_ds_nc_sz(ts_sz, SC_sz)

    ndata_nc = int(train_ratio * ds_nc.nreg * ds_nc.nsub)
    ndata_sz = int(train_ratio * ds_sz.nreg * ds_sz.nsub)
    ipe = int(np.ceil(ndata_nc / batch_size))
    train_mask_nc = np.zeros((ds_nc.nsub, ds_nc.nreg), dtype=bool)
    train_mask_nc[np.unravel_index(np.random.choice(ds_nc.nsub * ds_nc.nreg, ndata_nc, replace=False),
                                (ds_nc.nsub, ds_nc.nreg))] = True
    train_mask_sz = np.zeros((ds_sz.nsub, ds_sz.nreg), dtype=bool)
    train_mask_sz[np.unravel_index(np.random.choice(ds_sz.nsub * ds_sz.nreg, ndata_sz, replace=False),
                                (ds_sz.nsub, ds_sz.nreg))] = True
    np.save(os.path.join(output_dir, "train_mask_nc.npy"), train_mask_nc)
    np.save(os.path.join(output_dir, "train_mask_sz.npy"), train_mask_sz)

    nc_subj_ind, nc_yobs, nc_u, nc_u_upsampled = get_full_dataset(ds_nc, upsample_factor, train_mask_nc)
    sz_subj_ind, sz_yobs, sz_u, sz_u_upsampled = get_full_dataset(ds_sz, upsample_factor, train_mask_sz)
    sz_subj_ind = sz_subj_ind + ds_nc.nsub
    nc_subj_ind_test, nc_yobs_test, nc_u_test, nc_u_upsampled_test = get_full_dataset(ds_nc, upsample_factor, ~train_mask_nc)
    sz_subj_ind_test, sz_yobs_test, sz_u_test, sz_u_upsampled_test = get_full_dataset(ds_sz, upsample_factor, ~train_mask_sz)
    sz_subj_ind_test = sz_subj_ind_test + ds_nc.nsub

    model = models.RegX(ns=2, mreg=2, msub=2, nreg=100, nsub=509+546, nt=225, nobs=1, prediction='normal',
                               shared_input=False)
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay([300000 * ipe, 600000 * ipe], [1e-3, 3e-4, 1e-4])
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    runner = Runner(optimizer=optimizer, batch_size=batch_size, nbatches=50000, nsamples=8, clip_gradients=(-1000, 1000),
                                  betax=1., betap=linclip(0, 500, 0., 1.))

    # Create callback function
    if not os.path.exists(os.path.join(output_dir, "img")):
        os.makedirs(os.path.join(output_dir, "img"))
    if not os.path.exists(os.path.join(output_dir, "models")):
        os.makedirs(os.path.join(output_dir, "models"))

    def callback(state, model):
        if (state.nbatch % 500 != 0):
            return
        model.save_weights(os.path.join(output_dir, f"models/model_{state.nbatch:05d}"))

    # Run the training
    start_time = time.time()
    hist = train(model, nc_subj_ind, nc_yobs, nc_u, nc_u_upsampled, sz_subj_ind, sz_yobs, sz_u, sz_u_upsampled,
                 nc_subj_ind_test, nc_yobs_test, nc_u_test, nc_u_upsampled_test, sz_subj_ind_test, sz_yobs_test, sz_u_test,
                 sz_u_upsampled_test, runner, fh=sys.stdout, callback=callback)

    dfh = hist.as_dataframe()

    # Save the history
    dfh.to_csv(os.path.join(output_dir, "hist.csv"), index=False)
    # Save the model
    model.save_weights(os.path.join(output_dir, "model"))
    end_time = time.time()
    print("程序运行时间：%.2f秒" % (end_time - start_time))




