# Copied from https://github.com/emansim/baselines-mansimov/blob/master/baselines/a2c/visualize_atari.py
# and https://github.com/emansim/baselines-mansimov/blob/master/baselines/a2c/load.py
# Thanks to the author and OpenAI team!

import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from scipy.signal import medfilt
matplotlib.rcParams.update({'font.size': 8})
import pandas

def smooth_reward_curve(x, y):
    # Halfwidth of our smoothing convolution
    halfwidth = min(31, int(np.ceil(len(x) / 30)))
    k = halfwidth
    xsmoo = x[k:-k]
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='valid') / \
        np.convolve(np.ones_like(y), np.ones(2 * k + 1), mode='valid')
    downsample = max(int(np.floor(len(xsmoo) / 1e3)), 1)
    return xsmoo[::downsample], ysmoo[::downsample]


def fix_point(x, y, interval):
    np.insert(x, 0, 0)
    np.insert(y, 0, 0)

    fx, fy = [], []
    pointer = 0

    ninterval = int(max(x) / interval + 1)

    for i in range(ninterval):
        tmpx = interval * i

        while pointer + 1 < len(x) and tmpx > x[pointer + 1]:
            pointer += 1

        if pointer + 1 < len(x):
            alpha = (y[pointer + 1] - y[pointer]) / \
                (x[pointer + 1] - x[pointer])
            tmpy = y[pointer] + alpha * (tmpx - x[pointer])
            fx.append(tmpx)
            fy.append(tmpy)

    return fx, fy


def load_data(indir, smooth, bin_size):
    datas = []
    infiles = glob.glob(os.path.join(indir, '*.monitor.csv'))

    for inf in infiles:
        with open(inf, 'r') as f:
            f.readline()
            f.readline()
            for line in f:
                tmp = line.split(',')
                t_time = float(tmp[2])
                try:
                    tmp = [t_time, int(tmp[1]), float(tmp[0])]
                    datas.append(tmp)
                except ValueError:
                    pass

    # sort by time
    datas = sorted(datas, key=lambda d_entry: d_entry[0])
    result = []
    len_eps = []
    timesteps = 0
    for i in range(len(datas)):
        result.append([timesteps, datas[i][-1]])
        len_eps.append([timesteps, datas[i][1]])
        timesteps += datas[i][1]

    if len(result) < bin_size:
        return [None, None,None]

    x, y = np.array(result)[:, 0], np.array(result)[:, 1]
    y_len= np.array(len_eps)[:, 1]

    if smooth == 1:
        x, y = smooth_reward_curve(x, y)
        _, y_len = smooth_reward_curve(x, y_len)

    if smooth == 2:
        y = medfilt(y, kernel_size=9)
        y_len = medfilt(y_len, kernel_size=9)

    x, y = fix_point(x, y, bin_size)
    _, y_len = fix_point(x, y_len, bin_size)
    print("len y",y_len)
    return x, y,y_len


color_defaults = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]

_last_idx_plot=None
def td_plot(writer,folder,smooth=1,bin_size=100):
    global _last_idx_plot#todo fix last plot idx
    tx, ty_rw,y_len= load_data(folder, smooth, bin_size)
    if tx is None:
        return
    idx= 0 if _last_idx_plot is None else _last_idx_plot

    _last_idx_plot=len(tx)
    for i,rw,len_ep in zip(tx[idx:],ty_rw[idx:],y_len[idx:]):
        writer.add_scalar('train/rw',rw,i)
        writer.add_scalar('train/len_ep',len_ep,i)

def visdom_plot(viz, win, folder, game, name, num_steps, bin_size=100, smooth=1):
    tx, ty,_ = load_data(folder, smooth, bin_size)
    if tx is None or ty is None:
        return win

    fig = plt.figure()
    plt.plot(tx, ty, label="{}".format(name))

    tick_fractions = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    ticks = tick_fractions * num_steps
    tick_names = ["{:.0e}".format(tick) for tick in ticks]
    plt.xticks(ticks, tick_names)
    plt.xlim(0, num_steps * 1.01)

    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')

    plt.title(game)
    plt.legend(loc=4)
    plt.show()
    plt.draw()

    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    plt.close(fig)

    # Show it in visdom
    image = np.transpose(image, (2, 0, 1))
    return viz.image(image, win=win)


if __name__ == "__main__":
    from visdom import Visdom
    viz = Visdom()
    visdom_plot(viz, None, '/tmp/gym/', 'BreakOut', 'a2c', bin_size=100, smooth=1)
