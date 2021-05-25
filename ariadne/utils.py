import torch
import numpy as np
import itertools
import pandas as pd
from copy import deepcopy
from ariadne.tracknet_v2.metrics import point_in_ellipse
import faiss
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
import glob
import os
import random
import logging

def brute_force_hits_two_first_stations(df, return_momentum=False):
    first_station = df[df['station'] == 0][['r', 'phi', 'z','px', 'py', 'pz', 'track']]
    first_station.columns = ['r_left', 'phi_left', 'z_left', 'px_left', 'py_left', 'pz_left', 'track_left']
    second_station = df[df['station'] == 1][['r', 'phi', 'z', 'px', 'py', 'pz', 'track']]
    second_station.columns = ['r_right', 'phi_right', 'z_right',  'px_right', 'py_right', 'pz_right', 'track_right']
    temp_y = df[df['station'] > 1][['phi', 'z', 'track']]
    rows = itertools.product(first_station.iterrows(), second_station.iterrows())
    df = pd.DataFrame(left.append(right) for (_, left), (_, right) in rows)
    df = df.sample(frac=1).reset_index(drop=True)
    df_label = (df['track_left'] == df['track_right']) & (df['track_left'] != -1)

    x = df[['r_left', 'phi_left', 'z_left', 'r_right', 'phi_right', 'z_right']].values.reshape((-1, 2, 3))
    y = np.full((len(df_label), 2), -2.)
    if return_momentum:
        df_all_fake = (df['track_left'] == df['track_right']) & (df['track_left'] == -1)
        df_first_fake = (df['track_left'] == -1) & (df['track_right'] != -1)
        temp_momentum = df[['px_left', 'py_left', 'pz_left']]
        temp_momentum.loc[df_first_fake, ['px_left', 'py_left', 'pz_left']] = 0.
        temp_momentum.loc[df_all_fake, ['px_left', 'py_left', 'pz_left']] = df[['px_right', 'py_right', 'pz_right']]
    y[df_label == 1] = np.squeeze(np.array(list(map(lambda x: temp_y[temp_y['track'] == x][['phi', 'z']].values,
                                              df[df_label == 1]['track_left']))), 1)
    if not return_momentum:
        return x, y, df_label
    momentum = np.full((len(df_label), 3), -2.)
    momentum[df_label == 1] = np.squeeze(
            np.array(list(map(lambda x: temp_momentum[temp_momentum['track'] == x][['px', 'py', 'pz']].values,
                              df[df_label == 1]['track_left']))), 1)
    return x, y, momentum, df_label

def brute_force_hits_two_first_stations_cart(df, return_momentum=False):
    first_station = df[df['station'] == 0][['z', 'x', 'y','px', 'py', 'pz', 'track']]
    first_station.columns = ['z_left', 'x_left', 'y_left', 'px_left', 'py_left', 'pz_left', 'track_left']
    second_station = df[df['station'] == 1][['z', 'x', 'y', 'px', 'py', 'pz', 'track']]
    second_station.columns = ['z_right', 'x_right', 'y_right',  'px_right', 'py_right', 'pz_right', 'track_right']
    temp_y = df[df['station'] == 2][['x', 'y', 'track']]
    rows = itertools.product(first_station.iterrows(), second_station.iterrows())
    df = pd.DataFrame(left.append(right) for (_, left), (_, right) in rows)
    df = df.sample(frac=1).reset_index(drop=True)
    df_label = (df['track_left'] == df['track_right']) & (df['track_left'] != -1)

    x = df[['z_left', 'x_left', 'y_left', 'z_right', 'x_right', 'y_right']].values.reshape((-1, 2, 3))
    y = np.full((len(df_label), 2), -2.)
    if return_momentum:
        df_all_fake = (df['track_left'] == df['track_right']) & (df['track_left'] == -1)
        df_first_fake = (df['track_left'] == -1) & (df['track_right'] != -1)
        temp_momentum = df[['px_left', 'py_left', 'pz_left']]
        temp_momentum.loc[df_first_fake, ['px_left', 'py_left', 'pz_left']] = 0.
        temp_momentum.loc[df_all_fake, ['px_left', 'py_left', 'pz_left']] = df[['px_right', 'py_right', 'pz_right']]
    y[df_label == 1] = np.squeeze(np.array(list(map(lambda x: temp_y[temp_y['track'] == x][['x', 'y']].values,
                                              df[df_label == 1]['track_left']))), 1)
    if not return_momentum:
        return x, y, df_label
    momentum = np.full((len(df_label), 3), -2.)
    momentum[df_label == 1] = np.squeeze(
            np.array(list(map(lambda x: temp_momentum[temp_momentum['track'] == x][['px', 'py', 'pz']].values,
                              df[df_label == 1]['track_left']))), 1)
    return x, y, momentum, df_label

def weights_update(model, checkpoint):
    model_dict = model.state_dict()
    pretrained_dict = checkpoint['state_dict']
    real_dict = {}
    for (k, v) in model_dict.items():
        needed_key = None
        for pretr_key in pretrained_dict:
            if k in pretr_key:
                needed_key = pretr_key
                break
        assert needed_key is not None, "key %s not in pretrained_dict %r!" % (k, pretrained_dict.keys())
        real_dict[k] = pretrained_dict[needed_key]

    model.load_state_dict(real_dict)
    model.eval()
    return model

def load_data(input_dir, file_mask, n_samples=None):
    '''Function to load input data for tracknet-like models training.
    Helps load prepared input of tracknet or classiier (like inputs,
    input_lengths, labels etc) or last station hits stored somewhere
    as npz-file (with info about positions and events).

    Arguments:
        input_dir (string): directory with files
        file_mask (string): mask to be applied to (f.e. *.npz)
        n_samples (int, None by default): Maximum number of samples to be loaded
    '''
    flag = 0
    files = []
    data_merged = {}
    datafiles = glob.glob(f'{input_dir}/{file_mask}')
    for f in datafiles:
        one_file_data = np.load(f, mmap_mode='r', allow_pickle=False)
        first_file_len = len(one_file_data[one_file_data.files[0]])
        files = one_file_data.files
        lengths = [len(one_file_data[f]) == first_file_len for f in files]
        assert all(lengths), 'Lengths of files in npz are not equal!'
        if flag == 0:
            data_merged = dict(one_file_data.items())
            flag = 1
        else:
            for k, i in one_file_data.items():
                data_merged[k] = np.concatenate((data_merged[k], i), 0)
    if n_samples is None:
        n_samples = len(data_merged[files[0]])
    else:
        assert isinstance(n_samples, int), 'n_samples must be int or None'
        n_samples = min(n_samples, len(data_merged[files[0]]))
    return {key: item[:n_samples] for key, item in data_merged.items()}


def get_checkpoint_path(model_dir, version=None, checkpoint='latest'):
    '''Function to get checkpoint of model in given directory.
    If it is needed to use not specific, but newest version of model, it can bee found automatically
    Arguments:
        model_dir (str): directory with model checkpoints
        version (str, None by default): name of directory with needed checkpoint.
                                           If 'latest', directory with maximum change time will be used
                                           If None, only model_dir and checkpoint will be used
        checkpoint (str, 'latest' by default): name of checkpoint (with .ckpt).
                                           If 'latest', checkpoint with maximum change time will be used
    '''

    if version is not None:
        if version == 'latest':
            list_of_files = glob.glob(f"{model_dir}/*")
            version = max(list_of_files, key=os.path.getmtime).split('/')[-1]
        model_dir = model_dir+'/'+version
    if checkpoint == 'latest':
        list_of_files = glob.glob(f"{model_dir}/*.ckpt")
        checkpoint = max(list_of_files, key=os.path.getmtime).split('/')[-1]
    return f'{model_dir}/{checkpoint}'

def find_nearest_hit_no_faiss(ellipses, last_station_hits, return_numpy=False):
    centers = ellipses[:, :2]
    dists = torch.cdist(last_station_hits.float(), centers.float())
    minimal = last_station_hits[torch.argmin(dists, dim=0)]
    is_in_ellipse = point_in_ellipse(ellipses, minimal)
    if return_numpy:
        minimal = minimal.detach().cpu().numpy()
        is_in_ellipse = is_in_ellipse.detach().cpu().numpy()
    return minimal, is_in_ellipse

def find_nearest_hit(ellipses, last_station_hits, index=None, find_n=10):
    """ Function to end-to-end find nearest_hits on one plane (2-DIM)! Used for BES now (maybe outdated)"""
    #numpy, numpy -> numpy, numpy
    if index is None:
        index = faiss.IndexFlatL2(2)
        index.train(last_station_hits.astype('float32'))
        index.add(last_station_hits.astype('float32'))
    #ellipses = torch_ellipses.detach().cpu().numpy()
    centers = ellipses[:, :2]
    find_n = min(find_n, len(last_station_hits))
    _, i = index.search(np.ascontiguousarray(centers.astype('float32')), find_n)
    found_hits = last_station_hits[i.flatten()]
    found_hits = found_hits.reshape(-1, find_n, found_hits.shape[-1])
    ellipses = np.expand_dims(ellipses, 2)
    x_part = abs(found_hits[:, :, 0] - ellipses[:, 0].repeat(find_n, 1)) / ellipses[:, 2].repeat(find_n, 1)**2
    y_part = abs(found_hits[:, :, 1] - ellipses[:, 1].repeat(find_n, 1)) / ellipses[:, 3].repeat(find_n, 1)**2
    left_side = x_part + y_part
    is_in_ellipse = left_side <= 1
    return found_hits, is_in_ellipse

def store_in_index(event_hits, index=None, num_components=3):
    """Function to create faiss.IndexFlatL2 or add hits to it.
    Args:
        event_hits (np.ndarray of shape (N,num_components)): array with N hits to store in faiss index of event
        index ( faiss.IndexFlatL2 index or None): if None, new index is created, else hits are added to it
        num_components (int, 3 by default): number of dimentions in event space
    """
    #numpy -> numpy
    if index is None:
        index = faiss.IndexFlatL2(num_components)
        index.train(event_hits.astype('float32'))
    index.add(event_hits.astype('float32'))
    return index

def search_in_index(centers, index, find_n=100):
    """Function to search in index for nearest hits in 3d-space
    Args:
        centers (np.array of shape (*,3): centers for which nearest hits are needed
        index (faiss.IndexFlatL2 or other): some index with stored information about hits
        find_n (int, 100 by default): n_hits to find. If less hits were found, redundant positions contain -1
    Returns:
        numpy.ndarray with hits positions in original array
    """
    assert centers.shape[1] == 3, 'index is 3-dimentional, please add z-coordinate to centers'
    _, i = index.search(np.ascontiguousarray(centers.astype('float32')), find_n)
    return i

def filter_hits_in_ellipse(ellipse, nearest_hits, z_last=True, filter_station=True):
    """Function to get hits, which are in given ellipse.
    Space is 3-dimentional, so either first component of ellipce must be z-coordinate or third.
    Ellipse semiaxises must include x- and y-axis.
    Arguments:
        ellipse (np.array of size 5): predicted index with z-component like
                                      (x,y,z, x-semiaxis, y_semiaxis) or (z, x,y, x-semiaxis, y_semiaxis)
        nearest_hits (np.array of shape (n_hits, 3) or only 3): some hits in 3-dim space
        z_last (bool): If True, first component of vector is interpreted as z, if other, third.
        filter_station (bool): if True, only hits with same z-coordinate are considered, else all hits
    Returns:
        numpy.ndarry with filtered hits, all of them in given ellipse, sorted by increasing of distance
    """
    assert nearest_hits.shape[1] == 3, "index is 3-dimentional, please add z-coordinate to centers"
    if nearest_hits.ndim < 2:
        nearest_hits = np.expand_dims(nearest_hits, 0)
    assert ellipse.shape[-1] == 5, "index is 3-dimentional, you need to provide z-coordinate (z_c, x_c, y_c, x_r, y_r) or (x_c, y_c, z_c, x_r, y_r)"
    ellipses = np.expand_dims(ellipse, -1)
    find_n = len(nearest_hits)
    is_this_station = np.ones(len(nearest_hits))
    if z_last:
        x_part = abs(nearest_hits[:, 0] - ellipses[0].repeat(find_n)) / ellipses[-2].repeat(find_n) ** 2
        y_part = abs(nearest_hits[:, 1] - ellipses[1].repeat(find_n)) / ellipses[-1].repeat(find_n) ** 2
    else:
        x_part = abs(nearest_hits[:, 1] - ellipses[1].repeat(find_n)) / ellipses[-2].repeat(find_n) ** 2
        y_part = abs(nearest_hits[:, 2] - ellipses[2].repeat(find_n)) / ellipses[-1].repeat(find_n) ** 2
    left_side = x_part + y_part
    is_in_ellipse = left_side <= 1
    return nearest_hits[is_in_ellipse, :], is_in_ellipse

def get_extreme_points(xcenter,
                       ycenter,
                       width,
                       height):
    # width dependent measurements
    half_width = width / 2
    left = xcenter - half_width
    right = xcenter + half_width
    # height dependent measurements
    half_height = height / 2
    bottom = ycenter - half_height
    top = ycenter + half_height
    return (left, right, bottom, top)

def is_ellipse_intersects_module(ellipse_params,
                                 module_params):
    # calculate top, bottom, left, right
    ellipse_bounds = get_extreme_points(*ellipse_params)
    module_bounds = get_extreme_points(*module_params)
    # check conditions
    if ellipse_bounds[0] > module_bounds[1]:
        # ellipse left > rectangle rigth
        return False

    if ellipse_bounds[1] < module_bounds[0]:
        # ellipse rigth < rectangle left
        return False

    if ellipse_bounds[2] > module_bounds[3]:
        # ellipse bottom > rectangle top
        return False

    if ellipse_bounds[3] < module_bounds[2]:
        # ellipse top < rectangle bottom
        return False

    return True

def is_ellipse_intersects_station(ellipse_params,
                                  station_params):
    module_intersections = []
    for module in station_params:
        ellipse_in_module = is_ellipse_intersects_module(
            ellipse_params, module)
        # add to the list
        module_intersections.append(ellipse_in_module)
    return any(module_intersections)

def find_nearest_hit_old(ellipses, last_station_hits):
    #numpy, numpy -> numpy, numpy
    index = faiss.IndexFlatL2(2)
    index.train(last_station_hits.astype('float32'))
    index.add(last_station_hits.astype('float32'))

    #ellipses = torch_ellipses.detach().cpu().numpy()
    centers = ellipses[:, :2]
    _, i = index.search(np.ascontiguousarray(centers.astype('float32')), 1)
    found_hits = last_station_hits[i.flatten()]
    x_part = abs(found_hits[:, 0] - ellipses[:, 0]).flatten() / ellipses[:, 2].flatten()**2
    y_part = abs(found_hits[:, 1] - ellipses[:, 1]).flatten() / ellipses[:, 3].flatten()**2
    left_side = x_part + y_part
    is_in_ellipse = left_side <= 1
    return last_station_hits[i.flatten()], is_in_ellipse

def get_diagram_arr_linspace(all_real_hits, found_hits, start, end, num, col):
    spac = np.linspace(start, end, num=num)  # точки по х для диаграммы

    arr = []

    for i in range(len(spac) - 1): # для каждой точки
        beg = spac[i]  # начальная точка отрезка
        end = spac[i + 1]  # конечная точка отрезка
        elems_real = all_real_hits[(all_real_hits[col] > beg) & (all_real_hits[col] < end)] # все настоящие точки, попавшие в отрезок
        elems_pred = found_hits[(found_hits[col] > beg) & (found_hits[col] < end)] # все найденные точки, попавшие в отрезок
        if elems_real.empty: # есл иничего не попало
            arr.append(np.NaN)
            continue
        arr.append(len(elems_pred) / len(elems_real))  # добавляем пропорцию

    return arr, spac[:-1]  # значения и точки, где будет рисоваться линия

def draw_for_col(tracks_real,
                 tracks_pred_true,
                 col,
                 col_pretty,
                 total_events,
                 n_ticks=150,
                 n_boxes=10,
                 model_name='',
                 metric='efficiency',
                 style='boxplot',
                 verbose=True,
                 logger='ariadne.test'):
    if verbose:
        LOGGER = logging.getLogger(logger)
        LOGGER.info(f'Plotting results for metric {metric} vs {col} ==>')
    start = tracks_real[tracks_real[col] > -np.inf][col].min()  # берем минимальную точку по х
    end = tracks_real[tracks_real[col] < np.inf][col].max()  # берем максимальную точку по х

    initial_values, initial_ticks = get_diagram_arr_linspace(tracks_real, tracks_pred_true, start, end, n_ticks, col)  # значения и точки (х,у)
    # mean line
    # find number of ticks until no nans present

    if style == 'boxplot':
        interval_size = len(initial_values) / n_boxes
        subarrays = {}
        positions = {}
        means = {}
        stds = {}
        first = 0
        #get_boxes
        for interval in range(n_boxes):
            second = int(first + interval_size)
            values_in_interval = initial_values[first:min(second, len(initial_values))]
            pos_in_interval = initial_ticks[first:min(second, len(initial_values))]
            means[interval] = np.mean(values_in_interval)
            stds[interval] = np.std(values_in_interval)
            positions[interval] = np.mean(pos_in_interval)
            first = second
        draw_from_data(title=f'{model_name} track {metric} vs {col_pretty} ({total_events} events)',
                       data_x=list(positions.values()),
                       data_y=list(means.values()),
                       data_y_err=list(stds.values()),
                       axis_x=col_pretty)
    elif style=='plot':
            interp_values = np.array([np.nan])
            ticks_count = n_ticks // 5
            while np.isnan(interp_values).any(): # пока во всех бинах не будет хоть один реальный пример, ищем бины все больше и больше
                interp_values, interp_ticks = get_diagram_arr_linspace(tracks_real, tracks_pred_true, start, end, ticks_count, col)  # более грубые оценки ........ -> .... -> ..,
                ticks_count = ticks_count - ticks_count // 2
            new_ticks = np.linspace(interp_ticks.min(), interp_ticks.max(), ticks_count) #сохраняем х для результата
            try:  # делаем интерполяцию - получаем боее плавное представление
                spl = make_interp_spline(interp_ticks, interp_values, k=3)  # type: BSpline
                power_smooth = spl(new_ticks)
            except:
                spl = make_interp_spline(interp_ticks, interp_values, k=0)  # type: BSpline
                power_smooth = spl(new_ticks)
            maxX = end
            plt.figure(figsize=(8, 7))
            plt.subplot(111)
            plt.ylabel(f'Track {metric}', fontsize=12)
            plt.xlabel(col_pretty, fontsize=12)
            # plt.axis([0, maxX, 0, 1.005])
            plt.plot(initial_ticks, initial_values, alpha=0.8, lw=0.8) # оригинальные значения
            plt.title(f'{model_name} track {metric} vs {col_pretty} ({total_events} events)', fontsize=14)
            plt.plot(new_ticks, power_smooth, ls='--', label='mean', lw=2.5)  # после интерполяции
            plt.xticks(np.linspace(start, maxX, 8))
            plt.yticks(np.linspace(0, 1, 9))
            plt.legend(loc=0)
            plt.grid()
            plt.tight_layout()
            plt.rcParams['savefig.facecolor'] = 'white'
            os.makedirs('../output', exist_ok=True)
            plt.savefig(f'../output/{model_name}_img_track_{metric}_{col}_ev{total_events}_t{n_ticks}.png', dpi=300)
            plt.show()
    else:
        raise NotImplementedError(f"Style of plotting '{style}' is not supported yet")
    if verbose:
        LOGGER = logging.getLogger(logger)
        LOGGER.info(f'Plotted successful')

def draw_treshold_plots(title,
                        data_x,
                        data_y_dict,
                        axis_x=None,
                        axis_y=None,
                        model_name='tracknet',
                        total_events=100,
                        **kwargs):
    plt.figure(figsize=(8, 7))
    ax = plt.subplot(111)

    plt.title(title, fontsize=14)
    plt.ylabel(f'Value of metrics', fontsize=12)
    plt.xlabel('Treshold value for classifier output', fontsize=12)
    for metric_name, data_y in data_y_dict.items():
        plt.plot(data_x, data_y, alpha=0.8, lw=1.3, label=metric_name)  # оригинальные значения
    plt.xticks(np.linspace(data_x.min(), data_x.max(), 10))
    plt.yticks(np.linspace(0, 1, 9))
    plt.legend(loc=0)
    plt.grid()
    plt.tight_layout()
    plt.rcParams['savefig.facecolor'] = 'white'
    os.makedirs('../output', exist_ok=True)
    plt.savefig(f'../output/{model_name}_metrics_vs_treshold_ev{total_events}_n_tr{len(data_x)}.png', dpi=300)
    plt.show()

def boxplot_style_data(bp):
    for box in bp['boxes']:
        # change outline color
        # box.set( color='#7570b3', linewidth=2)
        # change fill color
        box.set(facecolor='silver')

    ## change color and linewidth of the whiskers
    # for whisker in bp['whiskers']:
    #    whisker.set(color='#7570b3', linewidth=2)
    #
    ### change color and linewidth of the caps
    # for cap in bp['caps']:
    #    cap.set(color='#7570b3', linewidth=2)
    #
    ### change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='tab:cyan', linewidth=3, alpha=0)

    for median in bp['means']:
        median.set(color='tab:green', linewidth=4, ls='-', zorder=5)
    #
    ### change the style of fliers and their fill
    # for flier in bp['fliers']:
    #    flier.set(marker='o', color='#e7298a', alpha=0.5)


def draw_from_data(title,
                   data_x,
                   data_y,
                   data_y_err,
                   axis_x=None,
                   axis_y=None,
                   model_name='tracknet',
                   **kwargs):
    data_x = np.array(data_x)
    data_y_init = np.array(data_y)
    dataep = data_y + np.array(data_y_err)
    dataem = data_y - np.array(data_y_err)

    data_y = np.expand_dims(data_y, axis=-1)
    dataep = np.expand_dims(dataep, axis=-1)
    dataem = np.expand_dims(dataem, axis=-1)

    data_y = np.concatenate((data_y, dataep, dataem), axis=1).T

    plt.figure(figsize=(8, 7))

    ax = plt.subplot(111)

    plt.title(title, fontsize=14)
    plt.locator_params(axis='x', nbins=len(data_x) + 1)
    delta_x = (data_x[1] - data_x[0]) / 2
    bp = plt.boxplot(data_y, positions=data_x,
                     manage_ticks=False, meanline=True, showmeans=True,
                     widths=delta_x, patch_artist=True, sym='', zorder=3)

    xnew = np.linspace(data_x.min(), data_x.max(), len(data_x))
    if axis_x:
        plt.xlabel(axis_x, fontsize=12)
    if axis_y:
        plt.ylabel(axis_y, fontsize=12)
    mean_data = data_y_init
    spl = make_interp_spline(data_x, mean_data, k=1)  # type: BSpline
    power_smooth = spl(xnew)
    if 'mean_label' in kwargs:
        label = kwargs['mean_label']
    else:
        label = 'mean efficiency'
    plt.plot(xnew, power_smooth, ls='--', color='tab:orange', label=label, lw=3, zorder=4)

    boxplot_style_data(bp)
    ax.grid()
    ax.legend(loc=0)
    if data_y.max() < 1.1:
        plt.yticks(np.round(np.linspace(0, 1, 11), decimals=2))
    if 'scale' in kwargs:
        plt.yscale(kwargs['scale'])
    plt.tight_layout()
    plt.rcParams['savefig.facecolor'] = 'white'
    os.makedirs('../output', exist_ok=True)
    plt.savefig(f'../output/{model_name}_{title.lower().replace(" ", "_").replace(".","_")}.png', dpi=300)
    plt.show()

def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]            # [N,D]

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False