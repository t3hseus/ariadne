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
    temp_y = df[df['station'] > 1][['z', 'phi', 'track']]
    rows = itertools.product(first_station.iterrows(), second_station.iterrows())
    df = pd.DataFrame(left.append(right) for (_, left), (_, right) in rows)
    df = df.sample(frac=1).reset_index(drop=True)
    df_label = (df['track_left'] == df['track_right']) & (df['track_left'] != -1)
    x = df[['z_left', 'phi_left', 'r_left', 'z_right', 'phi_right', 'r_right']].values.reshape((-1, 2, 3))
    y = np.full((len(df_label), 2), -2.)
    y[df_label == 1] = np.squeeze(np.array(list(map(lambda x: temp_y[temp_y['track'] == x][['z', 'phi']].values,
                                                    df[df_label == 1]['track_left']))), 1)
    if not return_momentum:
        return x, y, df_label
    df_all_fake = (df['track_left'] == df['track_right']) & (df['track_left'] == -1)
    df_first_fake = (df['track_left'] == -1) & (df['track_right'] != -1)
    temp_momentum = df[['px_left', 'py_left', 'pz_left', 'track_left']]
    temp_momentum.loc[df_first_fake, ['px_left', 'py_left', 'pz_left']] = 0.
    temp_momentum.loc[df_all_fake, ['px_left', 'py_left', 'pz_left']] = df[['px_right', 'py_right', 'pz_right']]
    momentum = temp_momentum[['px_left', 'py_left', 'pz_left']].values
    return x, y, momentum, df_label

def get_seeds_with_vertex(hits):
    # first station hits
    st0_hits = hits[hits.station == 0][['x', 'y', 'z']].values
    vertex = hits[(hits.station == 0) & (hits.track != -1)][['vx', 'vy', 'vz']][0]  # change to mode
    # create seeds
    seeds = np.zeros((len(st0_hits), 2, 3))
    seeds[:, 0] = vertex
    seeds[:, 1] = st0_hits
    return seeds


def get_seeds(hits, columns=('x','y','z')):
    temp1 = hits[hits.station == 0]
    st0_hits = hits[hits.station == 0][columns].values
    temp2 = hits[hits.station == 1]
    st1_hits = hits[hits.station == 1][columns].values
    # all possible combinations
    idx0 = range(len(st0_hits))
    idx1 = range(len(st1_hits))
    idx_comb = itertools.product(idx0, idx1)
    # unpack indices
    idx0, idx1 = zip(*idx_comb)
    idx0 = list(idx0)
    idx1 = list(idx1)
    # create seeds array
    seeds = np.zeros((len(idx0), 2, 3))
    seeds[:, 0, ] = st0_hits[idx0]
    seeds[:, 1, ] = st1_hits[idx1]
    return seeds


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

def get_extreme_points_tensor(params):
    # width dependent measurements
    result = torch.zeros((len(params),4))
    half_widths = params[:,3] / 2
    result[:,0] = params[:,0] - half_widths
    result[:,1] = params[:,0] + half_widths
    # height dependent measurements
    half_height = params[:,4]/ 2
    result[:,2] = params[:,1] - half_height
    result[:,3] = params[:,1] + half_height
    del half_height
    del half_widths
    return result

def find_hits_in_ellipse(ellipses, last_station_hits, return_numpy=False):
    torch.cuda.empty_cache()
    dists = torch.cdist(ellipses[:,:3].float(), last_station_hits.float())
    minimal = last_station_hits[torch.argsort(dists, dim=1)]
    is_in_ellipse = point_in_ellipse(ellipses, minimal)
    if return_numpy:
        minimal = minimal.detach().cpu().numpy()
        is_in_ellipse = is_in_ellipse.detach().cpu().numpy()
    del dists
    return minimal, is_in_ellipse

def filter_hits_in_one_ellipse(ellipse, nearest_hits, z_last=True, filter_station=True, find_n=10):
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
    ellipses = np.expand_dims(ellipse, 2)
    found_hits = nearest_hits.reshape(-1, find_n, nearest_hits.shape[-1])
    if z_last:
        x_part = (nearest_hits[:, 0] - ellipses[0].repeat(find_n)) ** 2 / ellipses[3].repeat(find_n) ** 2
        y_part = (nearest_hits[:, 1] - ellipses[1].repeat(find_n)) ** 2 / ellipses[4].repeat(find_n) ** 2
    else:

        x_part = (found_hits[:, :, 1] - ellipses[:, 1].repeat(find_n, 1)) ** 2 / ellipses[:, -2].repeat(find_n, 1) ** 2
        y_part = (found_hits[:, :, 2] - ellipses[:, 2].repeat(find_n, 1)) ** 2 / ellipses[:, -1].repeat(find_n, 1) ** 2
    left_side = x_part + y_part
    is_in_ellipse = left_side <= 1
    return nearest_hits[is_in_ellipse, :], is_in_ellipse

def find_nearest_hit(ellipses, last_station_hits, index=None, find_n=10):
    """ Function to end-to-end find nearest_hits on one plane (2-DIM)! Used for BES now (maybe outdated)"""
    #numpy, numpy -> numpy, numpy
    if index is None:
        index = faiss.IndexFlatL2(2)
        #index.train(last_station_hits.astype('float32'))
        index.add(last_station_hits.astype('float32'))
    #ellipses = torch_ellipses.detach().cpu().numpy()
    centers = ellipses[:, :2]
    find_n = min(find_n, len(last_station_hits))
    _, i = index.search(np.ascontiguousarray(centers.astype('float32')), find_n)
    found_hits = last_station_hits[i.flatten()]
    found_hits = found_hits.reshape(-1, find_n, found_hits.shape[-1])
    ellipses = np.expand_dims(ellipses, 2)
    x_part = (found_hits[:, :, 0] - ellipses[:, 0].repeat(find_n, 1))**2 / ellipses[:, 2].repeat(find_n, 1)**2
    y_part = (found_hits[:, :, 1] - ellipses[:, 1].repeat(find_n, 1))**2 / ellipses[:, 3].repeat(find_n, 1)**2
    left_side = x_part + y_part
    is_in_ellipse = left_side <= 1
    return found_hits, is_in_ellipse

def find_nearest_hits_in_index(centers, index, find_n=100, n_dim=2):
    """Function to search in index for nearest hits in 3d-space
    Args:
        centers (np.array of shape (*,3): centers for which nearest hits are needed
        index (faiss.IndexFlatL2 or other): some index with stored information about hits
        find_n (int, 100 by default): n_hits to find. If less hits were found, redundant positions contain -1
    Returns:
        numpy.ndarray with hits positions in original array
    """
    assert centers.shape[1] == n_dim, f'index is {n_dim}-dimentional, please add z-coordinate to centers'
    _, i = index.search(np.ascontiguousarray(centers.astype('float32')), find_n)
    return i


def filter_hits_in_ellipses(ellipses, nearest_hits, hits_index, z_last=True, filter_station=True, find_n=10, n_dim=3):
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
    assert nearest_hits.shape[-1] == n_dim, f"index is {n_dim}-dimentional, please add z-coordinate to centers"
    if nearest_hits.ndim < 2:
        nearest_hits = np.expand_dims(nearest_hits, 0)
    if nearest_hits.ndim < 3:
        nearest_hits = np.expand_dims(nearest_hits, 0)
    assert ellipses.shape[-1] == nearest_hits.shape[-1]+2, f"index is {n_dim}-dimentional, you need to provide z-coordinate (z_c, x_c, y_c, x_r, y_r) or (x_c, y_c, z_c, x_r, y_r)"
    #ellipses = np.expand_dims(ellipses, -1)
    #find_n = len(nearest_hits)
    ellipses = np.expand_dims(ellipses, 2)
    #found_hits = nearest_hits.reshape(-1, find_n, nearest_hits.shape[-1])
    if z_last:
        x_part = (ellipses[:,0].repeat(find_n,1) - nearest_hits[:, :, 0]) / ellipses[:, -2].repeat(find_n,1)
        #print(x_part**2)
        y_part = (ellipses[:,1].repeat(find_n,1) - nearest_hits[:, :, 1]) / ellipses[:, -1].repeat(find_n,1)
        #print(y_part**2)
    else:
        x_part = (nearest_hits[:, :, 1] - ellipses[:, 1].repeat(find_n, 1)) / ellipses[:, 1].repeat(find_n, 1)
        y_part = (nearest_hits[:, :, 2] - ellipses[:, 2].repeat(find_n, 1)) / ellipses[:, 2].repeat(find_n, 1)
    left_side = x_part**2 + y_part**2
    is_in_ellipse = left_side <= 1
    is_in_ellipse *= hits_index != -1
    return nearest_hits, is_in_ellipse

def get_tracks(df, min_len=4):
    tracks = df[df['track'] != -1].groupby('track')
    multiplicity = tracks.ngroups
    tracks_vs_len = {3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    all_tracks = []
    for i, track in tracks:
        temp_track = track[['x', 'y', 'z']].values
        if len(temp_track) >= min_len:
            tracks_vs_len[len(temp_track)].append(temp_track)
            all_tracks.append(temp_track)
    for stations_in_track, this_track_list in tracks_vs_len.items():
        if len(this_track_list) > 0:
            tracks_vs_len[stations_in_track] = np.stack(this_track_list, 0)
    return tracks_vs_len, all_tracks, multiplicity

def get_data_for_empty_ellipses(x, preds, grus, mask):
    empty_ellipses_mask = (mask.sum(axis=-1) == 0) # if nothing is found in ellipse
    empty_ellipses = preds[empty_ellipses_mask]#.detach().cpu().numpy()
    full_ellipses = preds[~empty_ellipses_mask]
    empty_xs = x[empty_ellipses_mask]
    empty_grus = grus[empty_ellipses_mask]#.detach().cpu().numpy()
    return empty_xs, empty_ellipses, empty_grus, full_ellipses

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

def is_ellipse_intersects_module_tensor(ellipse_params,
                                       module_params):
    # calculate top, bottom, left, right
    ellipse_bounds = get_extreme_points_tensor(ellipse_params)
    module_bounds = get_extreme_points(*module_params)
    # check conditions
    mask = torch.zeros(len(ellipse_bounds), dtype=torch.bool)
    mask = ellipse_bounds[:, 0].le(module_bounds[1])
    mask = mask * ellipse_bounds[:, 1].ge(module_bounds[0])
    mask = mask * ellipse_bounds[:, 2].le(module_bounds[3])
    mask = mask * ellipse_bounds[:, 2].ge(module_bounds[2])
    del ellipse_bounds
    return mask

def is_ellipse_intersects_module_with_border(ellipse_params,
                                 module_params,
                                 treshold=0.5):
    # calculate top, bottom, left, right
    ellipse_bounds = get_extreme_points(*ellipse_params)
    module_bounds = list(get_extreme_points(*module_params))
    module_bounds[0] += module_params[-2] * treshold/2.
    module_bounds[2] += module_params[-1] * treshold/2.
    module_bounds[1] -= module_params[-2] * treshold/2.
    module_bounds[3] -= module_params[-1] * treshold/2.
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

def is_ellipse_intersects_station_tensor(ellipse_params,
                                  station_params):
    module_intersections = torch.zeros(len(ellipse_params), dtype=torch.bool)
    for module in station_params:
        ellipse_in_module = is_ellipse_intersects_module_tensor(
            ellipse_params, module)
        # add to the list
        module_intersections += ellipse_in_module
    return module_intersections

def is_ellipse_intersects_station_with_border(ellipse_params,
                                              station_params,
                                              treshold=0.5):
    module_intersections = []
    for module in station_params:
        ellipse_in_module = is_ellipse_intersects_module_with_border(
            ellipse_params, module, treshold)
        # add to the list
        module_intersections.append(ellipse_in_module)
    return any(module_intersections)

def search_in_index(centers, index, find_n=100, n_dim=2):
    """Function to search in index for nearest hits in 3d-space
    Args:
        centers (np.array of shape (*,3): centers for which nearest hits are needed
        index (faiss.IndexFlatL2 or other): some index with stored information about hits
        find_n (int, 100 by default): n_hits to find. If less hits were found, redundant positions contain -1
    Returns:
        numpy.ndarray with hits positions in original array
    """
    assert centers.shape[1] == n_dim, f'index is {n_dim}-dimentional, please add z-coordinate to centers'
    _, i = index.search(np.ascontiguousarray(centers.astype('float32')), find_n)
    return i

def store_in_index(event_hits, index=None, num_components=3):
    """Function to create faiss.IndexFlatL2 or add hits to it.
    Args:
        event_hits (np.ndarray of shape (N,num_components)): array with N hits to store in faiss index of event
        index ( faiss.IndexFlatL2 index or None): if None, new index is created, else hits are added to it
        num_components (int, 3 by default): number of dimentions in event space
    """
    #numpy -> numpy
    index = faiss.IndexFlatL2(num_components)
    #index.train(event_hits.astype('float32'))
    index.add(event_hits.astype('float32'))
    return index
