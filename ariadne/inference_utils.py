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

LOGGER = logging.getLogger('ariadne.prepare')


def get_seeds_with_vertex(hits):
    # first station hits
    st0_hits = hits[hits.station == 0][['x', 'y', 'z']].values
    vertex = hits[(hits.station == 0) & (hits.track != -1)][['vx', 'vy', 'vz']][0]  # change to mode
    # create seeds
    seeds = np.zeros((len(st0_hits), 2, 3))
    seeds[:, 0] = vertex
    seeds[:, 1] = st0_hits
    return seeds


def get_seeds(hits):
    temp1 = hits[hits.station == 0]
    st0_hits = hits[hits.station == 0][['x', 'y', 'z']].values
    temp2 = hits[hits.station == 1]
    st1_hits = hits[hits.station == 1][['x', 'y', 'z']].values
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

def compute_metrics(real_tracks,
                    labels,
                    tracks_predicted_right,
                    tracks_predicted_all,
                    precision_list,
                    recall_list,
                    verbose=False):
    tracks_predicted_right += labels.sum()
    tracks_predicted_all += len(labels)
    if verbose:
        LOGGER.info(labels.sum().astype("float") / len(labels))
    if len(real_tracks) > 0:
        recall_list.append(labels.sum().astype("float") / len(real_tracks))
        precision_list.append(labels.sum().astype("float") / len(labels))
    return tracks_predicted_right, tracks_predicted_all


def get_labels(gt_tracks, predicted_tracks, use_gpu=False):
    torch.cuda.empty_cache()
    if use_gpu:
        labels_for_ellipses = torch.zeros(len(predicted_tracks), dtype=torch.bool, device="cuda")
        assert len(predicted_tracks) > 0, 'Can not compute labels for empty set of tracks!'
        if len(gt_tracks) > 0:
            expanded_tracks = gt_tracks.unsqueeze(0).repeat((len(predicted_tracks), 1, 1, 1))
            expanded_xs = predicted_tracks.unsqueeze(1).repeat((1, expanded_tracks.shape[1], 1,1))
            labels_for_ellipses += torch.any(torch.all(torch.all(torch.isclose(expanded_tracks.float(), expanded_xs.float().to("cuda")), dim=-1), dim=-1),
                                             dim=-1)
            assert len(labels_for_ellipses) == len(expanded_xs), 'length of labels and xs is different!'
        labels_for_ellipses_numpy = labels_for_ellipses.detach().cpu().numpy()
        del labels_for_ellipses
        return labels_for_ellipses_numpy
    else:
        labels_for_ellipses = np.zeros(len(predicted_tracks), dtype=bool)
        assert len(predicted_tracks) > 0, 'Can not compute labels for empty set of tracks!'
        if len(gt_tracks) > 0:
            expanded_tracks = np.expand_dims(gt_tracks, 0).repeat(len(predicted_tracks), 0)
            expanded_xs = np.expand_dims(predicted_tracks, 1).repeat(expanded_tracks.shape[1], 1)
            labels_for_ellipses += np.any(
                np.all(np.all(np.equal(expanded_tracks, expanded_xs), axis=-1),
                       axis=-1), axis=-1)
            assert len(labels_for_ellipses) == len(expanded_xs), 'length of labels and xs is different!'
    return labels_for_ellipses


def prolong(x, gru, nearest_hits_mask, nearest_hits, stations_gone, use_gpu=False):
    if use_gpu:

        xs_for_prolong = x.unsqueeze(1).repeat((1, nearest_hits_mask.shape[-1], 1,1))
        grus_for_prolong = gru.unsqueeze(1).repeat((1, nearest_hits_mask.shape[-1], 1,1))
        prolonged_xs = torch.zeros((len(xs_for_prolong),
                                    nearest_hits_mask.shape[-1],
                                    xs_for_prolong.shape[2] + 1,
                                    3))
        prolonged_xs[:, :, :xs_for_prolong.shape[2], :] = xs_for_prolong
        prolonged_xs[:, :, xs_for_prolong.shape[2], :] = nearest_hits
        prolonged_xs = prolonged_xs[nearest_hits_mask].reshape(-1, stations_gone + 1, 3)
        prolonged_grus = grus_for_prolong[nearest_hits_mask].reshape(-1,
                                                                     grus_for_prolong.shape[-2],
                                                                     grus_for_prolong.shape[-1])
    else:
        xs_for_prolong = np.expand_dims(x, 1).repeat(nearest_hits_mask.shape[-1], 1)
        grus_for_prolong = np.expand_dims(gru.detach().cpu().numpy(), 1).repeat(nearest_hits_mask.shape[-1], 1)
        prolonged_xs = np.zeros(
            (len(xs_for_prolong), nearest_hits_mask.shape[-1], xs_for_prolong.shape[2] + 1, 3))

        prolonged_xs[:, :, :xs_for_prolong.shape[2], :] = xs_for_prolong
        prolonged_xs[:, :, xs_for_prolong.shape[2], :] = nearest_hits
        prolonged_xs = prolonged_xs[nearest_hits_mask].reshape(-1, stations_gone + 1, 3)
        prolonged_grus = grus_for_prolong[nearest_hits_mask].reshape(-1,
                                                                     grus_for_prolong.shape[-2],
                                                                     grus_for_prolong.shape[-1])
    return prolonged_xs, prolonged_grus