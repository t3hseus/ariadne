import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
import os
import logging

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
