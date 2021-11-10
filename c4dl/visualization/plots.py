from datetime import datetime, timedelta
import os
import string

from matplotlib import colors, gridspec, patches, pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
from pyresample.plot import area_def2basemap
from pyresample.geometry import AreaDefinition
import seaborn as sns

from .. import projection
from ..ml.models import base, gradboost


def storm_tracks(motion_vectors, tracks, time, marker_index=(0,-1), Z_time=None,
    cax=None):

    if Z_time is None:
        Z_time = time
    Z = motion_vectors.reader.variable_for_time(Z_time,
        motion_vectors.motion_var)
    (X, V) = motion_vectors.get_vectors(time)
    current_tracks = tracks[time]

    # plot radar field
    plt.imshow(Z, norm=colors.Normalize(20,60))
    plt.gca().tick_params(bottom=False, left=False, 
        labelbottom=False, labelleft=False)
    cb = plt.colorbar(cax=cax)
    cb.ax.set_title("dBZ")

    # plot motion vectors
    plt.quiver(
        X[:, 0], X[:, 1], V[:, 0], V[:, 1],
        color="black", angles="xy", scale_units="xy", scale=0.2,
    )

    for i in range(current_tracks[0].shape[0]):
        for k in range(2):
            (x, y) = (current_tracks[k][i,:,0], current_tracks[k][i,:,1])
            plt.plot(x, y, 'tab:orange', linewidth=0.5)
            if k == marker_index[0]:
                j = marker_index[1]
                plt.scatter(x[j], y[j], color=(0,0,0,0), edgecolors='tab:orange', 
                    s=5, marker='o')
        x = [current_tracks[0][i,-1,0], current_tracks[1][i,0,0]]
        y = [current_tracks[0][i,-1,1], current_tracks[1][i,0,1]]
        plt.plot(x, y, 'tab:orange', linewidth=0.5)


def storm_tracks_anim(motion_vectors, tracks, time, out_dir):
    interval = motion_vectors.time_interval
    Z_time = time - interval * tracks[time][0].shape[1]
    for k in range(2):
        for j in range(tracks[time][k].shape[1]):
            storm_tracks(motion_vectors, tracks, time, marker_index=(k,j),
                Z_time=Z_time)
            out_fn = os.path.join(out_dir, "frame-{}-{}.png".format(k,j))
            plt.savefig(out_fn, bbox_inches='tight')
            plt.close()
            Z_time += interval


def track_features(tracks, reader, features, time, out_dir, 
    variable="MAXZ", flow_variables=["FLOW-U", "FLOW-V"],
    feature="nb-33", box_size=33,
    nb_readers=None, nb_variables=None, track_ind=0,
    interval=timedelta(minutes=5), data_range=(0,60)):

    var_ranges = {
        "mean_elevation": (0,500),
        "ABIC08": (273.15-64.65, 273.15-52)#273.15-29.25)
    }

    n = tracks[time][0].shape[1]
    t = time - n*interval

    track = tracks[time][0][track_ind,:,:]
    norm = colors.Normalize(*data_range)
    gs = gridspec.GridSpec(3,3)

    area = reader.grid_projection.area
    bmap = area_def2basemap(area)
    bmap.resolution = 'l'

    if nb_readers is None:
        nb_readers = [reader]
    if nb_variables is None:
        nb_variables = [variable]

    for k in range(n):
        fig = plt.figure(figsize=(10,10))
        (x,y) = track[k,:]
        data = reader.variable_for_time(t, variable)
        ax = fig.add_subplot(gs[:2,:3])
        im = bmap.imshow(np.flipud(data), norm=norm, ax=ax)
        x0 = int(round(x)) - box_size//2
        x1 = x0 + box_size
        y0 = int(round(y)) - box_size//2
        y1 = y0 + box_size
        box_x = np.array([x0,x1,x1,x0,x0])
        box_y = np.array([y0,y0,y1,y1,y0])
        (box_x, box_y) = bmap(*area.colrow2lonlat(box_x,box_y))
        bmap.plot(box_x, box_y, 'k', ax=ax)

        bmap.drawstates(ax=ax, color='gray')
        bmap.drawcoastlines(ax=ax, color='gray')

        if flow_variables is not None:
            u = reader.variable_for_time(t, flow_variables[0])
            v = reader.variable_for_time(t, flow_variables[1])
            (x,y) = np.mgrid[:data.shape[1]:40,:data.shape[0]:40]
            (x,y) = bmap(*area.colrow2lonlat(
                x.round().astype(int),
                y.round().astype(int)
            ))
            bmap.quiver(x, y, u[::40,::40], v[::40,::40], ax=ax)
            
        fig.colorbar(im, ax=ax)
        
        for (m,(r,v)) in enumerate(zip(nb_readers, nb_variables)):
            ax = fig.add_subplot(gs[2,m])
            feat_data = features[time][r.name][v][feature][track_ind,k,...]
            if r.name == reader.name:
                f_norm = norm
            else:
                f_norm = colors.Normalize(*var_ranges.get(v))
            im = ax.imshow(feat_data, norm=f_norm)
            fig.colorbar(im, ax=ax)
            ax.set_title(r.name+"::"+v)

        out_fn = "track-{}.png".format(t.strftime("%Y%m%d%H%M"))
        fig.savefig(os.path.join(out_dir,out_fn), bbox_inches='tight')
        plt.close()

        t += interval


def plot_study_area(config):
    area_def = AreaDefinition(**projection.known_area(config["area"]))
    (lon_bounds, lat_bounds) = area_def.get_boundary_lonlats()

    sides = [
        (lon_bounds.side1, lat_bounds.side1),
        (lon_bounds.side2, lat_bounds.side2),
        (lon_bounds.side3, lat_bounds.side3),
        (lon_bounds.side4, lat_bounds.side4)
    ]

    m = Basemap(
        width=5000000,
        height=4000000,
        resolution='l',
        projection='stere',
        lat_ts=config["area"]["params"]["lat0"],
        lat_0=config["area"]["params"]["lat0"],
        lon_0=config["area"]["params"]["lon0"]
    )

    m.drawcoastlines(linewidth=1.5)
    m.drawcountries(linewidth=1)
    m.drawstates(color=(0.5,0.5,0.5))

    for (lon_side, lat_side) in sides:
        (x,y) = m(lon_side, lat_side)
        m.plot(x, y, color="tab:blue", linewidth=2)

    radar_coords = np.array([
        [-70.25694, 43.89139],                
        [-71.13750, 41.95583],
        [-72.86444, 40.86556],
        [-73.16639, 44.51111],
        [-74.06444, 42.58639],
        [-74.41111, 39.94694],        
        [-75.68000, 43.75583],
        [-75.98500, 42.19972],
        [-78.00389, 40.92306],
        [-78.73694, 42.94861],
        [-80.21833, 40.53167]
    ])

    (radar_x, radar_y) = m(radar_coords[:,0], radar_coords[:,1])
    m.plot(radar_x, radar_y, 'o', markeredgecolor='tab:orange',
        markerfacecolor=(0,0,0,0))


source_colors = {
    "nexrad": "tab:blue",
    "ecmwf": "tab:orange",
    "goesabi": "tab:green",
    "goesglm": "tab:purple",
    "aster": "tab:brown"
}
def get_source_color(source):
    return source_colors.get(source, "tab:gray")


source_names = {
    "nexrad": "NEXRAD",
    "ecmwf": "ECMWF",
    "goesabi": "GOES ABI",
    "goesglm": "GOES GLM",
    "aster": "ASTER",
    "composites": "Composite",
}
def get_source_name(source):
    return source_names.get(source, "")


feature_names = {
    "MAXZ": "Column maximum reflectivity",
    "VIL": "Vertical integrated liquid",
    "ECHOTOP-25": "25 dBZ echo top height",
    "ECHOTOP-35": "35 dBZ echo top height",
    "ECHOTOP-45": "45 dBZ echo top height",
    "FLOW-U": "Optical flow U-direction",
    "FLOW-V": "Optical flow V-direction",
    "deg0l": "$0\\degree$C isothermal level",
    "2d": "$2$ m dewpoint temperature",
    "2t": "$2$ m temperature",
    "10u": "$10$ m wind U component",
    "10v": "$10$ m wind V component",
    "100u": "$100$ m wind U component",
    "100v": "$100$ m wind V component",
    "200u": "$200$ m wind U component",
    "200v": "$200$ m wind V component",
    "litota1": "Last hour lightning density",
    "bld": "Boundary layer dissipation",
    "blh": "Boundary layer height",
    "cbh": "Cloud base height",
    "cape": "CAPE",
    "capes": "CAPE shear",
    "cin": "Convective inhibition",
    "cp": "Convective precipitation",
    "crr": "Convective rain rate",
    "csfr": "Convective snowfall rate",
    "e": "Evaporation",
    "zust": "Friction velocity",
    "z": "Geopotential",
    "hcct": "Height of convective cloud top",
    "hwbt1": "Height of $1\\degree$C wet-bulb T",
    "hwbt0": "Height of $0\\degree$C wet-bulb T",
    "hcc": "High cloud cover",
    "kx": "K index",
    "lsrr": "Large scale rain rate",
    "lssfr": "Large scale snowfall rate",
    "lsp": "Large-scale precipitation",
    "lspf": "Large-scale precipitation fraction",
    "lcc": "Low cloud cover",
    "mxcape6": "Maximum CAPE last 6 h",
    "mxcapes6": "Maximum CAPES last 6 h",
    "msl": "Mean sea level pressure",
    "mcc": "Medium cloud cover",
    "pev": "Potential evaporation",
    "ptype": "Precipitation type",
    "src": "Skin reservoir content",
    "skt": "Skin temperature",
    "sf": "Snowfall",
    "slhf": "Surface latent heat flux",
    "ssr": "Surface net solar radiation",
    "ssrc": "Surface net solar radiation, clear sky",
    "str": "Surface net thermal radiation",
    "strc": "Surface net thermal radiation, clear sky",
    "sp": "Surface pressure",
    "sshf": "Surface sensible heat flux",
    "tcc": "Total cloud cover",
    "tciw": "Column cloud ice water",
    "tclw": "Column cloud liquid water",
    "tcrw": "Column rain water",
    "tcsw": "Column snow water",
    "tcslw": "Column supercooled liquid water",
    "tcw": "Total column water",
    "tcwv": "Column water vapour",
    "tp": "Total precipitation",
    "tprate": "Total precipitation rate",
    "totalx": "Total totals index",
    "viwve": "Eastward water vapour flux",
    "viwvn": "Northward water vapour flux",
    "vimd": "Moisture divergence",
    "ABIC01": "Band 1",
    "ABIC02": "Band 2",
    "ABIC03": "Band 3",
    "ABIC04": "Band 4",
    "ABIC05": "Band 5",
    "ABIC06": "Band 6",
    "ABIC07": "Band 7",
    "ABIC08": "Band 8",
    "ABIC09": "Band 9",
    "ABIC10": "Band 10",
    "ABIC11": "Band 11",
    "ABIC12": "Band 12",
    "ABIC13": "Band 13",
    "ABIC14": "Band 14",
    "ABIC15": "Band 15",
    "ABIC16": "Band 16",
    "HT": "Cloud top height",
    "PRES": "Cloud top pressure",
    "CAPE": "CAPE",
    "KI": "K index",
    "LI": "Lifted index",
    "SI": "Showalter index",
    "TT": "Total totals index",
    "COD": "Cloud optical depth",
    "flash_density": "Lightning flash density",
    "flash_energy_density": "Lightning flash energy density",
    "event_density": "Lightning event density",
    "event_energy_density": "Lightning event energy density",
    "ABIC07-C08": "Bands 7-8",
    "ABIC07-C09": "Bands 7-9",
    "ABIC07-C10": "Bands 7-10",
    "ABIC08-C09": "Bands 8-9",
    "ABIC08-C10": "Bands 8-10",
    "ABIC11-C13": "Bands 11-13",
    "ABIC12-C13": "Bands 12-13",
    "upslope_flow_radar": "Upslope flow",
    "gradient_x": "Slope in x-direction",
    "gradient_y": "Slope in y-direction",
    "gradient_abs": "Slope absolute value",
    "mean_elevation": "Mean elevation",
    "roughness": "Surface roughness",
    "trt": "Thunderstorm rank",
    "poh": "Probability of hail"
}
def get_feature_name(feature):
    return feature_names.get(feature, feature)


def plot_gradboost_importance(total_importance, source_importance,
    source_leadtimes, num_importance=20,
    source_x_range=(0,60), source_style="lines", legend=True, title=None,
    fig=None, axes=None):

    if axes is None:
        fig = plt.figure(figsize=(8,16))
        gs = gridspec.GridSpec(4, 1, hspace=0.1)

    sources = [s for (g,s) in sorted(source_importance[0])]
    
    ax = axes[0] if (axes is not None) else fig.add_subplot(gs[:-1,0])
    bar_y = np.arange(min(len(total_importance), num_importance))[::-1]
    max_importance = total_importance[0][0]
    bar_width = np.array([g for (g,f) in total_importance[:num_importance]]) / max_importance
    color = [get_source_color(f.split("::")[0]) for (g,f) in total_importance]
    ax.barh(bar_y, bar_width, color=color)
    for (i,y) in enumerate(bar_y):
        (source, feature) =  total_importance[i][1].split("::")
        label = get_source_name(source) + " " + get_feature_name(feature)
        ax.text(0.01, y, label, verticalalignment='center')
    if legend:
        legend_elements = [
            patches.Patch(facecolor=get_source_color(s), edgecolor=None,
                label=source_names[s])
            for s in sorted(sources)
        ]
        ax.legend(handles=legend_elements, loc='lower right')
    
    ax.set_xlim((0,1.03))
    ax.set_ylim((-0.5, bar_y[0]+0.5))
    ax.tick_params(axis='y', left=False, labelleft=False, labelbottom=False)
    if title:
        ax.set_title(title)
    

    ax = axes[1] if (axes is not None) else fig.add_subplot(gs[-1,0])
    bar_width = np.diff(source_leadtimes)
    
    cumul = np.zeros(len(source_importance))
    for source in sources:
        imp = []
        for source_imp in source_importance:
            si = {s: g for (g,s) in source_imp}
            imp.append(si[source] / sum(si[s] for s in sources))
        imp = np.array(imp)

        color = get_source_color(source)
        if source_style == "lines":
            ax.fill_between(
                source_leadtimes,
                cumul,
                cumul+imp,
                facecolor=color,
                edgecolor=None
            )
        elif source_style == "bars":
            ax.bar(
                source_leadtimes[:-1],
                imp,
                width=bar_width,
                bottom=cumul,
                color=color,
                align='edge'
            )
        
        cumul += imp
    
    ax.set_xlim(*source_x_range)
    ax.set_ylim((0,1))
    #ax.tick_params(axis='y', left=False, labelleft=False)

    return fig


def exclusion_plot(combination_metrics, dataset="test", fig=None, axes=None,
    variable_name=None, subplot_index=0, significant_digits=3):
    labels = []
    metrics = {}

    notation = {
        "aster": "ASTER",
        "goesabi": "ABI",
        "goesglm": "GLM",
        "ecmwf": "ECMWF",
        "nexrad": "NEXRAD"
    }
    metric_notation = {
        "binary": "error rate",
        "cross_entropy": "cross-entropy",
        "mae": "MAE",
        "rmse": "RMSE"
    }

    for subset in combination_metrics:
        label = []
        for source in notation:
            if source in subset:
                label.append(
                    "$\\bf{{{}}}$".format(notation[source])
                )
        label = " ".join(label)
        labels.append(label)

        for metric in combination_metrics[subset]:
            if metric not in metrics:
                metrics[metric] = []
            metrics[metric].append(combination_metrics[subset][metric][dataset])

    metrics_names = [metric_notation[k] for k in metrics]
    metrics_tables = {metric: np.full((8,4), np.nan) for metric in metrics}
    metric_pos = {
        frozenset(("ecmwf", "goesglm", "aster", "nexrad", "goesabi")): (0,0),
        frozenset(("ecmwf", "goesglm", "aster", "nexrad")): (0,1),
        frozenset(("ecmwf", "goesglm", "aster", "goesabi")): (0,2),
        frozenset(("ecmwf", "goesglm", "aster")): (0,3),

        frozenset(("ecmwf", "goesglm", "nexrad", "goesabi")): (1,0),
        frozenset(("ecmwf", "goesglm", "nexrad")): (1,1),
        frozenset(("ecmwf", "goesglm", "goesabi")): (1,2),
        frozenset(("ecmwf", "goesglm")): (1,3),

        frozenset(("ecmwf", "aster", "nexrad", "goesabi")): (2,0),
        frozenset(("ecmwf", "aster", "nexrad")): (2,1),
        frozenset(("ecmwf", "aster", "goesabi")): (2,2),
        frozenset(("ecmwf", "aster")): (2,3),

        frozenset(("goesglm", "aster", "nexrad", "goesabi")): (3,0),
        frozenset(("goesglm", "aster", "nexrad")): (3,1),
        frozenset(("goesglm", "aster", "goesabi")): (3,2),
        frozenset(("goesglm", "aster")): (3,3),

        frozenset(("ecmwf", "nexrad", "goesabi")): (4,0),
        frozenset(("ecmwf", "nexrad")): (4,1),
        frozenset(("ecmwf", "goesabi")): (4,2),
        frozenset(("ecmwf",)): (4,3),

        frozenset(("goesglm", "nexrad", "goesabi")): (5,0),
        frozenset(("goesglm", "nexrad")): (5,1),
        frozenset(("goesglm", "goesabi")): (5,2),
        frozenset(("goesglm",)): (5,3),

        frozenset(("aster", "nexrad", "goesabi")): (6,0),
        frozenset(("aster", "nexrad")): (6,1),
        frozenset(("aster", "goesabi")): (6,2),
        frozenset(("aster",)): (6,3),

        frozenset(("nexrad", "goesabi")): (7,0),
        frozenset(("nexrad",)): (7,1),
        frozenset(("goesabi",)): (7,2),
        frozenset(()): (7,3),
    }
    metric_pos_inv = {v: k for (k, v) in metric_pos.items()}

    for metric in metrics:
        for subset in combination_metrics:
            subset_frozen = frozenset(subset)
            (i,j) = metric_pos[subset_frozen]
            metrics_tables[metric][i,j] = combination_metrics[subset][metric][dataset]

    xlabels_show = frozenset(("nexrad", "goesabi"))
    ylabels_show = frozenset(("ecmwf", "goesglm", "aster"))

    with sns.plotting_context("paper"):
        if fig is None:
            fig = plt.figure(figsize=(3.125*len(metrics),7.5))
        
        for (i,metric) in enumerate(metrics):
            xlabels = [
                "\n".join(sorted(notation[s] for s in metric_pos_inv[0,i] & xlabels_show))
                for i in range(metrics_tables[metric].shape[1])
            ]
            ylabels = [
                "\n".join(sorted(notation[s] for s in metric_pos_inv[i,0] & ylabels_show))
                for i in range(metrics_tables[metric].shape[0])
            ]

            ax = axes[i] if (axes is not None) else fig.add_subplot(1,len(metrics),i+1)
            heatmap = sns.heatmap(
                metrics_tables[metric],
                xticklabels=xlabels,
                yticklabels=ylabels,
                annot=True,
                fmt='#.{}g'.format(significant_digits),
                square=True,
                ax=ax,
                cbar_kws={"orientation": "horizontal"}
            )
            heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90, ha='right')
            heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, ha='right')
            ax.set_title("({}) {}{}".format(
                string.ascii_lowercase[i+subplot_index],
                variable_name+" " if variable_name else "",
                metric_notation[metric]
            ))
            ax.tick_params(axis='both', bottom=False, left=False,
                labelleft=(i+subplot_index==0))

    return fig


def metrics_by_time(models, metrics, past_features, future_features,
    interval=timedelta(minutes=5)):

    model_leadtimes = sorted([(int(k.split("::")[-1]), k) for k in models])

    metric_values = {metric: [] for metric in metrics}
    metric_values_persistence = {metric: [] for metric in metrics}
    metric_values_debiased = {metric: [] for metric in metrics}
    leadtimes = []

    error_funcs = {
        "mae": lambda y_pred, y: np.nanmean(abs(y-y_pred)),
        "rmse": lambda y_pred, y: np.sqrt(np.nanmean((y-y_pred)**2)),
        "cross_entropy": lambda y_pred, y: 
            -np.nanmean(y*np.log(y_pred) + (1-y)*np.log(1-y_pred)),
        "binary": lambda y_pred, y: np.nanmean(y_pred.round() != y),
    }

    for (i,k) in model_leadtimes:
        leadtimes.append(interval.total_seconds()*(i+1)/60)
        y = future_features[i]
        y_pers = models[k].preproc_model.predict(past_features)
        y_pers_debiased = y_pers + models[k].bias
        y_pred = models[k].predict(past_features)

        for metric in metrics:            
            metric_values[metric].append(error_funcs[metric](y_pred,y))
            metric_values_persistence[metric].append(
                error_funcs[metric](y_pers,y))
            metric_values_debiased[metric].append(
                error_funcs[metric](y_pers_debiased,y))

    metric_plot_params = {
        "mae": {"linestyle": "-", "label": "MAE"},
        "rmse": {"linestyle": "--", "label": "RMSE"},
        "binary": {"linestyle": "-", "label": "Binary error"},
        "cross_entropy": {"linestyle": "--", "label": "Cross-entropy"}
    }    

    fig = plt.figure()
    ax = plt.axes()

    for metric in metrics:
        params = metric_plot_params[metric].copy()

        params["label"] = "GB " + metric_plot_params[metric]["label"]
        ax.plot(leadtimes, metric_values[metric], 
            color="tab:blue", **params)
        params["label"] = "Persistence " + metric_plot_params[metric]["label"]
        ax.plot(leadtimes, metric_values_persistence[metric], 
            color="tab:orange", **params)
        params["label"] = "Debiased persistence " + metric_plot_params[metric]["label"]
        ax.plot(leadtimes, metric_values_debiased[metric], 
            color="tab:red", **params)

    ax.legend()
    ax.set_xlabel("Lead time [min]")
    ax.set_ylabel("Metric")
    ax.set_xlim((0, 60))
    ax.set_ylim((0, ax.get_ylim()[1]))

    return fig


def confusion_matrix(y_true, y_pred, axes=None, cbar_ax=None,
    xlabel=True, ylabel=True):

    if axes is None:
        axes = plt.gca()

    y_pred = y_pred.round().astype(bool)

    M = np.zeros((2,2))
    M[0,0] = np.count_nonzero(y_pred & y_true)
    M[0,1] = np.count_nonzero(y_pred & ~y_true)
    M[1,0] = np.count_nonzero(~y_pred & y_true)
    M[1,1] = np.count_nonzero(~y_pred & ~y_true)
    M /= M.sum()

    heatmap = sns.heatmap(
        M,
        xticklabels=["Yes", "No"],
        yticklabels=["Yes", "No"],
        annot=True,
        fmt='#.3f',
        square=True,
        ax=axes,
        cbar=(cbar_ax is not None),
        cbar_ax=cbar_ax,
        cbar_kws={"orientation": "horizontal"},
        vmin=0,
        vmax=1
    )
    if xlabel:
        axes.set_xlabel("Actual")
    if ylabel:
        axes.set_ylabel("Predicted")
    axes.tick_params(bottom=xlabel, labelbottom=xlabel,
        left=ylabel, labelleft=ylabel)
