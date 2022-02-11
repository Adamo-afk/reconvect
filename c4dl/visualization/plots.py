from datetime import datetime, timedelta
import os
import string

from matplotlib import colors, gridspec, patches, pyplot as plt
import numpy as np

from ..analysis import evaluation


def plot_calibration(p, occurrence_rate, names, colors_linestyles=None):
    fig = plt.figure()
    ax = fig.add_subplot()

    for model in occurrence_rate:
        if colors_linestyles is not None:
            (c, ls) = colors_linestyles[model]
        else:
            c = ls = None
        ax.plot(p, occurrence_rate[model], label=names[model],
            color=c, linestyle=ls)

    ax.plot([0,1], [0,1], color=(0.4,0.4,0.4), linestyle=":", label="_nolegend_")
    
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))
    ax.legend(loc='lower center', bbox_to_anchor=(0.5,1.03), ncol=4)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Occurrence rate")

    return fig


def plot_pr_curve(conf_matrix, names, colors_linestyles=None, show_auc=False):
    precision = {}
    recall = {}
    labels = {}
    for (k,cm) in conf_matrix.items():
        precision[k] = evaluation.precision(cm)
        recall[k] = evaluation.recall(cm)
        if show_auc:
            auc = evaluation.pr_area_under_curve(cm)
            labels[k] = f"{names[k]} (AUC: {auc:.3f})"
        else:
            labels[k] = names[k]

    fig = plt.figure()
    ax = fig.add_subplot()

    for model in precision:
        print(model, recall[model][-1], recall[model][0], precision[model][-1], precision[model][0])
        if colors_linestyles is not None:
            (c, ls) = colors_linestyles[model]
        else:
            c = ls = None
        ax.plot(recall[model], precision[model], label=labels[model],
            color=c, linestyle=ls)
    
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))
    ax.legend()
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    return fig


def plot_roc_curve(conf_matrix, names, colors_linestyles=None, show_auc=False):
    tpr = {}
    fpr = {}
    labels = {}
    for (k,cm) in conf_matrix.items():
        ((tp, fn), (fp, tn)) = cm
        tpr[k] = tp / (tp + fn)
        fpr[k] = fp / (fp + tn)
        if show_auc:
            auc = evaluation.roc_area_under_curve(cm)
            labels[k] = f"{names[k]} (AUC: {auc:.3f})"
        else:
            labels[k] = names[k]

    fig = plt.figure(figsize=())
    ax = fig.add_subplot()

    for model in tpr:
        print(model, fpr[model][-1], fpr[model][0], tpr[model][-1], tpr[model][0])
        if colors_linestyles is not None:
            (c, ls) = colors_linestyles[model]
        else:
            c = ls = None
        ax.plot(fpr[model], tpr[model], label=labels[model],
            color=c, linestyle=ls)
    
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))
    ax.legend()
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    return fig


metric_names = {
    "CSI": "Critical Success Index",
    "HSS": "Heidke Skill Score",
    "PSS": "Peirce Skill Score",
    "ETS": "Equitable Threat Score",
}
metric_funcs = {
    "CSI": evaluation.intersection_over_union,
    "HSS": evaluation.heidke_skill_score,
    "PSS": evaluation.peirce_skill_score,
    "ETS": evaluation.equitable_threat_score,
    "ROC AUC": evaluation.roc_area_under_curve,
    "PR AUC": evaluation.pr_area_under_curve,
    "POD": evaluation.recall,
    "FAR": evaluation.false_alarm_ratio
}

def plot_threshold_metric_curve(thresholds, conf_matrix, names, metric,
    colors_linestyles=None, fig=None, ax=None, legend=True,
    xlabel=True, show_best=False
):
    metric_scores = {}
    labels = {}
    for (k,cm) in conf_matrix.items():
        metric_scores[k] = metric_funcs[metric](cm)
        if show_best:      
            best = metric_scores[k].max()        
            labels[k] = f"{names[k]} (Best: {best:.3f})"
        else:
            labels[k] = names[k]

    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot()

    for (model, score) in metric_scores.items():
        if colors_linestyles is not None:
            (c, ls) = colors_linestyles[model]
        else:
            c = ls = None
        ax.plot(thresholds, score, label=labels[model],
            color=c, linestyle=ls)
    
    ylim = ax.get_ylim()
    ax.set_ylim((0,ylim[1]))
    ax.set_xlim((0,1))
    if legend:
        ax.legend()
    if xlabel:
        ax.set_xlabel("Threshold")
    ax.set_ylabel(metric_names[metric])

    return (fig, ax)


def plot_metric_leadtime(conf_matrix_lt, names, metrics=("CSI", "PSS"),
    colors_linestyles=None, fig=None, legend=True, dt_minutes=5
):
    if fig is None:
        fig = plt.figure(figsize=(6,6))
    
    leadtime = None
    
    for (i,metric) in enumerate(metrics):
        ax = fig.add_subplot(len(metrics), 1, i+1)            

        metric_scores = {}
        for (k,cm) in conf_matrix_lt.items():
            metric_scores[k] = metric_funcs[metric](cm).max(axis=0)
            if leadtime is None:
                leadtime = np.arange(1,len(metric_scores[k])+1) * dt_minutes

        for (model, score) in metric_scores.items():
            if colors_linestyles is not None:
                (c, ls) = colors_linestyles[model]
            else:
                c = ls = None
            ax.plot(leadtime, score, label=names[model],
                color=c, linestyle=ls)

        ax.text(
            0.01, 0.975,
            f"({string.ascii_lowercase[i]})",
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes
        )

        ax.set_xlim((0, leadtime[-1]))
        ylim = ax.get_ylim()
        ax.set_ylim((0, ylim[1]))
        ax.tick_params(right=True)

        if i == len(metrics)-1:
            ax.set_xlabel("Lead time [min]")
            ax.legend(loc='lower left')
        ax.set_ylabel(metric_names[metric])

    return fig


def plot_frame(ax, frame, norm=None):
    im = ax.imshow(frame.astype(np.float32), norm=norm)
    ax.tick_params(left=False, bottom=False,
        labelleft=False, labelbottom=False)
    return im


def plot_model_examples(X, Y, models, shown_inputs=(0,25,12,9),
    input_timesteps=(-4,-1), output_timesteps=(0,2,5,11),
    batch_member=0, interval_mins=5,
    input_names=("Rain rate", "Lightning", "HRV", "CTH")
):
    num_timesteps = len(input_timesteps)+len(output_timesteps)
    gs_rows = 2 * max(len(models),len(shown_inputs))
    gs_cols = num_timesteps
    gs = gridspec.GridSpec(gs_rows, gs_cols+4, wspace=0.02, hspace=0.05,
        width_ratios=[0.1,0.19]+[1]*gs_cols+[0.19,0.1])
    batch = [x[batch_member:batch_member+1,...] for x in X]
    obs = [y[batch_member:batch_member+1,...] for y in Y]

    fig = plt.figure(figsize=(gs_cols*1.5, gs_rows/2*1.5))

    # plot inputs
    input_transforms = {
        "Rain rate": lambda x: 10**(x*0.528-0.051),
        "Lightning": lambda x: x,
        "HRV": lambda x: x*100,
        "CTH": lambda x: x*2.810+5.260
    }
    input_norm = {
        "Rain rate": colors.LogNorm(0.01, 100, clip=True),
        "Lightning": colors.Normalize(0, 1),
        "HRV": colors.Normalize(0,100),
        "CTH": colors.Normalize(0,12)
    }
    input_ticks = {
        "Rain rate": [0.1, 1, 10, 100],
        "Lightning": [0, 0.5, 1],
        "HRV": [0, 25, 50, 75],
        "CTH": [0, 5, 10]
    }
    row0 = gs_rows//2 - len(shown_inputs)
    for (i,k) in enumerate(shown_inputs):
        row = row0 + 2*i        
        ip = batch[k][0,input_timesteps,:,:,0]
        ip = input_transforms[input_names[i]](ip)
        norm = input_norm[input_names[i]]
        for m in range(len(input_timesteps)):
            col = m+2
            ax = fig.add_subplot(gs[row:row+2,col])
            im = plot_frame(ax, ip[m,:,:], norm=norm)
            if i == 0:
                iv = (input_timesteps[m]+1) * interval_mins
                ax.set_title(f"${iv}\\,\\mathrm{{min}}$")
            if m == 0:
                ax.set_ylabel(input_names[i])
                cax = fig.add_subplot(gs[row:row+2,0])                
                cb = plt.colorbar(im, cax=cax)
                cb.set_ticks(input_ticks[input_names[i]])
                cax.yaxis.set_ticks_position('left')

    # plot outputs
    row0 = gs_rows//2 - len(models)
    #norm = colors.Normalize(0,1)
    min_p = 0.025
    norm = colors.LogNorm(min_p,1,clip=True)
    for (i,model) in enumerate(models):
        if model == "obs":
            Y_pred = obs[0]
        else:
            Y_pred = model.predict(batch)
        row = row0 + 2*i
        op = Y_pred[0,output_timesteps,:,:,0]        
        for m in range(len(output_timesteps)):
            col = m + len(input_timesteps) + 2
            ax = fig.add_subplot(gs[row:row+2,col])
            im = plot_frame(ax, op[m,:,:], norm=norm)
            if i==0:
                iv = (output_timesteps[m]+1) * interval_mins
                ax.set_title(f"$+{iv}\\,\\mathrm{{min}}$")
            if m == len(output_timesteps)-1:
                ax.yaxis.set_label_position("right")
                label = "Observed" if (model=="obs") else "Forecast"
                ax.set_ylabel(label)
        if i==0:
            cax = fig.add_subplot(gs[row0:gs_rows-row0,-1])            
            cb = plt.colorbar(im, cax=cax)
            cb.set_ticks([min_p, 0.05, 0.1, 0.2, 0.5, 1])
            cb.set_ticklabels([min_p, 0.05, 0.1, 0.2, 0.5, 1])
            cax.set_title("$p$", fontsize=12)

    return fig


def plot_study_area(radar_archive_path, dem_path):
    from pyresample.geometry import AreaDefinition
    import cartopy
    from ..datasets import mchradar, swissdem
    from .. import projection

    dt = datetime(2020,6,1) # dummy, not really needed

    area_def = AreaDefinition(**projection.ccs4_swiss_grid_area)
    (lons, lats) = area_def.get_lonlats()
    crs = area_def.to_cartopy_crs()
    ae = area_def.area_extent
    img_extent = [ae[0], ae[2], ae[1], ae[3]]
    ax = plt.axes(projection=crs)

    swissdem_reader = swissdem.SwissDEMReader(dem_file=dem_path)
    dem = swissdem_reader.variable_for_time(dt, "Altitude")
    dem[dem<2] = -1
    color_x = np.hstack(((np.zeros(8)+0.1),np.linspace(0.25,1,192)))
    dem_colors = plt.cm.terrain(color_x)
    dem_colors[0,:] = plt.cm.terrain(0.1)
    terrain_trunc = colors.ListedColormap(dem_colors)
    norm = colors.Normalize(-dem.max()*(8/192), dem.max())
    im = ax.imshow(dem, origin='upper', extent=img_extent, transform=crs,
        cmap=terrain_trunc, norm=norm)
    cb = plt.colorbar(im)
    cb.ax.set_ylabel("Elevation [m]")

    ax.add_feature(cartopy.feature.COASTLINE, linewidth=1.5)
    ax.add_feature(cartopy.feature.BORDERS, linewidth=1.0)    
    ax.set_global()    

    radar_coords = np.array([
        [8.512000, 47.284333],                
        [6.099415, 46.425113],
        [8.833217, 46.040791],
        [7.486552, 46.370646],
        [9.794458, 46.834974],
    ])

    ax.plot(
        radar_coords[:,0], radar_coords[:,1], 
        'o', markeredgecolor='tab:red',
        markerfacecolor=(0,0,0,0), transform=cartopy.crs.PlateCarree()
    )

    mchradar_reader = mchradar.MCHRadarReader(
        archive_path=radar_archive_path,
        variables=["RZC"],
        phys_values=False
    )
    data = mchradar_reader.variable_for_time(dt, "RZC")
    mask = (data == 255).astype(np.float32)
    img = np.zeros((mask.shape[0], mask.shape[1], 4))
    img[:,:,3] = mask * 0.3
    
    ax.imshow(img, origin='upper', extent=img_extent, transform=crs)

    sl_offset_x = 272000
    sl_offset_y = -136000
    sl_capsize = 12000
    sl_length = 256000
    sl_labelmargin = 4000
    sl_style = {"color": 'k', "linewidth": 0.85}
    ax.plot(
        [sl_offset_x, sl_offset_x+sl_length],
        [sl_offset_y, sl_offset_y],
        **sl_style
    )
    ax.plot(
        [sl_offset_x, sl_offset_x], 
        [sl_offset_y-sl_capsize/2, sl_offset_y+sl_capsize/2],
        **sl_style
    )
    ax.plot(
        [sl_offset_x+sl_length, sl_offset_x+sl_length], 
        [sl_offset_y-sl_capsize/2, sl_offset_y+sl_capsize/2],
        **sl_style
    )
    ax.text(
        sl_offset_x+sl_length/2, sl_offset_y+sl_labelmargin,
        f"{sl_length//1000} km",
        horizontalalignment='center', verticalalignment='bottom'
    )



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
