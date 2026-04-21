from datetime import datetime, timedelta
import os
import string

from matplotlib import colors, gridspec, lines, patches, pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
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


def plot_loss_leadtime(loss_lt, var_names, model_names, 
    colors_linestyles=None, fig=None, legend=True, dt_minutes=5
):   
    num_vars = loss_lt.shape[0]
    num_models = loss_lt.shape[1]
    leadtime = np.arange(1,loss_lt.shape[2]+1) * dt_minutes
    if fig is None:
        fig = plt.figure(figsize=(6,3*num_vars+1))
    
    for i in range(num_vars):
        ax = fig.add_subplot(num_vars, 1, i+1)

        for (j, model_name) in enumerate(model_names):
            if colors_linestyles is not None:
                (c, ls) = colors_linestyles[model_name]
            else:
                c = ls = None
            ax.plot(leadtime, loss_lt[i,j,:],
                label=model_name, color=c, linestyle=ls)

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

        if i == num_vars-1:
            ax.set_xlabel("Lead time [min]")
            ax.legend(loc='lower right')
        ax.set_ylabel(var_names[i])

    return fig


def plot_frame(ax, frame, norm=None):
    im = ax.imshow(frame.astype(np.float32), norm=norm)
    ax.tick_params(left=False, bottom=False,
        labelleft=False, labelbottom=False)
    return im


def plot_model_anim(X, Y, out_dir,
    shown_input=25, batch_member=0, interval_mins=5, 
    min_p=0.025):

    X = X[shown_input][batch_member,...,0]
    Y = Y[batch_member,...,0]
    norm = colors.LogNorm(min_p,1,clip=True)


    frame_ind = 0
    def save_frame(x):
        nonlocal frame_ind
        fig = plt.figure()
        ax = fig.add_subplot()
        plot_frame(ax, x, norm=norm)
        t_min = (frame_ind-X.shape[0]+1) * interval_mins
        ax.set_title(f"$t={t_min}\\ \\mathrm{{min}}$")
        path = os.path.join(out_dir, f"frame{frame_ind:02d}.png")
        fig.savefig(path, bbox_inches='tight', dpi=200)
        plt.close(fig)
        frame_ind += 1

    for i in range(X.shape[0]):
        save_frame(X[i,:,:])    
    for i in range(Y.shape[0]):
        save_frame(Y[i,:,:])


transform_TB = lambda x: x*10+250
transform_radiance = lambda x: x*100
transform_T = lambda x: x*7.2+290
input_transforms = {
    "Rain rate": lambda x: 10**(x*0.528-0.051),
    "CZC": lambda x: x*8.71+21.3,
    "EZC-20": lambda x: x*1.97,
    "EZC-45": lambda x: x*1.97,
    "HZC": lambda x: x*1.97,
    "LZC": lambda x: 10**(x*0.135-0.274),
    "Lightning": lambda x: x,
    "Light. dens.": lambda x: 10**(x*0.640-0.593),
    "Current dens.": lambda x: 10**(x*0.731-0.0718),
    "POH": lambda x: x,
    "$R > 10\\mathrm{mm\\,h^{-1}}$": lambda x: x,
    "HRV": lambda x: x*100,
    "CTH": lambda x: x*2.810+5.260,
    "CAPE-MU": lambda x: x*0.2,
    "CIN-MU": lambda x: x*21,
    "LCL": lambda x: x*1000,
    "MCONV": lambda x: x*3.8,
    "HZEROCL": lambda x: x*3300,
    "OMEGA": lambda x: x*4.2,
    "SLI": lambda x: x*3.5,
    "T-SO": transform_T,
    "T-2M": transform_T,
    "VIS006": transform_radiance,
    "VIS008": transform_radiance,
    "HRV": transform_radiance,
    "IR-016": transform_radiance,
    "IR-039": lambda x: x*17.5+274,
    "WV-062": transform_TB,
    "WV-073": transform_TB,
    "IR-087": transform_TB,
    "IR-097": transform_TB,
    "IR-108": transform_TB,
    "IR-120": transform_TB,
    "IR-134": transform_TB,
    "CTT": lambda x: x*19.1+260,
    "Altitude": lambda x: x * 280,
    "El. EW-deriv.": lambda x: x * 200,
    "El. NS-deriv.": lambda x: x * 200,
    "Solar zen. ang.": lambda x: x * 127
}
input_norm = {
    "Rain rate": colors.LogNorm(0.01, 100, clip=True),
    "LZC": colors.LogNorm(0.75, 100, clip=True),
    "Light. dens.": colors.LogNorm(0.01, 100, clip=True),
    "Current dens.": colors.LogNorm(0.01, 100, clip=True),
    "Lightning": colors.Normalize(0, 1),
    "POH": colors.Normalize(0, 1),
    "$R > 10\\mathrm{mm\\,h^{-1}}$": colors.Normalize(0, 1),
    "HRV": colors.Normalize(0,100),
    "CTH": colors.Normalize(0,12),
    "CAPE-MU": colors.Normalize(0,2)
}
input_ticks = {
    "Rain rate": [0.1, 1, 10, 100],
    "Lightning": [0, 0.5, 1],
    "POH": [0, 0.5, 1],
    "$R > 10\\mathrm{mm\\,h^{-1}}$": [0, 0.5, 1],
    "HRV": [0, 25, 50, 75],
    "CTH": [0, 5, 10],
    "CAPE-MU": [0.5, 1, 1.5, 2],
}


def plot_model_examples(X, Y, models, shown_inputs=(0,25,12,9),
    input_timesteps=(-4,-1), output_timesteps=(0,2,5,11),
    batch_member=0, interval_mins=5,
    input_names=("Rain rate", "Lightning", "HRV", "CTH"),
    future_input_names=("CAPE-MU",),
    min_p=0.025, plot_scale=256
):
    num_timesteps = len(input_timesteps)+len(output_timesteps)
    gs_rows = 2 * max(len(models),len(shown_inputs))
    gs_cols = num_timesteps
    width_ratios = (
        [0.1, 0.19] +
        [1]*len(input_timesteps) +
        [0.1] +
        [1]*len(output_timesteps) +
        [0.19, 0.1]
    )
    gs = gridspec.GridSpec(gs_rows, gs_cols+5, wspace=0.02, hspace=0.05,
        width_ratios=width_ratios)
    batch = [x[batch_member:batch_member+1,...] for x in X]
    obs = [y[batch_member:batch_member+1,...] for y in Y]

    fig = plt.figure(figsize=(gs_cols*1.5, gs_rows/2*1.5))

    # plot inputs
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
    row0 = 0
    future_input_ind = 0
    norm_log = colors.LogNorm(min_p,1,clip=True)
    for (i,model) in enumerate(models):
        if model == "obs":
            Y_pred = obs[0]
            norm = norm_log
            label = "Observed"
        elif isinstance(model, str) and model.startswith("input-future"):
            var_ind = int(model.split("-")[-1])
            Y_pred = batch[var_ind]
            input_name = future_input_names[future_input_ind]
            Y_pred = input_transforms[input_name](Y_pred)
            norm = input_norm[input_name]
            future_input_ind += 1
            label = input_name
        else:
            Y_pred = model.predict(batch)
            norm = norm_log
            label = "Forecast"
        row = row0 + 2*i
        op = Y_pred[0,output_timesteps,:,:,0]        
        for m in range(len(output_timesteps)):
            col = m + len(input_timesteps) + 3
            ax = fig.add_subplot(gs[row:row+2,col])
            im = plot_frame(ax, op[m,:,:], norm=norm)
            if i==0:
                iv = (output_timesteps[m]+1) * interval_mins
                ax.set_title(f"$+{iv}\\,\\mathrm{{min}}$")
            if m == len(output_timesteps)-1:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(label)
                if i == len(models)-1:
                    scalebar = AnchoredSizeBar(ax.transData,
                           op.shape[1],
                           f'{plot_scale} km',
                           'lower center', 
                           pad=0.1,
                           color='black',
                           frameon=False,
                           size_vertical=1,
                           bbox_transform=ax.transAxes,
                           bbox_to_anchor=(0.5,-0.27)
                    )
                    ax.add_artist(scalebar)

        if i==len(models)-1:
            r0 = row0 + 2*len(future_input_names)
            r1 = r0 + 4
            cax = fig.add_subplot(gs[r0:r1,-1])            
            cb = plt.colorbar(im, cax=cax)
            cb.set_ticks([min_p, 0.05, 0.1, 0.2, 0.5, 1])
            cb.set_ticklabels([min_p, 0.05, 0.1, 0.2, 0.5, 1])
            cax.set_xlabel("$p$", fontsize=12)
        elif i<len(future_input_names):
            cax = fig.add_subplot(gs[row:row+2,-1])            
            cb = plt.colorbar(im, cax=cax)
            cb.set_ticks(input_ticks[input_name])

    return fig


def plot_data_examples(X, names, columns=8):
    rows = len(names) // columns
    if len(names) % columns:
        rows += 1
    gs_rows = rows * 3
    height_ratios = rows * [1,0.15,0.35]
    fig = plt.figure(figsize=(columns*1.5, rows*1.5*1.5))
    gs = gridspec.GridSpec(gs_rows, columns, wspace=0.3, hspace=0.05,
        height_ratios=height_ratios)

    for (k,name) in enumerate(names):
        row = k // columns
        col = k % columns

        ax = fig.add_subplot(gs[row*3,col])
        transform = input_transforms.get(name, lambda x: x)
        img_data = transform(X[k])
        norm = input_norm.get(name, None)
        im = plot_frame(ax, img_data, norm=norm)
        ax.set_title(name)

        cax = fig.add_subplot(gs[row*3+1,col])  
        cb = plt.colorbar(im, cax=cax, orientation='horizontal')
        if name in input_ticks:
            cb.set_ticks(input_ticks[name])

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
    img[:,:,3] = mask * 0.5
    
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
    "r": "tab:blue",
    "n": "tab:orange",
    "s": "tab:green",
    "l": "tab:purple",
    "d": "tab:brown"
}
def get_source_color(source):
    return source_colors.get(source, "tab:gray")


source_names = {
    "r": "Rad",
    "n": "NWP",
    "s": "Sat",
    "l": "Lig",
    "aster": "DEM",
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

notation = {
    "r": "Rad",
    "l": "Lig",
    "s": "Sat",        
    "n": "NWP",
    "d": "DEM"
}
prefix_notation = {
    "lightning": "Lightning",
    "hail": "Hail",
    "rain": "Precipitation"
}

def exclusion_plot(metrics, metrics_names, fig=None, axes=None,
    variable_names=None, subplot_index=0, significant_digits=3):

    import seaborn as sns


    metric_notation = {
        "binary": "Error rate",
        "cross_entropy": "CE",
        "mae": "MAE",
        "rmse": "RMSE",
        "FL2": "FL $\\gamma=2$"
    }

    prefixes_names = [prefix_notation[k] for k in metrics]
    metrics_tables = {prefix: np.full((8,4), np.nan) for prefix in metrics}
    metric_pos = {
        frozenset(("n", "l", "d", "r", "s")): (0,0),
        frozenset(("n", "l", "d", "r")): (0,1),
        frozenset(("n", "l", "d", "s")): (0,2),
        frozenset(("n", "l", "d")): (0,3),

        frozenset(("n", "l", "r", "s")): (1,0),
        frozenset(("n", "l", "r")): (1,1),
        frozenset(("n", "l", "s")): (1,2),
        frozenset(("n", "l")): (1,3),

        frozenset(("n", "d", "r", "s")): (2,0),
        frozenset(("n", "d", "r")): (2,1),
        frozenset(("n", "d", "s")): (2,2),
        frozenset(("n", "d")): (2,3),

        frozenset(("l", "d", "r", "s")): (3,0),
        frozenset(("l", "d", "r")): (3,1),
        frozenset(("l", "d", "s")): (3,2),
        frozenset(("l", "d")): (3,3),

        frozenset(("n", "r", "s")): (4,0),
        frozenset(("n", "r")): (4,1),
        frozenset(("n", "s")): (4,2),
        frozenset(("n",)): (4,3),

        frozenset(("l", "r", "s")): (5,0),
        frozenset(("l", "r")): (5,1),
        frozenset(("l", "s")): (5,2),
        frozenset(("l",)): (5,3),

        frozenset(("d", "r", "s")): (6,0),
        frozenset(("d", "r")): (6,1),
        frozenset(("d", "s")): (6,2),
        frozenset(("d",)): (6,3),

        frozenset(("r", "s")): (7,0),
        frozenset(("r",)): (7,1),
        frozenset(("s",)): (7,2),
        frozenset(()): (7,3),
    }
    metric_pos_inv = {v: k for (k, v) in metric_pos.items()}

    for prefix in metrics:
        for subset in metrics[prefix]:
            subset_frozen = frozenset(subset)
            (i,j) = metric_pos[subset_frozen]
            metrics_tables[prefix][i,j] = metrics[prefix][subset]

    xlabels_show = frozenset(("r", "s"))
    ylabels_show = frozenset(("n", "l", "d"))

    with sns.plotting_context("paper"):
        if fig is None:
            fig = plt.figure(figsize=(3.125*len(metrics),7.5))
        
        for (i,prefix) in enumerate(metrics):
            xlabels = [
                "\n".join(sorted(notation[s] for s in metric_pos_inv[0,i] & xlabels_show))
                for i in range(metrics_tables[prefix].shape[1])
            ]
            ylabels = [
                "\n".join(sorted(notation[s] for s in metric_pos_inv[i,0] & ylabels_show))
                for i in range(metrics_tables[prefix].shape[0])
            ]

            ax = axes[i] if (axes is not None) else fig.add_subplot(1,len(metrics),i+1)
            heatmap = sns.heatmap(
                metrics_tables[prefix],
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
                prefixes_names[i]+" " if prefixes_names[i] else "",
                metric_notation[metrics_names[i]]+" " if metrics_names[i] else "",
            ))
            ax.tick_params(axis='both', bottom=False, left=False,
                labelleft=(i+subplot_index==0))

    return fig


def shapley_by_time(
        leadtimes,
        shapley_values,
        interval=timedelta(minutes=5),
        fig=None,
        ax=None,
        legend=True,
    ):

    interval_mins = interval.total_seconds() / 60
    leadtimes = leadtimes * interval_mins
    
    if ax is None:
        fig = plt.figure(figsize=(6,3))
        ax = fig.add_subplot()

    val_sum = None
    for values in shapley_values.values():
        if val_sum is None:
            val_sum = values.copy()
        else:
            val_sum += values

    for (source, values) in shapley_values.items():
        ax.plot(
            leadtimes, values/val_sum, linewidth=1,
            label=notation[source], c=source_colors[source]
        )
    if legend:
        ax.legend()
    ax.set_xlim((0, leadtimes[-1]))
    ax.set_xlabel("Lead time [min]")
    ax.set_ylabel("Normalized Shapley value")

    return fig

def shapley_values_full_legend(shapley_values_full, ax):
    val_sum_full = sum(shapley_values_full.values())
    labels = [
        f"{notation[s]}: {shapley_values_full[s]/val_sum_full:.03f}"
        for s in shapley_values_full
    ]
    custom_lines = [
        lines.Line2D([0], [0], color=source_colors[s], lw=1)
        for s in shapley_values_full
    ]
    ax.legend(custom_lines, labels, ncol=3, mode="expand")


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
