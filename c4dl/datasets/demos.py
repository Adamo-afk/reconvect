from datetime import datetime, timedelta
import os
import warnings

import dask
import numpy as np
from matplotlib import colors, gridspec, pyplot as plt
from pyresample.plot import area_def2basemap
from scipy.signal import convolve


def nexrad_goes_view(nexrad_reader, goes_reader, glm_reader, ecmwf_reader, aster_reader, time):
    fig = plt.figure(figsize=(14,15))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.20, wspace=0.10)

    maxZ = nexrad_reader.variable_for_time(time, "MAXZ")
    vil = nexrad_reader.variable_for_time(time, "VIL")
    echotop40 = nexrad_reader.variable_for_time(time, "ECHOTOP-40")

    c08 = goes_reader.variable_for_time(time, "ABIC08")
    c10 = goes_reader.variable_for_time(time, "ABIC10")
    c12 = goes_reader.variable_for_time(time, "ABIC12")
    c13 = goes_reader.variable_for_time(time, "ABIC13")

    flash_density = glm_reader.variable_for_time(time, "flash_density").copy()
    (x,y) = np.mgrid[-5:5,-5:5]
    k = (x**2+y**2 < 2.5**2).astype(float)
    k /= k.sum()
    flash_density = convolve(flash_density, k, mode='valid')
    flash_density[flash_density<1e-6] = np.nan

    cape = ecmwf_reader.variable_for_time(time, "cape")
    cin = ecmwf_reader.variable_for_time(time, "cin")
    kx = ecmwf_reader.variable_for_time(time, "kx")
    z = ecmwf_reader.variable_for_time(time, "z")/9.81

    mean_elevation = aster_reader.variable_for_time(time, "mean_elevation")
    roughness = aster_reader.variable_for_time(time, "roughness")
    gradient_x = aster_reader.variable_for_time(time, "gradient_x")
    gradient_y = aster_reader.variable_for_time(time, "gradient_y")
    gradient_abs = np.sqrt(gradient_x**2+gradient_y**2)

    bmap = area_def2basemap(nexrad_reader.grid_projection.area)

    def plot(i, j, field, value_range, label):
        ax = fig.add_subplot(gs[i,j])
        im = bmap.imshow(
            field, 
            norm=None if value_range is None else colors.Normalize(*value_range),
            ax=ax,
            origin='upper'
        )
        bmap.drawstates(ax=ax)
        ax.set_xlabel(label, fontsize=14)
        fig.colorbar(im, ax=ax)
        return (ax,im)

    plot_data = iter([
        (maxZ, (0, 60), "Column Max Z"),
        (np.log10(vil), (-2.5, 3), "log10(VIL)"),
        (echotop40, (0, 10000), "Echo Top 40 dBZ"),
        (c08-c10, (-26.2, 0.6), "6.2 um - 7.3 um"),
        (c12-c13, (-43.2, 6.7), "9.6 um - 10.3 um"),
        (c08, (273.15-64.65, 273.15-29.25), "6.2 um"),
        (cape, (0, 3000), "CAPE"),
        (cin, (0, 600), "CIN"),
        (z, None, "ECMWF Elevation"),
        (mean_elevation, None, "Elevation"),
        (roughness, None, "Roughness"),
        (gradient_abs, None, "Slope"),
    ])
    
    images = []
    for i in range(4):
        images.append([])
        for j in range(3):
            (field, value_range, label) = next(plot_data)
            (ax,im) = plot(i, j, field, value_range, label)
            images[-1].append((ax,im))

    bmap.imshow(flash_density, cmap='binary', norm=colors.Normalize(0,1e-6),
        ax=images[0][0][0], origin='upper')

    return fig


def nexrad_goes_frames(fig_dir, nexrad_dir, goes_dir, area,
    start_time, end_time, interval=timedelta(minutes=5)):

    import nexrad
    import goesabi
    import projection

    @dask.delayed
    def make_frame(time):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=UserWarning)

            nexrad_reader = nexrad.NEXRADReader(nexrad_dir, ["KGWX","KHTX","KBMX"],
                projection.GridProjection(area))
            goes_reader = goesabi.GOESABIReader(goes_dir, 
                projection.GridProjection(area),
                variables=["ABIC08", "ABIC10", "ABIC12", "ABIC13"])
            try:
                fig = nexrad_goes_view(nexrad_reader, goes_reader, time)
            except FileNotFoundError:
                return
        fn = "nexrad_goes-{}.png".format(time.strftime("%Y%m%d%H%M"))
        fig.savefig(os.path.join(fig_dir, fn), bbox_inches='tight')
        plt.close(fig)

    t = start_time
    tasks = []
    while t < end_time:
        tasks.append(make_frame(t))
        t += interval

    dask.compute(tasks, scheduler='processes')
