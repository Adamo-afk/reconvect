import os

from matplotlib import pyplot as plt
import numpy as np

from c4dl.visualization import plots

loss_names = {
    "BFL2": "FL $\gamma=2$",
    "BFL1": "FL $\gamma=1$",
    "WFL2": "Weighted FL $\gamma=2$",
    "WFL1": "Weighted FL $\gamma=1$",
    "WCE": "Weighted CE",
    "BCE": "CE",
    "IOU": "IoU",
}

loss_calibration_files = {
    "BFL2": "calibration-lightning_dropout_weightdecay_noclassweight.npy",
    "BFL1": "calibration-lightning_dropout_weightdecay_noclassweight_gamma1.npy",
    "WFL2": "calibration-lightning_baseline1.npy",
    "WFL1": "calibration-lightning_gamma1.npy",
    "WCE": "calibration-lightning_wce.npy",
    "BCE": "calibration-lightning_bce.npy",
    "IOU": "calibration-lightning_iou.npy",
}

loss_conf_matrix_files = {
    "BFL2": "conf_matrix-lightning_dropout_weightdecay_noclassweight.npy",
    "BFL1": "conf_matrix-lightning_dropout_weightdecay_noclassweight_gamma1.npy",
    "WFL2": "conf_matrix-lightning_baseline1.npy",
    "WFL1": "conf_matrix-lightning_gamma1.npy",
    "WCE": "conf_matrix-lightning_wce.npy",
    "BCE": "conf_matrix-lightning_bce.npy",
    "IOU": "conf_matrix-lightning_iou.npy",
}

def calibration_by_loss(out_file=None):
    occurrence_rate = {
        k: np.load(os.path.join("../results", fn))
        for (k,fn) in loss_calibration_files.items()
    }
    nbins = len(list(occurrence_rate.values())[0])
    p = np.linspace(0,1,nbins+1)
    p = 0.5 * (p[:-1] + p[1:])

    fig = plots.plot_calibration(p, occurrence_rate, loss_names)

    if out_file is not None:
        fig.savefig(out_file, bbox_inches='tight')
        plt.close(fig)

def pr_curve_by_loss(curve="PR", out_file=None):
    conf_matrix = {
        k: np.load(os.path.join("../results", fn))
        for (k,fn) in loss_conf_matrix_files.items()
    }

    if curve == "PR":
        fig = plots.plot_pr_curve(conf_matrix, loss_names)
    elif curve == "ROC":
        fig = plots.plot_roc_curve(conf_matrix, loss_names)

    if out_file is not None:
        fig.savefig(out_file, bbox_inches='tight')
        plt.close(fig)

def metric_curve_by_loss(metric="CSI", out_file=None):
    conf_matrix = {
        k: np.load(os.path.join("../results", fn))
        for (k,fn) in loss_conf_matrix_files.items()
    }

    thresholds = np.arange(0, 1.0001, 0.001)
    fig = plots.plot_threshold_metric_curve(thresholds,
        conf_matrix, loss_names, metric)

    if out_file is not None:
        fig.savefig(out_file, bbox_inches='tight')
        plt.close(fig)
