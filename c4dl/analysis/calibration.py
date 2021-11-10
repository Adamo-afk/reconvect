from numba import njit
import numpy as np
from scipy.interpolate import interp1d

from ..features import batch


def calibration_curve(model, batch_gen, dataset='valid', nbins=100):
    batch_seq = batch.BatchSequence(batch_gen, dataset=dataset)
    bin_counts = np.zeros(nbins, dtype=np.uint64)
    bin_occurrences = np.zeros(nbins, dtype=np.uint64)

    for i in range(len(batch_seq)):
        print("{}/{}".format(i,len(batch_seq)))
        (X,Y) = batch_seq[i]
        Y_pred = model.predict(X)
        accumulate_hits(Y[0], Y_pred, bin_counts, bin_occurrences)

    p = np.linspace(0,1,nbins+1)
    p = 0.5 * (p[:-1] + p[1:])
    occurrence_rate = bin_occurrences/bin_counts
    return (p, occurrence_rate)


@njit
def accumulate_hits(Y, Y_pred, bin_counts, bin_occurrences):
    n = len(bin_counts)
    Y_pred = Y_pred.ravel()
    Y = Y.ravel()
    
    for i in range(Y.shape[0]):
        bin_ind = int(Y_pred[i]*n)
        if bin_ind == n:
            bin_ind = n-1
        
        bin_counts[bin_ind] += 1
        if bool(Y[i]):
            bin_occurrences[bin_ind] += 1


def calibration_func(p, occurrence_rate):
    valid = np.isfinite(occurrence_rate)
    p = p[valid]
    occurrence_rate = occurrence_rate[valid]

    if p[0] != 0:
        p = np.hstack((0, p))
        occurrence_rate = np.hstack((occurrence_rate[0], occurrence_rate))
    if p[-1] != 1:
        p = np.hstack((p, 1))
        occurrence_rate = np.hstack((occurrence_rate, occurrence_rate[-1]))

    func = interp1d(p, occurrence_rate, kind='cubic')
    return func
    

def intersection_over_union(model, batch_gen, dataset='valid', threshold=0.5):
    batch_seq = batch.BatchSequence(batch_gen, dataset=dataset)
    tp = fp = fn = 0

    for i in range(len(batch_seq)):
        print("{}/{}".format(i,len(batch_seq)))
        (X,Y) = batch_seq[i]
        Y_pred = model.predict(X)
        Y_pred = (Y_pred >= threshold)
        Y = Y[0].astype(bool)
        tp += np.count_nonzero(Y_pred & Y)        
        fp += np.count_nonzero(Y_pred & ~Y)
        fn += np.count_nonzero(~Y_pred & Y)
        
    return tp / (tp+fp+fn)


def best_iou(model, batch_gen, dataset='valid')
    dtresh = 0.1
    min_dtresh = 0.01
    tresh = np.arange(0.1, 0.91, dtresh)    
    iou = np.zeros_like(x)

    # initial search
    for (i,t) in enumerate(tresh):
        iou[i] = intersection_over_union(model, batch_gen, dataset=dataset, threshold=t)

    dtresh /= 2
    while dtresh >= min_dtresh:
        ind_opt = np.argmax(iou)
        t_opt = tresh[ind_opt]
        print("Optimal treshold: {}, IoU: {}".format(t[ind_out]))

        t_new = [t_opt-dtresh, t_opt+dtresh]
        iou_new = [
            intersection_over_union(model, batch_gen, dataset=dataset, threshold=t)
            for t in t_new
        ]
        tresh = np.hstack(tresh[:ind_opt], t_new[0], tresh[ind_opt], t_new[1], tresh[ind_opt+1:])
        iou = np.hstack(iou[:ind_opt], iou_new[0], iou[ind_opt], iou_new[1], iou[ind_opt+1:])
        dtresh /= 2

    return (tresh, iou)
