import numpy as np


def grid_accumulate(i, j, grid, weights=None, weighted_grid=None):
    i0 = np.floor(i).astype(int)
    i1 = i0+1
    j0 = np.floor(j).astype(int)
    j1 = j0+1
    valid = (i0 >= 0) & (i1 < grid.shape[0]) & \
        (j0 >= 0) & (j1 < grid.shape[1])
    if not valid.any():
        return

    i0 = i0[valid]
    i1 = i1[valid]
    j0 = j0[valid]
    j1 = j1[valid]
    di = i[valid]-i0
    dj = j[valid]-j0
    w = np.empty((len(di),2,2))
    w[:,0,0] = (1-di)*(1-dj)
    w[:,0,1] = (1-di)*dj
    w[:,1,0] = di*(1-dj)
    w[:,1,1] = di*dj

    for (li0,li1,lj0,lj1,lw) in zip(i0,i1,j0,j1,w):
        grid[li0:li1+1,lj0:lj1+1] += lw
        
    if weights is not None:
        w *= weights[valid,None,None]
        for (li0,li1,lj0,lj1,lw) in zip(i0,i1,j0,j1,w):
            weighted_grid[li0:li1+1,lj0:lj1+1] += lw
