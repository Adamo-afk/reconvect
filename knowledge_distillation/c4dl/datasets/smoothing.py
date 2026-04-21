import numpy as np

def gaussian(std=1):
    def f(x):
        return np.exp(-0.5*(x/std)**2)
    return f
    
def tophat(rad=1):
    def f(x):
        y = np.zeros_like(x)
        y[x<rad] = 1
        return y
    return f
