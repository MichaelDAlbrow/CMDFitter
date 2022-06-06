import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from dynesty import NestedSampler
from dynesty import plotting as dyplot

sys.path.append(os.path.abspath(os.path.dirname(__file__))+'/../code')
from CMDFitter6 import Data, Isochrone, CMDFitter

pylab.rcParams.update({'axes.titlesize': 'x-large', 'axes.labelsize': 'x-large'})


samples = np.load(sys.argv[1])

if len(sys.argv) == 4:
    weights = np.load(sys.argv[2])
else:
    weights = None

output = sys.argv[-1]

print(samples.shape)
n = samples.shape[1]

fitter = CMDFitter()
fitter.freeze[6] = 1
fitter.freeze[7] = 1
fitter.freeze[8] = 1

labels = [fitter.labels[i] for i in range(fitter.ndim) if fitter.freeze[i] == 0]

figax  = plt.subplots(n,n, figsize=(18,18))

figax2 = dyplot.cornerplot_mda(samples,weights,fig=figax,color='b',quantiles=None,quantiles_2d=[0.4,0.85],show_titles=False,labels=labels)

fig, axes = figax2

#print(axes)

for axes2 in axes:
    for ax in axes2:
        ax.locator_params(tight=True, nbins=3)


plt.savefig(output)

for i in range(samples.shape[1]):
    s = samples[:,i]
    print(i,labels[i],np.min(s),np.percentile(s,[1,16,50,84,99]),np.max(s))

