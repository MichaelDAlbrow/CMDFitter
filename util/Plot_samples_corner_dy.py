import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from dynesty import NestedSampler
from dynesty import plotting as dyplot

from CMDFitter5 import CMDFitter

pylab.rcParams.update({'axes.titlesize': 'x-large', 'axes.labelsize': 'x-large'})


fitter = CMDFitter(13.5,18,q_model='legendre')


samples = np.load(sys.argv[1])

if len(sys.argv) == 4:
    weights = np.load(sys.argv[2])
else:
    weights = None

output = sys.argv[-1]

print(samples.shape)

figax  = plt.subplots(13,13, figsize=(18,18))

figax2 = dyplot.cornerplot_mda(samples,weights,fig=figax,color='b',labels=fitter.labels, \
                               quantiles=None,quantiles_2d=[0.4,0.85],show_titles=False)

fig, axes = figax2

#print(axes)

for axes2 in axes:
    for ax in axes2:
        ax.locator_params(tight=True, nbins=3)


plt.savefig(output)

for i in range(samples.shape[1]):
    s = samples[:,i]
    print(i,np.min(s),np.percentile(s,[1,16,50,84,99]),np.max(s))

