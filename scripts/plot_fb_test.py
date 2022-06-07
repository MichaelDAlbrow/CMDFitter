import sys
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pylab import subplots_adjust



sys.path.append(os.path.abspath(os.path.dirname(__file__))+'/../code')


from CMDFitter6 import Data, Isochrone, CMDFitter, PlotUtils

definition_file = sys.argv[1]


def load(file):
	extension = os.path.splitext(file)[1]
	if extension == '.npy':
		data = np.load(file)
	else:
		try:
			data = np.loadtxt(file)
		except ValueError:
			data = np.loadtxt(file,skiprows=1)
	return data


fitter = CMDFitter(definition_file)

fitter.freeze[6] = 1
fitter.freeze[7] = 1
fitter.freeze[8] = 1


s = load(sys.argv[2])

if len(sys.argv) == 5:
	w = load(sys.argv[3])
elif 'weighted' in sys.argv[2]:
	w = s[:,0]
	s = s[:,2:]
else:
	w = None


fig, ax = plt.subplots(2,1,figsize=(6,10))

PlotUtils.plot_q_distribution(fitter,s,w,ax=ax[0],save_figure=False)

PlotUtils.plot_fb_q(fitter,s,w,ax=ax[1],save_figure=False)

plt.savefig(sys.argv[-1])



