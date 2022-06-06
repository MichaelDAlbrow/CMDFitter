import sys
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pylab import subplots_adjust



sys.path.append(os.path.abspath(os.path.dirname(__file__))+'/../code')


from CMDFitter6 import Data, Isochrone, CMDFitter, PlotUtils

data_description = {'file' : '../data/CMD_data.txt', \
					'magnitude_min' : 14.5, \
					'magnitude_max' : 18.0, \
					'column_mag' : 0, \
					'column_blue' : 0, \
					'column_red' : 1, \
					'column_mag_err' : 3, \
					'column_blue_err' : 3, \
					'column_red_err' : 4, \
					'colour_label' : 'G - R', \
					'magnitude_label' : 'G'}

iso_description = {'file' : '../data/MIST_iso_5GYr_06Fe.txt', \
					'magnitude_min' : 13.5, \
					'magnitude_max' : 18.0, \
					'column_mag' : 30, \
					'column_blue' : 30, \
					'column_red' : 32, \
					'column_mass' : 3, \
					'magnitude_offset' : 9.55,
					'colour_offset' : 0.012}


data = Data(data_description)

iso = Isochrone(iso_description, colour_correction_data=data)

fitter = CMDFitter(data, iso)

fitter.freeze[6] = 1
fitter.freeze[7] = 1
fitter.freeze[8] = 1

s = np.load('NS_test_samples.npy')
w = np.load('NS_test_weights.npy')

#samples = np.zeros([s.shape[0],fitter.ndim])
#samples[:,np.where(fitter.freeze==0)[0]] = s

fig, ax = plt.subplots(2,1,figsize=(6,10))

PlotUtils.plot_q_distribution(fitter,s,w,ax=ax[0],save_figure=False)

PlotUtils.plot_fb_q(fitter,s,w,ax=ax[1],save_figure=False)

plt.savefig('q.png')



