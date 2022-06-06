import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__))+'/../code')


from CMDFitter6 import Data, Isochrone, CMDFitter

data_description = {'file' : 'data/CMD_data.txt', \
					'magnitude_min' : 13.5, \
					'magnitude_max' : 18.0, \
					'column_mag' : 0, \
					'column_blue' : 0, \
					'column_red' : 1, \
					'column_mag_err' : 3, \
					'column_blue_err' : 3, \
					'column_red_err' : 4, \
					'colour_label' : 'G - R', \
					'magnitude_label' : 'G'}

iso_description = {'file' : 'data/MIST_iso_5GYr_06Fe.txt', \
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

p = fitter.default_params

p[0] = 2.0
p[1] = 0.6
p[2] = 0.0
p[9] = 0.3
p[10] = 0.001
p[11] = 1.0
p[12] = 0.0

ll = fitter.lnlikelihood(p[fitter.freeze == 0])

print(ll)







