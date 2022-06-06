import sys
import os
import numpy as np
from scipy.optimize import minimize

sys.path.append(os.path.abspath(os.path.dirname(__file__))+'/../code')


from CMDFitter6 import Data, Isochrone, CMDFitter

data_description = {'file' : 'data/CMD_data.txt', \
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
#iso = Isochrone(iso_description, colour_correction_data=None)

fitter = CMDFitter(data, iso)

fitter.freeze[6] = 1
fitter.freeze[7] = 1
fitter.freeze[8] = 1

params = np.array([2.2,0.55,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5,0.01,1.0,0.00])
#params = np.array([2.33,0.55,0.04,0.0,0.4,0.07,0.0,0.0,0.0,0.56,0.0021,0.78,0.22])

p = params[fitter.freeze==0]

print('starting values')
print('p = ', p)
print('lnp = ',fitter.lnprob(p))
print('lnl = ',fitter.lnlikelihood(p))

result = minimize(fitter.neglnlikelihood,p,method='Nelder-Mead',options={'maxiter': 50000, 'maxfev': 50000})

print(result)
print(result.success)
print(result.x)

p_opt = result.x

print('maximum likelihood')
print('p = ', p)
print('lnp = ',fitter.lnprob(p_opt))
print('lnl = ',fitter.lnlikelihood(p_opt))

result = minimize(fitter.neglnprob,p,method='Nelder-Mead',options={'maxiter': 50000, 'maxfev': 50000})

print(result)
print(result.success)
print(result.x)

p_opt = result.x

print('maximum prob')
print('p = ', p)
print('lnp = ',fitter.lnprob(p_opt))
print('lnl = ',fitter.lnlikelihood(p_opt))

np.save(f'opt_l_fr.npy',p_opt)




