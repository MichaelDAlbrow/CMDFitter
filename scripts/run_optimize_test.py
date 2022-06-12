import sys
import os
import numpy as np
from scipy.optimize import minimize

sys.path.append(os.path.abspath(os.path.dirname(__file__))+'/../code')


from CMDFitter6 import CMDFitter

definition_file = sys.argv[1]

fitter = CMDFitter(definition_file)

fitter.freeze[6] = 1
fitter.freeze[7] = 1
fitter.freeze[8] = 1

params = fitter.default_params.copy()

p = params[fitter.freeze==0]


result = minimize(fitter.neglnprob,p,method='Nelder-Mead',options={'maxiter': 50000, 'maxfev': 50000})

print(result)
print(result.success)
print(result.x)

p_opt = result.x

print('maximum prob')
print('p = ', p)
print('lnp = ',fitter.lnprob(p_opt))
print('lnl = ',fitter.lnlikelihood(p_opt))

np.save(f'opt_prob.npy',p_opt)




