import sys
import numpy as np
import random
from statsmodels.stats.weightstats import DescrStatsW

from CMDFitter5 import CMDFitter


ext = int(sys.argv[1])

freeze = [(6,7,8),(),(6,7,8,12),(12,)]


fitter = CMDFitter(13.5,18,q_model='legendre')

fitter.freeze = np.zeros(13)
for i in freeze[ext-1]:
    fitter.freeze[i] = 1


s = np.load(f'NS5_t{ext}_samples.npy')
w = np.load(f'NS5_t{ext}_weights.npy')

ns = s.shape[0]

ind = random.choices(range(ns),k=100,weights=w.tolist())

lnp = np.empty(100)

print(fitter.freeze)
print(s.shape)

for i in range(100):
    lnp[i] = fitter.lnlikelihood(s[ind[i]])

imax = np.argmax(lnp)

p = fitter.default_params
p[np.where(fitter.freeze==0)] = s[ind[imax]]

fitter.emcee_sample(params_guess=p,prefix=f'EM5_t{ext}_',n_saves=2000)

