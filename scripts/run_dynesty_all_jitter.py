

import sys
import os

import numpy as np

from scipy.optimize import minimize
from statsmodels.stats.weightstats import DescrStatsW

from itertools import chain, combinations

sys.path.append(os.path.abspath(os.path.dirname(__file__))+'/../code')


from CMDFitter import CMDFitter
from PlotUtils import PlotUtils

print(sys.argv)


definition_file = sys.argv[1]


ext = int(sys.argv[2])

q_model = sys.argv[3]

assert q_model in ['legendre','power','piecewise','hist']


if q_model == 'legendre':
	freeze_groups = [[0,1],[3],[4],[8,9,10],[12],[15]]

if q_model == 'power':
	freeze_groups = [[0,1],[3],[4],[5,6],[7],[11],[13],[14]]

if q_model in ['piecewise','hist']:
	freeze_groups = [[0,1],[3],[4],[10],[13]]



n_groups = len(freeze_groups)


freeze_combinations = chain.from_iterable(combinations(freeze_groups, r) for r in range(n_groups+1))

freeze = []
for f in freeze_combinations:
	freeze.append(sum([g for g in f],[]) )

fitter = CMDFitter(definition_file,m_model='power',q_model=q_model,iso_correction_type='magnitude')


fitter.freeze = np.zeros(fitter.ndim)
for i in freeze[ext]:
	fitter.freeze[i] = 1

if fitter.freeze[0] == 1:
	fitter.default_params[0] = 4.10
	fitter.default_params[1] = 0.0

print(fitter.freeze)
print(fitter.labels)


if not os.path.exists(f'DY_j{ext}_samples.npy'):
	fitter.dynesty_sample(prefix=f'DY_j{ext}_',jitter=True,nlive=400)


s = np.load(f'DY_j{ext}_samples.npy')
w = np.load(f'DY_j{ext}_weights.npy')


#
#  Optimize solution
#

if not os.path.exists(f'DY_j{ext}_prob.npy'):

	res = np.zeros(s.shape[1])

	for param in range(s.shape[1]):

		wq = DescrStatsW(data=s[:,param],weights=w)
		result = wq.quantile(probs=0.5, return_pandas=False)
		res[param] = result[0]

	result = minimize(fitter.neglnprob,res,method='Nelder-Mead',options={'maxiter': 500000, 'maxfev': 500000})

	print(f'ext: {ext}')
	print(result)
	print(result.success)
	print(result.x)

	p_opt = result.x
	np.save(f'DY_j{ext}_prob.npy',p_opt)

	print('maximum prob')
	print('p = ', p_opt)
	print('lnp = ',fitter.lnprob(p_opt))
	print('lnl = ',fitter.lnlikelihood(p_opt))
	print()

	with open(f'DY_j{ext}_prob.txt','w') as file:
		file.write(f'p = {p_opt}\n')
		file.write(f'lnp = {fitter.lnprob(p_opt)}\n')
		file.write(f'lnl = {fitter.lnlikelihood(p_opt)}\n')


#
#  Now make some summary figures
#
map_sol = np.load(f'DY_j{ext}_prob.npy')

PlotUtils.plot_mass_function(fitter,s,weights=w,map_sol=map_sol,plot_file=f'DY_j{ext}_mf.png')

PlotUtils.plot_q_distribution(fitter,s,weights=w,map_sol=map_sol,plot_file=f'DY_j{ext}_qdist.png')

PlotUtils.plot_prior_q_distribution(fitter,plot_file=f'DY_j{ext}_qdist_prior.png')


#
# Summary figures from optimized solution
#

params = fitter.default_params.copy()

params[np.where(fitter.freeze==0)[0]] = map_sol

PlotUtils.plot_realisation(fitter,params,plot_file=f'DY_j{ext}_realisation_ns.png',scatter=False,outliers=False)
PlotUtils.plot_realisation(fitter,params,plot_file=f'DY_j{ext}_realisation.png',outliers=False)

PlotUtils.plot_fb_q(fitter,s,weights=w,plot_file=f'DY_j{ext}_fbq.png')








