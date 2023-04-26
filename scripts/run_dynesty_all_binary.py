

import sys
import os

import numpy as np

from scipy.optimize import minimize
from statsmodels.stats.weightstats import DescrStatsW

from itertools import chain, combinations

sys.path.append(os.path.abspath(os.path.dirname(__file__))+'/../code')


from CMDFitter_triple_flatprior import CMDFitter
from PlotUtils import PlotUtils

import cProfile


#
#  Command line arguments
#

print(sys.argv)

definition_file = sys.argv[1]

ext = int(sys.argv[2])

q_model = sys.argv[3]

assert q_model in ['quadratic','legendre','power','single_power','piecewise','hist']


#
# Groups of parameters we might want to freeze
#

q_min = 0.0

if q_model == 'legendre':
	freeze_groups = [[0,1],[3],[4],[8,9,10],[12],[15]]

if q_model == 'power':
	freeze_groups = [[5,7,8,11,12],[0,1],[3],[4],[5,6],[7],[11],[14],[15]]

if q_model == 'single_power':
	freeze_groups = [[0,1],[3],[4],[7,8],[10],[12],[14]]

if q_model == 'quadratic':
	freeze_groups = [[0,1],[3],[4],[7],[8],[17],[19]]

if q_model =='piecewise':
	freeze_groups = [[0,1],[3],[4],[10],[14],[15]]
	q_min = 0.4

if q_model == 'hist':
	freeze_groups = [[9,10,11,12],[14],[21],[22],[23]]
	q_min = 0.4

#
#  Construct a particular set of frozen parameters
#

n_groups = len(freeze_groups)

freeze_combinations = chain.from_iterable(combinations(freeze_groups, r) for r in range(n_groups+1))

print('Freeze combinations')
for i, f in enumerate(freeze_combinations):
	print(i,f)
print()

# Need to reset iterator
freeze_combinations = chain.from_iterable(combinations(freeze_groups, r) for r in range(n_groups+1))

freeze = []
for f in freeze_combinations:
	freeze.append(sum([g for g in f],[]) )


#
#  Set up the fitter
#

fitter = CMDFitter(definition_file,m_model='power',q_model=q_model,iso_correction_type='colour',q_min=q_min,outlier_scale=2.0)

fitter.freeze = np.zeros(fitter.ndim)
for i in freeze[ext]:
	fitter.freeze[i] = 1

# Mass distribution parameters for no cutoff
if fitter.freeze[0] == 1:
	fitter.default_params[0] = 4.10
	fitter.default_params[1] = 0.0

# Freeze triples and outlier covariance
for i in range(6):
	fitter.freeze[fitter.b_index+2+i] = 1


# For binaries with one-sided power law
if q_model == 'power' and fitter.freeze[7] == 1:
	fitter.default_params[5] = 1.101
	fitter.default_params[7] = 0.001
	fitter.default_params[8] = 0.0
	fitter.default_params[11] = 0.0
	fitter.default_params[12] = 0.0


print(fitter.freeze)
print(fitter.labels)

# Maximum outlier fraction
fitter.scale_fo = 0.15


#
#  Run the fitter
#

if not os.path.exists(f'DY_j{ext}_samples.npy'):
	pr = cProfile.Profile()
	pr.enable()
	fitter.dynesty_sample(prefix=f'DY_j{ext}_',jitter=True,nlive=400)
	pr.disable()
	pr.print_stats(sort='cumtime')

#
#  Load output samples
#

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

PlotUtils.plot_realisation(fitter,params,plot_file=f'DY_j{ext}_realisation_ns.png',scatter=False,outliers=False,outlier_distribution=True)
PlotUtils.plot_realisation(fitter,params,plot_file=f'DY_j{ext}_realisation.png',outliers=False,outlier_distribution=True)

PlotUtils.plot_fb_q(fitter,s,weights=w,plot_file=f'DY_j{ext}_fbq.png')








