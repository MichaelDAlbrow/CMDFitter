

import sys
import os

import numpy as np

import pprint
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from statsmodels.stats.weightstats import DescrStatsW

from itertools import chain, combinations

sys.path.append(os.path.abspath(os.path.dirname(__file__))+'/../code')


from CMDFitter_triple_flatprior import CMDFitter
from PlotUtils import PlotUtils

from dynesty import utils as dyfunc


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
pdict = None

q_min_plot = 0.4
q_min = 0.0

if q_model == 'legendre':
	n_legendre = 5
	pdict = {'n_legendre': n_legendre}
	freeze_groups = [[0,1],[3],[4],[11,12,13,14,15],[17],[26]]
	q_min = 0.0

if q_model == 'power':
	freeze_groups = [[0,1],[7],[11],[20]]

if q_model == 'single_power':
	freeze_groups = [[0,1],[3],[4],[7,8],[10],q[19]]

if q_model == 'quadratic':
	freeze_groups = [[0,1],[3],[4],[7],[8],[17],[19]]

if q_model =='piecewise':
	freeze_groups = [[0,1],[3],[4],[10],[14],[15]]
	q_min = 0.4

if q_model == 'hist':
	n_hist_bins = 10
	pdict = {'n_q_hist_bins': n_hist_bins}
	freeze_groups = [[0,1],[2],[3],[4],[i + 5 + n_hist_bins for i in range(n_hist_bins)],[6+2*n_hist_bins],[13+2*n_hist_bins],[14+2*n_hist_bins],[15+2*n_hist_bins]]

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
fitter = CMDFitter(definition_file,m_model='power',q_model=q_model,iso_correction_type='magnitude',q_min=q_min,outlier_scale=2.0,parameters_dict=pdict)

print(fitter.labels)
print(fitter.default_params)
print(fitter.ndim)

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
	#pr = cProfile.Profile()
	#pr.enable()
	#fitter.dynesty_sample(prefix=f'DY_j{ext}_',jitter=True,nlive=400,bound='multi')
	fitter.dynesty_sample(prefix=f'DY_j{ext}_',jitter=True,bound='multi',sample='rwalk',n_parallel=1,nlive=500)
	#pr.disable()
	#pr.print_stats(sort='cumtime')

#
#  Load output samples
#

s = np.load(f'DY_j{ext}_samples.npy')
w = np.load(f'DY_j{ext}_weights.npy')

#
# Equally-weighted samples
#
rstate = np.random.Generator(np.random.PCG64())
ws = dyfunc.resample_equal(s,w,rstate=rstate)


#
# Calculate the maximum likelihood sample
#


s_lnl = np.load(f'DY_j{ext}_samples_lnl.npy')
s_lnprior = np.zeros_like(s_lnl)

for j,sample in enumerate(s):
 	s_lnprior[j] = fitter.ln_prior(sample)

s_lnp = s_lnl + s_lnprior

p_opt = s[np.argmax(s_lnp[:,0])]


if q_model in ['hist']:

	np.save(f'DY_j{ext}_prob.npy',p_opt)

else:

	#
	#  Optimize solution
	#

	if not os.path.exists(f'DY_j{ext}_prob.npy'):

		res = np.zeros(s.shape[1])
		for param in range(s.shape[1]):

			res[param] = dyfunc.quantile(s[:,param],[0.5],weights=w)[0]

		mean, cov = dyfunc.mean_and_cov(s, w)
		fitter.emcee_walker_dispersion = np.sqrt(np.diag(cov))*0.001

		print('maximum probability sample',p_opt)
		print('lnp at maximum = ',fitter.lnprob(p_opt))
		print('lnl at maximum = ',fitter.lnlikelihood(p_opt))
		print('ln_prior at maximum = ',fitter.ln_prior(p_opt))

		result = minimize(fitter.neglnprob,p_opt,method='Nelder-Mead',options={'maxiter': 500000, 'maxfev': 500000, 'xatol': 0.0001, 'fatol': 0.0001})

		print(f'ext: {ext}')
		print(result)
		print(result.success)
		print(result.x)

		p_opt = result.x
		np.save(f'DY_j{ext}_prob.npy',p_opt)

		print('maximum prob')
		print('p = ', p_opt)
		print('lnp = ',fitter.lnprob(p_opt))
		print('lnl at max lnp = ',fitter.lnlikelihood(p_opt))
		print('ln_prior at max lnp = ',fitter.ln_prior(p_opt))

		print()

	else:

		p_opt = np.load(f'DY_j{ext}_prob.npy')


p = fitter.default_params.copy()
p[fitter.freeze==0] = p_opt

with open(f'DY_j{ext}_prob.txt','w') as file:
	file.write(f'p = {p_opt}\n')
	file.write(f'lnp = {fitter.lnprob(p_opt)}\n')
	file.write(f'lnl at max lnp = {fitter.lnlikelihood(p_opt)}\n')
	file.write(f'ln_prior at max lnp = {fitter.ln_prior(p_opt)}\n')



# if not os.path.exists(f'DY_j{ext}_like.npy'):

	
# 	res = np.zeros(s.shape[1])

# 	for param in range(s.shape[1]):

# 	# 	wq = DescrStatsW(data=s[:,param],weights=w)
# 	# 	result = wq.quantile(probs=0.5, return_pandas=False)
# 	# 	res[param] = result[0]
# 		res = dyfunc.quantile(s[:,param],[0.5],weights=w)[0]

# 	#result = minimize(fitter.neglnlikelihood,res,method='Nelder-Mead',options={'maxiter': 500000, 'maxfev': 500000})


# 	result = minimize(fitter.neglnlikelihood,res,method='Powell',options={'maxiter': 500000, 'maxfev': 500000})

# 	print(f'ext: {ext}')
# 	print(result)
# 	print(result.success)
# 	print(result.x)

# 	p_like = result.x
# 	np.save(f'DY_j{ext}_like.npy',p_like)

# 	print('maximum likelihood')
# 	print('p = ', p_like)
# 	print('lnp at max lnl = ',fitter.lnprob(p_like))
# 	print('lnl = ',fitter.lnlikelihood(p_like))
# 	print('ln_prior at max lnl = ',fitter.ln_prior(p_like))
# 	print()

# 	with open(f'DY_j{ext}_like.txt','w') as file:
# 		file.write(f'p = {p_like}\n')
# 		file.write(f'lnp at max lnl = {fitter.lnprob(p_like)}\n')
# 		file.write(f'lnl = {fitter.lnlikelihood(p_like)}\n')
# 		file.write(f'ln_prior at max lnl = {fitter.ln_prior(p_like)}\n')





#
#  Sample with emcee
#
# if not os.path.exists(f'DY_j{ext}_em_samples.npy'):
# 	p = fitter.default_params.copy()
# 	p[fitter.freeze==0] = p_opt
# 	fitter.emcee_sample(p,prefix=f'DY_j{ext}_em_')

# if not os.path.exists(f'DY_j{ext}_em_mean_samples.npy'):
# 	p = fitter.default_params.copy()
# 	p[fitter.freeze==0] = mean
# 	fitter.emcee_sample(p,prefix=f'DY_j{ext}_em_mean_')

# ems = np.load(f'DY_j{ext}_em_samples.npy')
# ems_med = np.median(ems,axis=0)

# ems_mean = np.load(f'DY_j{ext}_em_mean_samples.npy')
# ems_mean_med = np.median(ems,axis=0)



#
#  Now make some summary figures
#

map_sol = np.load(f'DY_j{ext}_prob.npy')

# Uncomment to use maximum likelihood
# mal_sol = np.load(f'DY_j{ext}_like.npy')

# Uncomment to use median of equally-weighted samples
map_sol = np.median(ws,axis=0)
print("weighted median solution:")
print(map_sol)
np.savetxt(f'DY_j{ext}_median.txt',map_sol)


PlotUtils.plot_mass_function(fitter,ws,weights=None,map_sol=map_sol,plot_salpeter=True,plot_chabrier=True,plot_file=f'DY_j{ext}_mf.png')

PlotUtils.plot_q_distribution(fitter,ws,weights=None,map_sol=map_sol,plot_file=f'DY_j{ext}_qdist.png',plot_max=5.0)

ax = PlotUtils.plot_q_distribution(fitter,ws,weights=None,map_sol=map_sol,plot_file=f'DY_j{ext}_qm{int(10*q_min_plot):d}_qdist.png',q_min=q_min_plot,save_figure=False,plot_max=5.0)

#
# Uncomment to overlay exponential q distribution implied by mass function. This only works if logk, M0, c0 and c1 are frozen.
#
def get_exp(gamma,q_min):
	qq = np.linspace(q_min,1.0,1001)
	delta_q = (1-q_min)/1001
	pq = qq**(-gamma)
	pq /= np.sum(pq)*delta_q
	return qq, pq

# plt.figure(figsize=(3,2))
# ax = plt.axes()
p = fitter.default_params.copy()
p[np.where(fitter.freeze == 0)] = map_sol
qq, pq = get_exp(p[2],q_min_plot)
ax.plot(qq,pq,'-',c='coral')
lusol = np.quantile(ws[:,0],[0.16,0.84])
qq, pql = get_exp(lusol[0],q_min_plot)
qq, pqu = get_exp(lusol[1],q_min_plot)
ax.fill_between(qq,y1=pql,y2=pqu,color='coral',alpha=0.4)
chab = qq**(-1.65)
chab /= np.sum(chab)*(qq[1]-qq[0])
ax.plot(qq,chab,'y-',lw=2)
print('Plotting exp with',p[2],lusol[0],lusol[1])
print('maximum values are',pq.max(),pql.max(),pqu.max())

plt.tight_layout()
plt.savefig(f'DY_j{ext}_qm{int(10*q_min_plot):d}_qdist.png')


# PlotUtils.plot_q_distribution(fitter,ems,map_sol=ems_med,plot_file=f'DY_j{ext}_em_qdist.png')
# PlotUtils.plot_q_distribution(fitter,ems,map_sol=ems_med,plot_file=f'DY_j{ext}_qm{int(10*q_min_plot):d}_em_qdist.png',q_min=q_min_plot)

# PlotUtils.plot_q_distribution(fitter,ems,map_sol=ems_mean_med,plot_file=f'DY_j{ext}_em_mean_qdist.png')
# PlotUtils.plot_q_distribution(fitter,ems,map_sol=ems_mean_med,plot_file=f'DY_j{ext}_qm{int(10*q_min_plot):d}_em_mean_qdist.png',q_min=q_min_plot)

PlotUtils.plot_prior_q_distribution(fitter,plot_file=f'DY_j{ext}_qdist_prior.png',plot_max=5.0)
PlotUtils.plot_prior_q_distribution(fitter,plot_file=f'DY_j{ext}_qm{int(10*q_min_plot):d}_qdist_prior.png',q_min=q_min_plot,plot_max=5.0)


#
# Summary figures from optimized solution
#

params = fitter.default_params.copy()

params[np.where(fitter.freeze==0)[0]] = map_sol

PlotUtils.plot_CMD(fitter,params,plot_file=f'DY_j{ext}_CMD.png')


PlotUtils.plot_realisation(fitter,params,plot_file=f'DY_j{ext}_realisation_ns.png',scatter=False,outliers=False,outlier_distribution=False,n_realisations=7)
PlotUtils.plot_realisation(fitter,params,plot_file=f'DY_j{ext}_realisation.png',outliers=False,outlier_distribution=False,n_realisations=7)

print('Printing fbq for [0.4,0.5,0.6,0.7,0.8,0.9]')
result = PlotUtils.print_fb_q(fitter,ws,np.array([0.4,0.5,0.6,0.7,0.8,0.9]),weights=None)
print(result)
print('Printing fbq for [0.5,0.75]')
result = PlotUtils.print_fb_q(fitter,ws,np.array([0.5,0.75]),weights=None)
print(result)


with open(f'DY_j{ext}.state_run','w') as fout:
	pp = pprint.PrettyPrinter(indent=4,stream=fout)
	pp.pprint(vars(fitter))



print('Computing Fq75 metric')
results = PlotUtils.print_Fq75(fitter,ws,weights=None,mf=True)
print(f'Fq75 = {results[0]:.2f}  - {results[1]:.2f} + {results[2]:.2f}')
with open(f'DY_j{ext}_Fq75.txt','w') as file:
	file.write(f'{results[0]:.2f}  {results[1]:.2f}  {results[2]:.2f}\n')
print(f'Fq75_MF = {results[3]:.2f}  - {results[4]:.2f} + {results[5]:.2f}')
with open(f'DY_j{ext}_Fq75_MF.txt','w') as file:
	file.write(f'{results[3]:.2f}  {results[4]:.2f}  {results[5]:.2f}\n')


print('Plotting fbq')
PlotUtils.plot_fb_q(fitter,ws,weights=None,plot_file=f'DY_j{ext}_fbq.png',q_min=q_min_plot,ymax=0.25)








