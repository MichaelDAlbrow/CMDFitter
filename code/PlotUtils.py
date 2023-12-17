import sys
import os
import numpy as np
#import cunumeric as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


#from CMDFitter import CMDFitter
from plot_ellipse import plot_ellipse


class PlotUtils():

	"""Helper class to store plotting functions."""


	def plot_mass_function(fitter,samples,map_sol=None,mal_sol=None,logp=False,weights=None,ax=None,n_plot_samples=1000,save_figure=True,plot_file='mass_function.png',
				plot_name=True,map_colour='c',mal_colour='chocolate',plot_salpeter=False,salpeter_colour='k',plot_chabrier=False,chabrier_colour='k'):

		"""Plot the implied mass function for n_plot_samples drawn randomly from samples."""

		import random

		#print('type fitter:',type(fitter))
		#assert isinstance(fitter,CMDFitter)

		if ax is None:
			plt.figure(figsize=(3,2))
			ax = plt.axes()

		m = np.linspace(fitter.mass_slice[0],fitter.mass_slice[1],1001)

		ns = samples.shape[0]

		if weights is not None:
			weights = weights.tolist()

		for i in range(n_plot_samples):
			p = fitter.default_params.copy()
			ind = random.choices(range(ns),k=1,weights=weights)
			p[np.where(fitter.freeze == 0)] = samples[ind]
			if logp:
				ax.plot(m,np.log10(fitter.M_distribution(m,p[:fitter.q_index])),'b-',alpha=0.01)
			else:
				ax.plot(m,fitter.M_distribution(m,p[:fitter.q_index]),'b-',alpha=0.01)


		if map_sol is not None:
			p = fitter.default_params.copy()
			p[np.where(fitter.freeze == 0)] = map_sol
			args = p[fitter.q_index:fitter.b_index].tolist()
			if logp:
				ax.plot(m,np.log10(fitter.M_distribution(m,p[:fitter.q_index])),c=mal_colour,ls='-',alpha=1,lw=2)
			else:
				ax.plot(m,fitter.M_distribution(m,p[:fitter.q_index]),c=mal_colour,ls='-',alpha=1,lw=2)
				print('map_sol',np.sum(fitter.M_distribution(m,p[:fitter.q_index])))

		if mal_sol is not None:
			p = fitter.default_params.copy()
			p[np.where(fitter.freeze == 0)] = map_sol
			args = p[fitter.q_index:fitter.b_index].tolist()
			if logp:
				ax.plot(m,np.log10(fitter.M_distribution(m,p[:fitter.q_index])),c=mal_colour,ls='-',alpha=1,lw=2)
			else:
				ax.plot(m,fitter.M_distribution(m,p[:fitter.q_index]),c=mal_colour,ls='-',alpha=1,lw=2)

		if plot_salpeter:
			c = 1.35 / ( m[0]**(-1.35) - m[-1]**(-1.35) )
			ax.plot(m,c*m**(-2.35),c=salpeter_colour,ls='-.',lw=1.0,alpha=0.5)
			print('salpeter',np.sum(c*m**(-2.35)))

		if plot_chabrier:
			dm = m[1] - m[0]
			s1 = np.sum(np.exp(-( (np.log10(m[m<1.0]) - np.log10(0.08))**2 / (2*0.69**2) ) ) / m[m<1.0])
			c = 1.0 / (s1*dm + 0.28263*(m[-1]**(-1.35) - 1.0)/(-1.35) ) 
			f = c * np.exp(-( (np.log10(m) - np.log10(0.08))**2 / (2*0.69**2) ) ) / m
			f[m>=1.0] = 0.28263 * c * m[m>=1.0]**(-2.35)
			ax.plot(m,f,c=chabrier_colour,ls='--',lw=1,alpha=0.5)
			print('chabrier',np.sum(f))

		ax.set_ylim((0,4))

		if plot_name:
			xlimits = ax.get_xlim()
			ylimits = ax.get_ylim()
			ax.scatter(-100,-100,marker='.',s=0.0001,c='w',label=fitter.data.name)
			ax.legend(frameon=False,loc='upper right',fontsize='small')
			ax.set_xlim(xlimits)
			ax.set_ylim(ylimits)


		ax.set_xlabel(r'$M / M_\odot$')

		if logp:
			ax.set_ylabel(r'$P(\log_{10} M)$')
		else:
			ax.set_ylabel(r'$dP/dM$')

		ax.tick_params(axis='y',which='both',direction='in',right=True)
		ax.tick_params(axis='x',which='both',direction='in',top=True)

		if save_figure:
			plt.tight_layout()
			plt.savefig(plot_file)

		return ax




	def plot_q_distribution(fitter,samples,weights=None,map_sol=None,mal_sol=None,ax=None,n_plot_samples=2000,save_figure=True,plot_file='q_dist.png',plot_name=True,
							map_colour='r',mal_colour='chocolate',plot_max=4.0,q_min=None):

		"""Plot the implied binary mass-ratio distributiion function for n_plot_samples drawn randomly from samples."""

		import random

		#assert isinstance(fitter,CMDFitter)

		if ax is None:
			plt.figure(figsize=(3,2))
			ax = plt.axes()

		q = np.linspace(fitter.q_min,1.0,1001)
		delta_q = (1-fitter.q_min)/1001

		ns = samples.shape[0]

		if weights is not None:
			weights = weights.tolist()

		if weights is None:
			weights = np.ones(samples.shape[0]).tolist()

		if q_min is None:
			q_min = fitter.q_min

		for i in range(n_plot_samples):
			p = fitter.default_params.copy()
			ind = random.choices(range(ns),k=1,weights=weights)
			p[np.where(fitter.freeze == 0)] = samples[ind]
			args = p[fitter.q_index:fitter.b_index].tolist()
			args.append(fitter.M_ref)
			y = fitter.q_distribution(q,args)
			y /= np.mean(y[q>=q_min])*(1-q_min)
			#y /= np.sum(y[q>=q_min])*delta_q
			ax.plot(q,y,'b-',alpha=0.01)

		if mal_sol is not None:
			p = fitter.default_params.copy()
			p[np.where(fitter.freeze == 0)] = mal_sol
			args = p[fitter.q_index:fitter.b_index].tolist()
			args.append(fitter.M_ref)
			y = fitter.q_distribution(q,args)
			y /= np.mean(y[q>=q_min])
			#y /= np.sum(y[q>=q_min])*delta_q
			print('q_min, mean y', q_min, np.mean(y[q>=q_min]))
			ax.plot(q,y,c=mal_colour,ls='-',alpha=1,lw=2)
			if (fitter.q_model == 'legendre') and (np.sum(fitter.freeze[fitter.q_index+3:fitter.b_index]) < 3) or \
					(fitter.q_model == 'single_power') and (np.sum(fitter.freeze[fitter.q_index+2:fitter.b_index]) < 2) or \
					(fitter.q_model == 'quadratic') and (np.sum(fitter.freeze[fitter.q_index+2:fitter.b_index]) < 2) or \
					(fitter.q_model == 'hist') and (np.sum(fitter.freeze[fitter.q_index+fitter.n_q_hist_bins-1:fitter.b_index]) < 4):
				args = p[fitter.q_index:fitter.b_index].tolist()
				args.append(fitter.mass_slice[0])
				y = fitter.q_distribution(q,args)
				y /= np.mean(y[q>=q_min])*(1-q_min)*(1-q_min)
				#y /= np.sum(y[q>=q_min])*delta_q
				ax.plot(q,y,c=mal_colour,ls='--',alpha=1,lw=1)
				args = p[fitter.q_index:fitter.b_index].tolist()
				args.append(fitter.mass_slice[1])
				y = fitter.q_distribution(q,args)
				y /= np.mean(y[q>=q_min])*(1-q_min)
				#y /= np.sum(y[q>=q_min])*delta_q
				ax.plot(q,y,c=mal_colour,ls=':',alpha=1,lw=1)

		if map_sol is not None:
			p = fitter.default_params.copy()
			p[np.where(fitter.freeze == 0)] = map_sol
			args = p[fitter.q_index:fitter.b_index].tolist()
			args.append(fitter.M_ref)
			y = fitter.q_distribution(q,args)
			y /= np.mean(y[q>=q_min])*(1-q_min)
			#y /= np.sum(y[q>=q_min])*delta_q
			print('q_min, mean y', q_min, np.mean(y[q>=q_min]))
			ax.plot(q,y,c=map_colour,ls='-',alpha=1,lw=2)
			if (fitter.q_model == 'legendre') and (np.sum(fitter.freeze[fitter.q_index+3:fitter.b_index]) < 3) or \
					(fitter.q_model == 'single_power') and (np.sum(fitter.freeze[fitter.q_index+2:fitter.b_index]) < 2) or \
					(fitter.q_model == 'quadratic') and (np.sum(fitter.freeze[fitter.q_index+2:fitter.b_index]) < 2) or \
					(fitter.q_model == 'hist') and (np.sum(fitter.freeze[fitter.q_index+fitter.n_q_hist_bins-1:fitter.b_index]) < 4):
				args = p[fitter.q_index:fitter.b_index].tolist()
				args.append(fitter.mass_slice[0])
				y = fitter.q_distribution(q,args)
				y /= np.mean(y[q>=q_min])*(1-q_min)
				#y /= np.sum(y[q>=q_min])*delta_q
				ax.plot(q,y,c=map_colour,ls='--',alpha=1,lw=1)
				args = p[fitter.q_index:fitter.b_index].tolist()
				args.append(fitter.mass_slice[1])
				y = fitter.q_distribution(q,args)
				y /= np.mean(y[q>=q_min])*(1-q_min)
				#y /= np.sum(y[q>=q_min])*delta_q
				ax.plot(q,y,c=map_colour,ls=':',alpha=1,lw=1)

		
		if plot_name:
			xlimits = ax.get_xlim()
			ylimits = ax.get_ylim()
			ax.scatter(-100,-100,marker='.',s=0.0001,c='w',label=fitter.data.name)
			ax.legend(frameon=False,loc='upper left',fontsize='small')
			ax.set_xlim(xlimits)
			ax.set_ylim(ylimits)



		ax.set_ylim((0,plot_max))
		ax.set_xlim((q_min,1))

		ax.set_xlabel(r'$q$')
		ax.set_ylabel(r'$dP/dq$')

		ax.tick_params(axis='y',which='both',direction='in',right=True)
		ax.tick_params(axis='x',which='both',direction='in',top=True)

		if save_figure:
			plt.tight_layout()
			plt.savefig(plot_file)

		return ax


	def plot_prior_q_distribution(fitter,ax=None,n_plot_samples=1000,save_figure=True,plot_file='q_dist.png',alpha=0.01,plot_name=True,q_min=None,plot_max=4.0):

		"""Plot the implied binary mass-ratio distributiion function for n_plot_samples drawn randomly from samples."""

		import random

		#assert isinstance(fitter,CMDFitter)

		if q_min is None:
			q_min = fitter.q_min

		if ax is None:
			plt.figure(figsize=(3,2))
			ax = plt.axes()

		p = fitter.default_params.copy()

		q = np.linspace(fitter.q_min,1.0,1001)

		nfreeze = int(np.sum(fitter.freeze))

		i = 0
		while i < n_plot_samples:

			x = np.random.rand(fitter.ndim-nfreeze)
			pt = fitter.prior_transform(x)
			p[np.where(fitter.freeze == 0)[0]] = pt
			args = p[fitter.q_index:fitter.b_index].tolist()
			args.append(fitter.M_ref)
			y = fitter.q_distribution(q,args)
			y /= np.mean(y[q>=q_min])
			if np.min(y) < 0.0:
				continue

			ax.plot(q,y,'b-',alpha=alpha)
			i += 1

		if plot_name:
			xlimits = ax.get_xlim()
			ylimits = ax.get_ylim()
			ax.scatter(-100,-100,marker='.',s=0.0001,c='w',label='Prior')
			ax.legend(frameon=False,loc='upper left',fontsize='small')
			ax.set_xlim(xlimits)
			ax.set_ylim(ylimits)


		ax.set_ylim((0,plot_max))
		ax.set_xlim((q_min,1))

		ax.set_xlabel(r'$q$')
		ax.set_ylabel(r'$dP/dq$')

		ax.tick_params(axis='y',which='both',direction='in',right=True)
		ax.tick_params(axis='x',which='both',direction='in',top=True)

		if save_figure:
			plt.tight_layout()
			plt.savefig(plot_file)

		return ax


	def plot_fb_q(fitter,samples,weights=None,ax=None,save_figure=True,plot_file='fb_q.png',plot_name=True,q_min=None,ymax=0.35):

		"""Using all samples, plot the implied binary mass-fraction for q' > q along with its 1- and 2-sigma uncertainty."""

		from statsmodels.stats.weightstats import DescrStatsW

		#assert isinstance(fitter, CMDFitter)

		if q_min is None:
			q_min = fitter.q_min

		if ax is None:
			plt.figure(figsize=(3,2))
			ax = plt.axes()

		ns, nq = samples.shape

		if weights is None:
			weights = np.ones(ns)

		q = np.linspace(fitter.q_min,1.0,101)
		m = np.linspace(fitter.mass_slice[0],fitter.mass_slice[1],1001)

		sig1 = 0.5 * 68.27
		sig2 = 0.5 * 95.45

		yq1 = np.zeros(101)
		yq2 = np.zeros(101)
		yq3 = np.zeros(101)
		yq4 = np.zeros(101)
		yq5 = np.zeros(101)

		p = fitter.default_params.copy()

		for j in range(101):

			y = np.zeros(len(samples))

			for i in range(len(samples)):

				p[fitter.freeze == 0] = samples[i]

				# Weight fb by number of stars of each mass
				fb = np.trapz(fitter.M_distribution(m,p[:fitter.q_index])*(p[fitter.b_index]+p[fitter.b_index+1]*(fitter.mass_slice[1]-m)),x=m) / \
							np.trapz(fitter.M_distribution(m,p[:fitter.q_index]),x=m)

				# Adjust for outliers
				fb /= 1.0 - p[fitter.i_index-3]

				args = p[fitter.q_index:fitter.b_index].tolist()
				args.append(fitter.M_ref)
				y[i] = fb * fitter.q_distribution_integral(args,q[j],1.0)



			wq  = DescrStatsW(data=y,weights=weights)
			qq = wq.quantile(probs=np.array(0.01*np.array([50.0-sig2,50.0-sig1,50.0,50.0+sig1,50.0+sig2])),\
			            return_pandas=False)

			yq1[j] = qq[0]
			yq2[j] = qq[1]
			yq3[j] = qq[2]
			yq4[j] = qq[3]
			yq5[j] = qq[4]


		# debugging

		j = 75
		i = len(samples) - 1
		p[fitter.freeze == 0] = samples[i]
		fb = np.trapz(fitter.M_distribution(m,p[:fitter.q_index])*(p[fitter.b_index]+p[fitter.b_index+1]*(fitter.mass_slice[1]-m)),x=m) / \
					np.trapz(fitter.M_distribution(m,p[:fitter.q_index]),x=m)
		fb /= 1.0 - p[fitter.i_index-3]
		args = p[fitter.q_index:fitter.b_index].tolist()
		args.append(fitter.M_ref)
		print('debugging')
		print('p',p)
		print('args',args)
		print('q,fb,integral =',q[j],fb,fitter.q_distribution_integral(args,q[j],1.0))


		# end debugging


		ax.fill_between(q,y1=yq1,y2=yq5,color='b',alpha=0.1)
		ax.fill_between(q,y1=yq2,y2=yq4,color='b',alpha=0.4)

		ax.plot(q,yq3,'r-')

		ax.set_xlabel(r'$q$')
		ax.set_ylabel(r"$f_B \, (q'>q)$")

		if plot_name:
			xlimits = ax.get_xlim()
			ylimits = ax.get_ylim()
			ax.scatter(-100,-100,marker='.',s=0.0001,c='w',label=fitter.data.name)
			ax.legend(frameon=False,loc='upper right',fontsize='small')
			ax.set_xlim(xlimits)
			ax.set_ylim(ylimits)

		ax.set_xlim((q_min,1))
		ax.set_ylim((0,ymax))

		ax.tick_params(axis='y',which='both',direction='in',right=True)
		ax.tick_params(axis='x',which='both',direction='in',top=True)

		if save_figure:
			plt.tight_layout()
			plt.savefig(plot_file)

		return ax, yq3, yq3-yq2, yq4-yq3


	def print_fb_q(fitter,samples,q,weights=None):

		"""Using all samples, print the implied binary mass-fraction for q' > q along with its 1- and 2-sigma uncertainty."""

		from statsmodels.stats.weightstats import DescrStatsW

		#assert isinstance(fitter, CMDFitter)

		ns, nq = samples.shape

		if weights is None:
			weights = np.ones(ns)

		sig1 = 0.5 * 68.27
		sig2 = 0.5 * 95.45


		yq1 = np.zeros(nq)
		yq2 = np.zeros(nq)
		yq3 = np.zeros(nq)
		yq4 = np.zeros(nq)
		yq5 = np.zeros(nq)

		p = fitter.default_params.copy()

		for j in range(len(q)):

			y = np.zeros(len(samples))

			for i in range(len(samples)):

				p[fitter.freeze == 0] = samples[i]

				# Weight fb by number of stars of each mass
				m = np.linspace(fitter.mass_slice[0],fitter.mass_slice[1],1001)
				fb = np.trapz(fitter.M_distribution(m,p[:fitter.q_index])*(p[fitter.b_index]+p[fitter.b_index+1]*(fitter.mass_slice[1]-m)),x=m) / \
							np.trapz(fitter.M_distribution(m,p[:fitter.q_index]),x=m)

				# Adjust for outliers
				fb /= 1.0 - p[fitter.i_index-3]

				args = p[fitter.q_index:fitter.b_index].tolist()
				args.append(fitter.M_ref)
				y[i] = fb * fitter.q_distribution_integral(args,q[j],1.0)

			wq  = DescrStatsW(data=y,weights=weights)
			qq = wq.quantile(probs=np.array(0.01*np.array([50.0-sig2,50.0-sig1,50.0,50.0+sig1,50.0+sig2])),\
			            return_pandas=False)


			yq1[j] = qq[0]
			yq2[j] = qq[1]
			yq3[j] = qq[2]
			yq4[j] = qq[3]
			yq5[j] = qq[4]

		return yq1, yq2, yq3, yq4, yq5


	def print_Fq75(fitter,samples,weights=None,mf=True):

		"""Using all samples, print the implied binary mass-fraction for q' > q along with its 1- and 2-sigma uncertainty.
			If mf=True, assume a power-law mass function, and also return the implied RQ75"""

		from statsmodels.stats.weightstats import DescrStatsW

		#assert isinstance(fitter, CMDFitter)

		ns, nq = samples.shape

		if weights is None:
			weights = np.ones(ns)

		sig = 0.5 * 68.27

		yq1 = np.zeros(nq)
		yq2 = np.zeros(nq)
		yq3 = np.zeros(nq)
		p = fitter.default_params.copy()

		R = np.zeros(len(samples))
		Rmf = np.zeros(len(samples))

		for i in range(len(samples)):

			p[fitter.freeze == 0] = samples[i]

			# Weight fb by number of stars of each mass
			m = np.linspace(fitter.mass_slice[0],fitter.mass_slice[1],1001)
			fb = np.trapz(fitter.M_distribution(m,p[:fitter.q_index])*(p[fitter.b_index]+p[fitter.b_index+1]*(fitter.mass_slice[1]-m)),x=m) / \
						np.trapz(fitter.M_distribution(m,p[:fitter.q_index]),x=m)

			# Adjust for outliers
			fb /= 1.0 - p[fitter.i_index-3]

			args = p[fitter.q_index:fitter.b_index].tolist()
			args.append(fitter.M_ref)
			y50 = fb * fitter.q_distribution_integral(args,0.5,1.0)
			y75 = fb * fitter.q_distribution_integral(args,0.75,1.0)

			# Original definition
			#R[i] = y75 / (y50 - y75)
			R[i] = y75 / y50 

			gamma = p[2]
			Rmf[i] = (1-(0.75)**(-gamma)) / (1-(0.5)**(-gamma))

		wq  = DescrStatsW(data=R,weights=weights)
		qq = wq.quantile(probs=np.array(0.01*np.array([50.0-sig,50.0,50.0+sig])),\
		            return_pandas=False)

		if mf:
			wq  = DescrStatsW(data=Rmf,weights=weights)
			qq_mf = wq.quantile(probs=np.array(0.01*np.array([50.0-sig,50.0,50.0+sig])),\
		            return_pandas=False)
			return qq[1],qq[1]-qq[0],qq[2]-qq[1], qq_mf[1],qq_mf[1]-qq_mf[0],qq_mf[2]-qq_mf[1]

		return qq[1],qq[1]-qq[0],qq[2]-qq[1]



	def plot_prior_fb_q(fitter,n_samples=1000,ax=None,save_figure=True,plot_file='fb_q.png',plot_name=True,q_min=None):

		"""Using all samples, plot the implied binary mass-fraction for q' > q along with its 1- and 2-sigma uncertainty."""

		from statsmodels.stats.weightstats import DescrStatsW

		#assert isinstance(fitter, CMDFitter)

		if q_min is None:
			q_min = fitter.q_min

		scale = (1.0 - fitter.q_min) / (1.0 - q_min)

		assert fitter.q_model == 'legendre'

		if ax is None:
			plt.figure(figsize=(3,2))
			ax = plt.axes()

		q = np.linspace(fitter.q_min,1.0,101)

		sig1 = 0.5 * 68.27
		sig2 = 0.5 * 95.45

		yq1 = np.zeros(101)
		yq2 = np.zeros(101)
		yq3 = np.zeros(101)
		yq4 = np.zeros(101)
		yq5 = np.zeros(101)

		x = np.random.rand(n_samples,fitter.ndim)

		y = np.zeros(n_samples)

		for j in range(101):

			for i in range(n_samples):
		
				p = fitter.prior_transform(x[i])

				# Weight fb by number of stars of each mass
				m = np.linspace(fitter.mass_slice[0],fitter.mass_slice[1],1001)
				fb = np.trapz(fitter.M_distribution(m,p[:fitter.q_index])*(p[fitter.b_index]+p[fitter.b_index+1]*(fitter.mass_slice[1]-m)),x=m) / \
							np.trapz(fitter.M_distribution(m,p[:fitter.q_index]),x=m)

				# Adjust for outliers
				fb /= 1.0 - p[fitter.i_index-3]

				y[i] = fb * fitter.q_distribution_integral(p[fitter.q_index:fitter.b_index],q[j],1.0)

			wq  = DescrStatsW(data=y)
			qq = wq.quantile(probs=np.array(0.01*np.array([50.0-sig2,50.0-sig1,50.0,50.0+sig1,50.0+sig2])),\
			            return_pandas=False)

			yq1[j] = qq[0]
			yq2[j] = qq[1]
			yq3[j] = qq[2]
			yq4[j] = qq[3]
			yq5[j] = qq[4]

		ax.fill_between(q,y1=yq1,y2=yq5,color='b',alpha=0.1)
		ax.fill_between(q,y1=yq2,y2=yq4,color='b',alpha=0.4)

		ax.plot(q,yq3,'r-')

		ax.set_xlabel(r'$q$')
		ax.set_ylabel(r"$f_B \, (q'>q)$")

		if plot_name:
			xlimits = ax.get_xlim()
			ylimits = ax.get_ylim()
			ax.scatter(-100,-100,marker='.',s=0.0001,c='w',label='Prior')
			ax.legend(frameon=False,loc='upper right',fontsize='small')
			ax.set_xlim(xlimits)
			ax.set_ylim(ylimits)

		ax.set_xlim((q_min,1))
		ax.set_ylim((0,0.5))

		ax.tick_params(axis='y',which='both',direction='in',right=True)
		ax.tick_params(axis='x',which='both',direction='in',top=True)

		if save_figure:
			plt.tight_layout()
			plt.savefig(plot_file)

		return ax


	def plot_realisation(fitter,params,plot_file='realisation.png',outliers=True,scatter=True,outlier_distribution=False,plot_name=True,s_colour='k',b_colour='k',o_colour='w',n_realisations=3):

		"""Plot the data CMD and 3 comparative random realisations of the model implied by params."""

		assert n_realisations in [3,7]  

		if outlier_distribution:
			oc, om, ocv, omv, ocov = params[fitter.i_index-8:fitter.i_index-3]
			cov = np.array([[ocv,ocov],[ocov,omv]])

		n = len(fitter.data.magnitude)

		xmag = np.linspace(fitter.data.magnitude_min,fitter.data.magnitude_max,1001)

		if n_realisations == 3:
			figsize = (6,3)
			n_rows = 1
		else:
			figsize = (6,6)
			n_rows = 2

		fig, ax = plt.subplots(n_rows,4,figsize=figsize,sharey='row')

		print(figsize, n_rows, n_realisations, ax.shape)

		if n_realisations == 3:
			ax0 = ax[0]
		else:
			ax0 = ax[0,0]

		ax0.scatter(fitter.data.colour,fitter.data.magnitude,s=0.5,c='k')
		ax0.plot(fitter.iso.mag_colour_interp(xmag),xmag,'g-',alpha=0.6)
		ax0.invert_yaxis()
		xlimits = ax0.get_xlim()
		ylimits = ax0.get_ylim()

		if outlier_distribution:
			plot_ellipse(ax0,x_cent=oc,y_cent=om,cov=cov,plot_kwargs={'color':'orange'})

		if plot_name:
			ax0.scatter(-100,-100,marker='.',s=0.0001,c='w',label=fitter.data.name)
			ax0.legend(frameon=False,loc='upper right',fontsize='small')
			ax0.set_xlim(xlimits)
			ax0.set_ylim(ylimits)


		ax0.set_xlabel(fitter.data.colour_label)
		ax0.set_ylabel(fitter.data.magnitude_label)
		ax0.set_ylim([fitter.data.magnitude_max+0.3,fitter.data.magnitude_min-1])
		ax0.tick_params(axis='y',which='both',direction='in',right=True)
		ax0.tick_params(axis='x',which='both',direction='in',top=True)

		for j in range(n_rows):
			for i in range(4):

				if j == 0 and i == 0:
					continue

				if n_rows == 1:
					axi = ax[i]
				else:
					axi = ax[j,i]

				if i == 0:
					axi.set_ylabel(fitter.data.magnitude_label)
				
				mag, colour, star_type = fitter.model_realisation(params,n,add_observational_scatter=scatter,outliers=outliers)
				axi.scatter(colour[star_type==0],mag[star_type==0],s=0.5,color=s_colour)
				axi.scatter(colour[star_type==1],mag[star_type==1],s=0.5,color=b_colour)
				axi.scatter(colour[star_type==2],mag[star_type==2],s=0.5,color=o_colour)
				axi.plot(fitter.iso.mag_colour_interp(xmag),xmag,'g-',alpha=0.6)
				#ax[i].grid()
				axi.set_xlabel(fitter.data.colour_label)
				axi.set_xlim(xlimits)
				axi.set_ylim(ylimits)
				axi.tick_params(axis='y',which='both',direction='in',right=True)
				axi.tick_params(axis='x',which='both',direction='in',top=True)



		plt.tight_layout()

		plt.savefig(plot_file)

		plt.close()


	def plot_CMD(fitter,params,plot_file='CMD.png',plot_name=True):

		"""Plot the data CMD."""

		n = len(fitter.data.magnitude)

		xmag = np.linspace(fitter.data.magnitude_min,fitter.data.magnitude_max,1001)

		fig, ax = plt.subplots(1,1,figsize=(2,3))

		ax.scatter(fitter.data.colour,fitter.data.magnitude,s=0.5,c='k')
		ax.plot(fitter.iso.mag_colour_interp(xmag),xmag,'r-',lw=3,alpha=0.3)
		ax.invert_yaxis()
		xlimits = ax.get_xlim()
		ylimits = ax.get_ylim()

		if plot_name:
			ax.scatter(-100,-100,marker='.',s=0.0001,c='w',label=fitter.data.name)
			ax.legend(frameon=False,loc='upper right',fontsize='small')
			ax.set_xlim(xlimits)
			ax.set_ylim(ylimits)


		ax.set_xlabel(fitter.data.colour_label)
		ax.set_ylabel(fitter.data.magnitude_label)
		ax.set_ylim([fitter.data.magnitude_max+0.3,fitter.data.magnitude_min-1])
		ax.tick_params(axis='y',which='both',direction='in',right=True)
		ax.tick_params(axis='x',which='both',direction='in',top=True)

		plt.tight_layout()

		plt.savefig(plot_file)

		plt.close()


