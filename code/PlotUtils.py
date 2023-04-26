import sys
import os
import numpy as np
#import cunumeric as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from CMDFitter import CMDFitter
from plot_ellipse import plot_ellipse


class PlotUtils():

	"""Helper class to store plotting functions."""


	def plot_mass_function(fitter,samples,map_sol=None,logp=False,weights=None,ax=None,n_plot_samples=1000,save_figure=True,plot_file='mass_function.png',plot_name=True):

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
				ax.plot(m,np.log10(fitter.M_distribution(m,p[:fitter.q_index])),'r-',alpha=1,lw=2)
			else:
				ax.plot(m,fitter.M_distribution(m,p[:fitter.q_index]),'r-',alpha=1,lw=2)

		if plot_name:
			xlimits = ax.get_xlim()
			ylimits = ax.get_ylim()
			ax.scatter(-100,-100,marker='.',s=0.0001,c='w',label=fitter.data.name)
			ax.legend(frameon=False,loc='upper right',fontsize='small')
			ax.set_xlim(xlimits)
			ax.set_ylim(ylimits)


		ax.set_xlabel(r'$M$')

		if logp:
			ax.set_ylabel(r'$P(\log_{10} M)$')
		else:
			ax.set_ylabel(r'$P(M)$')

		if save_figure:
			plt.tight_layout()
			plt.savefig(plot_file)

		return ax




	def plot_q_distribution(fitter,samples,weights=None,map_sol=None,ax=None,n_plot_samples=1000,save_figure=True,plot_file='q_dist.png',plot_name=True):

		"""Plot the implied binary mass-ratio distributiion function for n_plot_samples drawn randomly from samples."""

		import random

		#assert isinstance(fitter,CMDFitter)

		if ax is None:
			plt.figure(figsize=(3,2))
			ax = plt.axes()

		q = np.linspace(fitter.q_min,1.0,101)

		ns = samples.shape[0]

		if weights is not None:
			weights = weights.tolist()

		for i in range(n_plot_samples):
			p = fitter.default_params.copy()
			ind = random.choices(range(ns),k=1,weights=weights)
			p[np.where(fitter.freeze == 0)] = samples[ind]
			args = p[fitter.q_index:fitter.b_index].tolist()
			args.append(fitter.M_ref)
			ax.plot(q,fitter.q_distribution(q,args),'b-',alpha=0.01)

		if map_sol is not None:
			p = fitter.default_params.copy()
			p[np.where(fitter.freeze == 0)] = map_sol
			args = p[fitter.q_index:fitter.b_index].tolist()
			args.append(fitter.M_ref)
			ax.plot(q,fitter.q_distribution(q,args),'r-',alpha=1,lw=2)
			if (fitter.q_model == 'legendre') and (np.sum(fitter.freeze[fitter.q_index+3:fitter.b_index]) < 3) or \
					(fitter.q_model == 'single_power') and (np.sum(fitter.freeze[fitter.q_index+2:fitter.b_index]) < 2) or \
					(fitter.q_model == 'quadratic') and (np.sum(fitter.freeze[fitter.q_index+2:fitter.b_index]) < 2) or \
					(fitter.q_model == 'hist') and (np.sum(fitter.freeze[fitter.q_index+4:fitter.b_index]) < 4):
				args = p[fitter.q_index:fitter.b_index].tolist()
				args.append(fitter.mass_slice[0])
				ax.plot(q,fitter.q_distribution(q,args),'r--',alpha=1,lw=1)
				args = p[fitter.q_index:fitter.b_index].tolist()
				args.append(fitter.mass_slice[1])
				ax.plot(q,fitter.q_distribution(q,args),'r:',alpha=1,lw=1)

		
		if plot_name:
			xlimits = ax.get_xlim()
			ylimits = ax.get_ylim()
			ax.scatter(-100,-100,marker='.',s=0.0001,c='w',label=fitter.data.name)
			ax.legend(frameon=False,loc='upper left',fontsize='small')
			ax.set_xlim(xlimits)
			ax.set_ylim(ylimits)



		ax.set_ylim((0,5))
		ax.set_xlim((fitter.q_min,1))

		ax.set_xlabel(r'$q$')
		ax.set_ylabel(r'$P(q)$')

		if save_figure:
			plt.tight_layout()
			plt.savefig(plot_file)

		return ax


	def plot_prior_q_distribution(fitter,ax=None,n_plot_samples=1000,save_figure=True,plot_file='q_dist.png',alpha=0.01,plot_name=True):

		"""Plot the implied binary mass-ratio distributiion function for n_plot_samples drawn randomly from samples."""

		import random

		#assert isinstance(fitter,CMDFitter)

		if ax is None:
			plt.figure(figsize=(3,2))
			ax = plt.axes()

		p = fitter.default_params.copy()

		q = np.linspace(fitter.q_min,1.0,101)

		nfreeze = int(np.sum(fitter.freeze))

		for i in range(n_plot_samples):

			x = np.random.rand(fitter.ndim-nfreeze)
			pt = fitter.prior_transform(x)
			p[np.where(fitter.freeze == 0)[0]] = pt
			args = p[fitter.q_index:fitter.b_index].tolist()
			args.append(fitter.M_ref)
			ax.plot(q,fitter.q_distribution(q,args),'b-',alpha=alpha)

		if plot_name:
			xlimits = ax.get_xlim()
			ylimits = ax.get_ylim()
			ax.scatter(-100,-100,marker='.',s=0.0001,c='w',label='Prior')
			ax.legend(frameon=False,loc='upper left',fontsize='small')
			ax.set_xlim(xlimits)
			ax.set_ylim(ylimits)


		ax.set_ylim((0,5))
		ax.set_xlim((fitter.q_min,1))

		ax.set_xlabel(r'$q$')
		ax.set_ylabel(r'$P(q)$')

		if save_figure:
			plt.tight_layout()
			plt.savefig(plot_file)

		return ax


	def plot_fb_q(fitter,samples,weights=None,ax=None,save_figure=True,plot_file='fb_q.png',plot_name=True):

		"""Using all samples, plot the implied binary mass-fraction for q' > q along with its 1- and 2-sigma uncertainty."""

		from statsmodels.stats.weightstats import DescrStatsW

		#assert isinstance(fitter, CMDFitter)

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

		p = fitter.default_params.copy()

		for j in range(101):

			y = np.zeros(len(samples))

			for i in range(len(samples)):

				p[fitter.freeze == 0] = samples[i]

				# Weight fb by number of stars of each mass
				m = np.linspace(fitter.mass_slice[0],fitter.mass_slice[1],1001)
				fb = np.trapz(fitter.M_distribution(m,p[:fitter.q_index])*(p[fitter.b_index]+p[fitter.b_index+1]*(fitter.mass_slice[1]-m)),x=m) / \
							np.trapz(fitter.M_distribution(m,p[:fitter.q_index]),x=m)

				# Adjust for outliers
				fb /= 1.0 - p[fitter.b_index+3]

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

		ax.set_xlim((fitter.q_min,1))
		ax.set_ylim((0,0.5))

		ax.tick_params(axis='y',which='both',direction='in',right=True)
		ax.tick_params(axis='x',which='both',direction='in',top=True)

		if save_figure:
			plt.tight_layout()
			plt.savefig(plot_file)

		return ax, yq3, yq3-yq2, yq4-yq3


	def print_fb_q(fitter,samples,weights,q_dash):

		"""Using all samples, print the implied binary mass-fraction for q' > q along with its 1- and 2-sigma uncertainty."""

		from statsmodels.stats.weightstats import DescrStatsW

		#assert isinstance(fitter, CMDFitter)

		q = (q_dash-fitter.q_min)/(1.0-fitter.q_min)

		sig1 = 0.5 * 68.27
		sig2 = 0.5 * 95.45

		yq1 = np.zeros_like(q)
		yq2 = np.zeros_like(q)
		yq3 = np.zeros_like(q)
		yq4 = np.zeros_like(q)
		yq5 = np.zeros_like(q)

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
				fb /= 1.0 - p[fitter.b_index+3]

				y[i] = fb * fitter.q_distribution_integral(p[fitter.q_index:fitter.b_index],q[j],1.0)

			wq  = DescrStatsW(data=y,weights=weights)
			qq = wq.quantile(probs=np.array(0.01*np.array([50.0-sig2,50.0-sig1,50.0,50.0+sig1,50.0+sig2])),\
			            return_pandas=False)

			yq1[j] = qq[0]
			yq2[j] = qq[1]
			yq3[j] = qq[2]
			yq4[j] = qq[3]
			yq5[j] = qq[4]

		return yq1, yq2, yq3, yq4, yq5


	def plot_prior_fb_q(fitter,n_samples=1000,ax=None,save_figure=True,plot_file='fb_q.png',plot_name=True):

		"""Using all samples, plot the implied binary mass-fraction for q' > q along with its 1- and 2-sigma uncertainty."""

		from statsmodels.stats.weightstats import DescrStatsW

		#assert isinstance(fitter, CMDFitter)

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
				fb /= 1.0 - p[fitter.b_index+3]

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

		ax.set_xlim((fitter.q_min,1))
		ax.set_ylim((0,0.5))

		ax.tick_params(axis='y',which='both',direction='in',right=True)
		ax.tick_params(axis='x',which='both',direction='in',top=True)

		if save_figure:
			plt.tight_layout()
			plt.savefig(plot_file)

		return ax


	def plot_realisation(fitter,params,plot_file='realisation.png',outliers=True,scatter=True,outlier_distribution=False,plot_name=True,s_colour='k',b_colour='k',o_colour='k'):

		"""Plot the data CMD and 3 comparative random realisations of the model implied by params."""

		#assert isinstance(fitter, CMDFitter)

		if outlier_distribution:
			oc, om, ocv, omv, ocov = params[fitter.i_index-8:fitter.i_index-3]
			cov = np.array([[ocv,ocov],[ocov,omv]])

		n = len(fitter.data.magnitude)

		xmag = np.linspace(fitter.data.magnitude_min,fitter.data.magnitude_max,1001)

		fig, ax = plt.subplots(1,4,figsize=(6,3),sharey='row')

		ax[0].scatter(fitter.data.colour,fitter.data.magnitude,s=0.5,c='k')
		ax[0].plot(fitter.iso.mag_colour_interp(xmag),xmag,'g-',alpha=0.6)
		ax[0].invert_yaxis()
		xlimits = ax[0].get_xlim()
		ylimits = ax[0].get_ylim()

		if outlier_distribution:
			plot_ellipse(ax[0],x_cent=oc,y_cent=om,cov=cov,plot_kwargs={'color':'orange'})

		if plot_name:
			ax[0].scatter(-100,-100,marker='.',s=0.0001,c='w',label=fitter.data.name)
			ax[0].legend(frameon=False,loc='upper right',fontsize='small')
			ax[0].set_xlim(xlimits)
			ax[0].set_ylim(ylimits)


		ax[0].set_xlabel(fitter.data.colour_label)
		ax[0].set_ylabel(fitter.data.magnitude_label)
		ax[0].set_ylim([fitter.data.magnitude_max+0.3,fitter.data.magnitude_min-1])
		ax[0].tick_params(axis='y',which='both',direction='in',right=True)
		ax[0].tick_params(axis='x',which='both',direction='in',top=True)

		for i in range(1,4):

			mag, colour, star_type = fitter.model_realisation(params,n,add_observational_scatter=scatter,outliers=outliers)
			ax[i].scatter(colour[star_type==0],mag[star_type==0],s=0.5,color=s_colour)
			ax[i].scatter(colour[star_type==1],mag[star_type==1],s=0.5,color=b_colour)
			ax[i].scatter(colour[star_type==2],mag[star_type==2],s=0.5,color=o_colour)
			ax[i].plot(fitter.iso.mag_colour_interp(xmag),xmag,'g-',alpha=0.6)
			#ax[i].grid()
			ax[i].set_xlabel(fitter.data.colour_label)
			ax[i].set_xlim(xlimits)
			ax[i].set_ylim(ylimits)
			ax[i].tick_params(axis='y',which='both',direction='in',right=True)
			ax[i].tick_params(axis='x',which='both',direction='in',top=True)



		plt.tight_layout()

		plt.savefig(plot_file)

		plt.close()


