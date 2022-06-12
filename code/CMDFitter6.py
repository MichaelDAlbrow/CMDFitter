import sys
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize, nnls

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pylab import subplots_adjust

import json

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from likelihood_gaussbf_CUDA import likelihood_functions
likelihood = likelihood_functions.get_function("likelihood")

####################################################################################################


class Data():

	"""Container class to hold CMD data"""

	def __init__(self,data_dict):

		"""
		Set up a Data instance.


		Inputs:

			data_dict:			(dictionary) with entries:
								file : (string) data file name
								magnitude_min: (float) minimum magnitude to use
								magnitude_max: (float) maximum magnitude to use
								column_mag: (int) column number in file corresponding to magnitude
								column_colour_blue: (int) column number in file for blue component of colour  
								column_colour_red: (int) column number in file for red component of colour
								column_mag_err: (int) column number in file corresponding to magnitude
								column_colour_blue_err: (int) column number in file for blue component of colour  
								column_colour_red_err: (int) column number in file for red component of colour
								colour_label: (string)
								magnitude_label: (string)

		"""

		print('data definition:',data_dict)

		data = np.loadtxt(data_dict['file'])

		self.magnitude = data[:,data_dict['column_mag']]

		self.colour = data[:,data_dict['column_blue']] - data[:,data_dict['column_red']]

		self.magnitude_err = data[:,data_dict['column_mag_err']]
		self.colour_err = np.sqrt(data[:,data_dict['column_blue_err']]**2 + data[:,data_dict['column_red_err']]**2)

		self.magnitude_min = data_dict['magnitude_min']
		self.magnitude_max = data_dict['magnitude_max']

		self.colour_label = data_dict['colour_label']
		self.magnitude_label = data_dict['magnitude_label']

		# set up data covariance matrices

		self.cov = np.empty((len(self.magnitude),2,2),dtype='float32')

		if data_dict['column_mag'] == data_dict['column_blue']:

			self.cov[:,0,0] = data[:,data_dict['column_blue_err']]**2 + data[:,data_dict['column_red_err']]**2
			self.cov[:,0,1] = data[:,data_dict['column_blue_err']]**2
			self.cov[:,1,0] = data[:,data_dict['column_blue_err']]**2
			self.cov[:,1,1] = data[:,data_dict['column_blue_err']]**2
			self.magnitude_type = 'blue'

		elif data_dict['column_mag'] == data_dict['column_red']:

			self.cov[:,0,0] = data[:,data_dict['column_blue_err']]**2 + data[:,data_dict['column_red_err']]**2
			self.cov[:,0,1] = data[:,data_dict['column_red_err']]**2
			self.cov[:,1,0] = data[:,data_dict['column_red_err']]**2
			self.cov[:,1,1] = data[:,data_dict['column_red_err']]**2
			self.magnitude_type = 'red'

		else:

			self.cov[:,0,0] = data[:,data_dict['column_blue_err']]**2 + data[:,data_dict['column_red_err']]**2
			self.cov[:,0,1] = 0.0
			self.cov[:,1,0] = 0.0
			self.cov[:,1,1] = data[:,data_dict['column_mag_err']]**2
			self.magnitude_type = 'independent'

		return



	def upload_to_GPU(self):

		# upload data to GPU texture memory

		c_cov = self.cov.reshape((len(self.magnitude),4),order='F')
		self.cov_CUDA = likelihood_functions.get_texref("cov")
		drv.matrix_to_texref(np.float32(c_cov),self.cov_CUDA,order='F')
		#texarray = drv.np_to_array(np.float32(np.moveaxis(self.cov,0,-1)),order='F')
		#self.cov_CUDA.set_array(texarray)
		self.cov_CUDA.set_filter_mode(drv.filter_mode.POINT)

		col_mag = np.vstack((self.colour,self.magnitude))

		self.col_mag_CUDA = likelihood_functions.get_texref("data")
		drv.matrix_to_texref(np.float32(col_mag),self.col_mag_CUDA,order='C')
		self.col_mag_CUDA.set_filter_mode(drv.filter_mode.POINT)

		return


	def trim(self,isochrone,plot=True,plot_file='data_trim.png'):

		"""Apply some cuts to the data, based on the isochrone."""

		assert isinstance(isochrone,Isochrone)

		q = np.linspace(0.1,1,11)

		M_min = isochrone.mag_M_interp(self.magnitude_min)
		M_max = isochrone.mag_M_interp(self.magnitude_max)

		B_min_mag, B_min_colour = isochrone.binary(M_min,q)
		B_max_mag, B_max_colour = isochrone.binary(M_max,q)

		k_min = 0
		k_max = 0

		k_min_flag = B_min_mag[1:]-B_min_mag[:-1] > 0
		k_max_flag = B_max_mag[1:]-B_max_mag[:-1] > 0

		if k_min_flag.any():
			k_min = np.where(k_min_flag)[0][-1] + 1

		if k_max_flag.any():
			k_max = np.where(k_max_flag)[0][-1] + 1

		B_min_interp = PchipInterpolator(np.flip(B_min_mag[k_min:]),np.flip(B_min_colour[k_min:]))
		B_max_interp = PchipInterpolator(np.flip(B_max_mag[k_max:]),np.flip(B_max_colour[k_max:]))

		good_points = np.where( ( (self.magnitude > self.magnitude_min) & (self.magnitude < self.magnitude_max - 0.75) ) | \
								( (self.magnitude > self.magnitude_min - 0.75) & \
									(self.magnitude < self.magnitude_min) & \
									(self.colour > B_min_interp(self.magnitude) - 0.005 ) ) | \
								( (self.magnitude < self.magnitude_max) & \
									(self.magnitude > self.magnitude_max - 0.75) & \
									(self.colour < B_max_interp(self.magnitude) + 0.01 ) ) )[0]
		if plot:

			plt.figure(figsize=(4.5,6))
			ax = plt.axes()
			ax.scatter(self.colour,self.magnitude,c='k',s=0.2,alpha=0.6,marker='.')


		self.magnitude = self.magnitude[good_points]
		self.colour = self.colour[good_points]
		self.cov = self.cov[good_points]

		data_iso_mag = isochrone.colour_mag_interp(self.colour)
		good_points = np.where( (self.magnitude - data_iso_mag > -1.3) & (self.magnitude - data_iso_mag < 0.55) )[0]

		self.magnitude = self.magnitude[good_points]
		self.colour = self.colour[good_points]
		self.cov = self.cov[good_points]

		if plot:

			ax.scatter(self.colour,self.magnitude,c='b',s=0.2,marker='.')

			xmag = np.linspace(self.magnitude_min-0.5,self.magnitude_max+0.5,1001)
			ax.plot(isochrone.mag_colour_interp(xmag),xmag,'r-',alpha=1.0)

			ax.plot(isochrone.colour,isochrone.magnitude,'b--',alpha=1.0)

			xmag = np.linspace(self.magnitude_min+0.5,self.magnitude_max+0.5,1001)
			ax.plot(isochrone.mag_colour_interp(xmag),xmag - 1.30,'c-',alpha=1.0)
			xmag = np.linspace(self.magnitude_min-0.55,self.magnitude_max-0.55,1001)
			ax.plot(isochrone.mag_colour_interp(xmag),xmag + 0.55,'c-',alpha=1.0)

			ax.plot(B_min_colour,B_min_mag,'c-')
			ax.plot(B_max_colour,B_max_mag,'c-')

			ax.set_xlabel(self.colour_label)
			ax.set_ylabel(self.magnitude_label)
			ax.set_ylim([self.magnitude_max+1,self.magnitude_min-1])
			ax.set_xlim([np.min(isochrone.mag_colour_interp(xmag))-0.25,np.max(isochrone.mag_colour_interp(xmag))+0.5])
			plt.savefig(plot_file)


		return



####################################################################################################


class Isochrone():

	"""Class to contain an isochrone and its methods."""

	def __init__(self,isochrone_dict,colour_correction_data=None):

		"""
		Set up an Isochrone instance.


		Inputs:

			isochrone_dict:				(dictionary) with entries:
										file : (string) data file name
										column_mag: (int) column number in file corresponding to magnitude
										column_colour_blue: (int) column number in file for blue component of colour  
										column_colour_red: (int) column number in file for red component of colour  
										column_mass: (int) column number in file for mass
										magnitude_offset: (float) add to isochrone magnitude to match data
										colour_offset: (float) add to isochrone colour to match data
										magnitude_min : (float) minimum magnitude for main sequence
										magnitude_max : (float) maximum magnitude for main sequence

			colour_correction_data:		(Data) Data instance used to correct isochrone colour

		"""

		print('isochrone definition:',isochrone_dict)

		iso_data = np.loadtxt(isochrone_dict['file'])

		# cut to exclude white dwarfs
		cut = np.where((iso_data[:,isochrone_dict['column_blue']]-iso_data[:,isochrone_dict['column_red']])>0.25)[0]
		iso_data = iso_data[cut]

		self.magnitude = iso_data[:,isochrone_dict['column_mag']] + isochrone_dict['magnitude_offset']
		self.colour = iso_data[:,isochrone_dict['column_blue']] - iso_data[:,isochrone_dict['column_red']] + isochrone_dict['colour_offset']
		self.colour_offset = isochrone_dict['colour_offset']


		if colour_correction_data is not None:

			self.colour_correction = self.colour_correction_interpolator(colour_correction_data)

		else:

			self.colour_correction = lambda x: x*0.0

		iso_red = iso_data[:,isochrone_dict['column_red']]
		iso_blue = iso_data[:,isochrone_dict['column_blue']]
		iso_M = iso_data[:,isochrone_dict['column_mass']]

		pts = np.where((self.magnitude > isochrone_dict['magnitude_min']) & (self.magnitude < isochrone_dict['magnitude_max']))[0]

		ind = np.argsort(self.magnitude[pts])
		self.mag_M_interp = PchipInterpolator(self.magnitude[pts][ind],iso_M[pts][ind])
		self.M_mag_interp = PchipInterpolator(iso_M[pts],self.magnitude[pts])
		self.M_red_interp = PchipInterpolator(iso_M[pts],iso_red[pts])
		self.M_blue_interp = PchipInterpolator(iso_M[pts],iso_blue[pts])
		colour = self.colour[pts]+self.colour_correction(self.magnitude[pts])
		ind = np.argsort(colour)
		self.colour_mag_interp = PchipInterpolator(colour[ind],self.magnitude[pts][ind])
		ind = np.argsort(self.magnitude[pts])
		self.mag_colour_interp = PchipInterpolator(self.magnitude[pts][ind],colour[ind])

		return


	def colour_correction_interpolator(self,data,plot=True,plot_file='colour_correction.png'):

		"""Return a function that computes a colour-correction (as a function of magnitude) to be added to the
		isochrone in order to match the main-sequence ridge line."""

		assert isinstance(data,Data)

		index = np.argsort(self.magnitude)

		iso_colour_interp = PchipInterpolator(self.magnitude[index],self.colour[index])
		data_delta = data.colour - iso_colour_interp(data.magnitude)

		nbins = np.int(2*(data.magnitude_max - data.magnitude_min) + 0.5)

		y = np.empty(nbins)

		edges = np.linspace(data.magnitude_min,data.magnitude_max,nbins+1)
		centres = 0.5*(edges[1:]+edges[:-1])

		for i in range(nbins):

			pts = np.where((data.magnitude > edges[i]) & (data.magnitude <= edges[i+1]))[0]

			# Construct a histogram and find the maximum of a parabola that passes through the maximum and the point on each side.

			med_delta = np.median(data_delta[pts])
			bin_edges = np.linspace(med_delta-0.1,med_delta+0.1,21)
			h, h_edges = np.histogram(data_delta[pts],bins=bin_edges)
			j = np.argmax(h)
			htop = h[j-1:j+2]
			xtop = 0.5*(h_edges[j-1:j+2]+h_edges[j:j+3])
			A = np.vstack((xtop**2,xtop,np.ones_like(xtop))).T
			c = np.linalg.solve(A,htop)
			y[i] = -0.5*c[1]/c[0]

		delta_interp = PchipInterpolator(centres,y)

		if plot:

			plt.figure(figsize=(4.5,6))
			ax = plt.axes()
			ax.scatter(data.colour,data.magnitude,marker='.',c='k',s=0.2)
			xmag = np.linspace(data.magnitude_min,data.magnitude_max,1001)
			ax.plot(iso_colour_interp(xmag),xmag,'b--',alpha=0.7)
			ax.plot(iso_colour_interp(xmag)+delta_interp(xmag),xmag,'r-',alpha=0.7)
			ax.set_xlabel(data.colour_label)
			ax.set_ylabel(data.magnitude_label)
			ax.set_ylim([data.magnitude_max+1,data.magnitude_min-1])
			ax.set_xlim([np.min(iso_colour_interp(xmag))-0.25,np.max(iso_colour_interp(xmag)+delta_interp(xmag))+0.5])
			plt.savefig(plot_file)

		return delta_interp


	def mag_to_flux(self,m,c=20):
		F = 10**(0.4*(c-m))
		return F


	def flux_to_mag(self,F,c=20):
		m = c - 2.5*np.log10(F)
		return m


	def binary(self,M,q):

		#Returns the magnitude and colour for a binary system with primary mass M and mass ratio q.

		M1 = M
		M2 = q*M

		mag1 = self.M_mag_interp(M1)
		mag2 = self.M_mag_interp(M2)

		blue1 = self.M_blue_interp(M1)
		blue2 = self.M_blue_interp(M2)

		red1 = self.M_red_interp(M1)
		red2 = self.M_red_interp(M2)

		mag = self.flux_to_mag(self.mag_to_flux(mag1) + self.mag_to_flux(mag2))
		blue = self.flux_to_mag(self.mag_to_flux(blue1) + self.mag_to_flux(blue2))
		red = self.flux_to_mag(self.mag_to_flux(red1) + self.mag_to_flux(red2))

		return mag, blue - red + self.colour_correction(mag) + self.colour_offset



####################################################################################################


class PlotUtils():


	def plot_q_distribution(fitter,samples,weights=None,ax=None,n_plot_samples=1000,save_figure=True,plot_file='q_dist.png'):

		"""Plot the implied binary mass-ratio distributiion function for n_plot_samples drawn randomly from samples."""

		import random

		assert isinstance(fitter,CMDFitter)

		assert fitter.q_model == 'legendre'

		if ax is None:
			ax = plt.axes()

		q = np.linspace(0,1,101)

		ns = samples.shape[0]

		if weights is not None:
			weights = weights.tolist()

		for i in range(n_plot_samples):
			p = fitter.default_params.copy()
			ind = random.choices(range(ns),k=1,weights=weights)
			p[np.where(fitter.freeze == 0)] = samples[ind]
			args = p[3:9].tolist()
			args.append(fitter.M_ref)
			ax.plot(q,fitter.q_distribution(q,args),'b-',alpha=0.01)

		ax.set_ylim((0,4))
		ax.set_xlim((0,1))

		ax.set_xlabel(r'$q$')
		ax.set_ylabel(r'$P(q)$')

		if save_figure:
			plt.savefig(plot_file)

		return ax


	def plot_prior_q_distribution(fitter,ax=None,n_plot_samples=1000,save_figure=True,plot_file='q_dist.png'):

		"""Plot the implied binary mass-ratio distributiion function for n_plot_samples drawn randomly from samples."""

		import random

		assert isinstance(fitter,CMDFitter)

		assert fitter.q_model == 'legendre'

		if ax is None:
			ax = plt.axes()

		q = np.linspace(0,1,101)


		for i in range(n_plot_samples):

			x = np.random.rand(13)
			p = fitter.prior_transform(x)
			args = p[3:9].tolist()
			args.append(fitter.M_ref)
			ax.plot(q,fitter.q_distribution(q,args),'b-',alpha=0.01)

		ax.set_ylim((0,4))
		ax.set_xlim((0,1))

		ax.set_xlabel(r'$q$')
		ax.set_ylabel(r'$P(q)$')

		if save_figure:
			plt.savefig(plot_file)

		return ax


	def plot_fb_q(fitter,samples,weights=None,ax=None,save_figure=True,plot_file='fb_q.png'):

		"""Using all samples, plot the implied binary mass-fraction for q' > q along with its 1- and 2-sigma uncertainty."""

		from statsmodels.stats.weightstats import DescrStatsW

		assert isinstance(fitter, CMDFitter)

		assert fitter.q_model == 'legendre'

		if ax is None:
			ax = plt.axes()

		q = np.linspace(0,1,101)

		sig1 = 0.5 * 68.27
		sig2 = 0.5 * 95.45

		yq1 = np.zeros(101)
		yq2 = np.zeros(101)
		yq3 = np.zeros(101)
		yq4 = np.zeros(101)
		yq5 = np.zeros(101)

		for j in range(101):

			y = np.zeros(len(samples))

			for i in range(len(samples)):

				p = fitter.default_params.copy()
				p[fitter.freeze == 0] = samples[i]
				y[i] = p[9] * ((1.0  - fitter.int_sl_0(q[j])) + \
						p[3]*(0.0 - fitter.int_sl_1(q[j])) + \
						p[4]*(0.0 - fitter.int_sl_2(q[j])) + \
						p[5]*(0.0 - fitter.int_sl_3(q[j])) )

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

		ax.set_xlim((0,1))
		ax.set_ylim((0,1.0))

		ax.tick_params(axis='y',which='both',direction='in',right=True)
		ax.tick_params(axis='x',which='both',direction='in',top=True)

		if save_figure:
			plt.savefig(plot_file)

		return ax


	def plot_prior_fb_q(fitter,n_samples=1000,ax=None,save_figure=True,plot_file='fb_q.png'):

		"""Using all samples, plot the implied binary mass-fraction for q' > q along with its 1- and 2-sigma uncertainty."""

		from statsmodels.stats.weightstats import DescrStatsW

		assert isinstance(fitter, CMDFitter)

		assert fitter.q_model == 'legendre'

		if ax is None:
			ax = plt.axes()

		q = np.linspace(0,1,101)

		sig1 = 0.5 * 68.27
		sig2 = 0.5 * 95.45

		yq1 = np.zeros(101)
		yq2 = np.zeros(101)
		yq3 = np.zeros(101)
		yq4 = np.zeros(101)
		yq5 = np.zeros(101)

		x = np.random.rand(n_samples,13)

		y = np.zeros(n_samples)

		for j in range(101):

			for i in range(n_samples):
		
				p = fitter.prior_transform(x[i])

				y[i] = p[9] * ((1.0  - fitter.int_sl_0(q[j])) + \
						p[3]*(0.0 - fitter.int_sl_1(q[j])) + \
						p[4]*(0.0 - fitter.int_sl_2(q[j])) + \
						p[5]*(0.0 - fitter.int_sl_3(q[j])) )

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

		ax.set_xlim((0,1))
		ax.set_ylim((0,1.0))

		ax.tick_params(axis='y',which='both',direction='in',right=True)
		ax.tick_params(axis='x',which='both',direction='in',top=True)

		if save_figure:
			plt.savefig(plot_file)

		return ax


	def plot_realisation(fitter,params,plot_file='realisation.png'):

		"""Plot the data CMD and 3 comparative random realisations of the model implied by params."""

		assert isinstance(fitter, CMDFitter)

		n = len(fitter.data.magnitude)

		fraction_good = 1.0 - params[-3]
		fraction_single = (1.0-params[-4]-params[-3])/fraction_good
		fraction_binary = 1.0 - fraction_single

		n_single = int(round(fraction_single * n))
		n_binary = int(n - n_single)

		xmag = np.linspace(fitter.data.magnitude_min,fitter.data.magnitude_max,1001)

		fig, ax = plt.subplots(1,4,figsize=(16,6))

		ax[0].scatter(fitter.data.colour,fitter.data.magnitude,s=1,c='k')
		ax[0].plot(fitter.iso.mag_colour_interp(xmag),xmag,'g-',alpha=0.6)
		ax[0].grid()
		ax[0].set_xlabel(fitter.data.colour_label)
		ax[0].set_ylabel(fitter.data.magnitude_label)
		ax[0].set_ylim([fitter.data.magnitude_max+1,fitter.data.magnitude_min-1])

		for i in range(1,4):

		    mag, colour = fitter.model_realisation(params,n,add_observational_scatter=True)
		    ax[i].scatter(colour[:n_single:],mag[:n_single],s=1,color='b')
		    ax[i].scatter(colour[n_single:],mag[n_single:],s=1,color='r')
		    ax[i].plot(fitter.iso.mag_colour_interp(xmag),xmag,'g-',alpha=0.6)
		    ax[i].grid()
		    ax[i].set_xlabel(fitter.data.colour_label)
		    ax[i].set_ylabel(fitter.data.magnitude_label)
		    ax[i].set_ylim([fitter.data.magnitude_max+1,fitter.data.magnitude_min-1])


		plt.savefig(plot_file)





####################################################################################################



class CMDFitter():


	def __init__(self,json_file=None,data=None,isochrone=None,trim_data=True,q_model='legendre'):


		"""
		Set up a CMDFitter instance. This can be defined by providing a json-format definition file, or separate Data and Isochrone objects.


		Inputs:

			json_file:		(string) the name of a json-format file with the data and isochrone defintions

			data:			(Data) input CMD data instance

			isochrone:		(Isochrone) input Isochrone instance

			trim_data:		(boolean) data will be filtered if true

			q_model:		(string) functional form for q distribution. Must be "power" or "legendre".

		"""

		self.version = 6.0

		self.citation = "Albrow, M.D., Ulusele, I.H., 2022, MNRAS, submitted"


		assert q_model in ['power','legendre']

		if q_model == 'power':
			self.ndim = 10
			self.labels = [r"$\log_{10} k$", r"$M_0$", r"$\gamma$",  r"$\beta$", r"$\alpha$", r"$B$", r"$f_B$", r"$f_O$", r"$h_0$", r"$h_1$"]
			self.default_params = np.array([2.2,0.55,0.0,0.0,1.0,0.0,0.5,0.01,1.0,0.00])

		if q_model == 'legendre':
			self.ndim = 13
			self.labels = [r"$\log_{10} k$", r"$M_0$", r"$\gamma$",  r"$a_1$", r"$a_2$", r"$a_3$", \
						r"$\dot{a}_1$", r"$\dot{a}_2$", r"$\dot{a}_3$",r"$f_B$", r"$f_O$", r"$h_0$", r"$h_1$"]
			self.default_params = np.array([2.4,0.55,0.006, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.35,0.001, 1.0,0.0])

		self.freeze = np.zeros(self.ndim)
		self.prefix = 'out_'

		# Options are 'power' and 'legendre' (actually a shifted-legendre basis)
		self.q_model = q_model

		# Shifted Legendre polynomials
		self.sl_0 = lambda x: x*0.0 + 1.0
		self.sl_1 = lambda x: 2.0*x - 1.0
		self.sl_2 = lambda x: 6.0*x**2 - 6.0*x + 1.0
		self.sl_3 = lambda x: 20.0*x**3 - 30.0*x**2 + 12.0*x - 1.0

		# Derivatives of shifted Legendre polynomials
		self.der_sl_0 = lambda x: 0.0
		self.der_sl_1 = lambda x: 2.0
		self.der_sl_2 = lambda x: 12.0*x - 6.0
		self.der_sl_3 = lambda x: 60.0*x**2 - 60.0*x + 12.0

		# Integrals of shifted Legendre polynomials
		self.int_sl_0 = lambda x: x
		self.int_sl_1 = lambda x: x**2 - x
		self.int_sl_2 = lambda x: 2.0*x**3 - 3.0*x**2 + x
		self.int_sl_3 = lambda x: 5.0*x**4 - 10.0*x**3 + 6.0*x**2 - x

		self.neginf = -np.inf


		# Exit now if no input provided
		if json_file is None and (data is None or isochrone is None):
			return

		# Setup from json-format description file
		if json_file is not None:

			with open(json_file) as file:
				data = json.load(file)
			
			data_description = data['data_description']
			iso_description = data['iso_description']
			
			data = Data(data_description)
			isochrone = Isochrone(iso_description, colour_correction_data=data)

		else:

			assert isinstance(data, Data)
			assert isinstance(isochrone, Isochrone)

		self.data = data

		self.iso = isochrone

		if trim_data:
			self.data.trim(self.iso)

		self.data.upload_to_GPU()

		self.mass_slice = np.array([self.iso.mag_M_interp(self.data.magnitude_max),self.iso.mag_M_interp(self.data.magnitude_min)])

		self.M_ref = np.mean(self.mass_slice)
		self.delta_M = np.abs(self.mass_slice[0] - self.M_ref)

		self.magnitude_ref = 0.5*(self.data.magnitude_min+self.data.magnitude_max)
		self.delta_mag = self.data.magnitude_max - self.magnitude_ref

		# mean colour, magnitude and dispersions for outlier distribution.

		self.outlier_description = np.ascontiguousarray([np.median(self.data.colour),self.magnitude_ref,0.75,4.0],dtype=np.float64)
		self.h_magnitude_ref = self.data.magnitude_min

		# set up basis functions for M and q

		self.n_bf = 50

		self.Mw = 0.01
		self.M0 = np.linspace(self.mass_slice[0]-0.2,self.mass_slice[1]-(self.Mw/2),self.n_bf)
		self.qw = 0.012
		self.q0 = np.linspace(self.qw*2,1,self.n_bf)

		self.qsigma = 0.0001
		self.qx = np.linspace(0.001,1,200)
		self.qbf = np.zeros([self.n_bf,200])
		for i in range(self.n_bf):
			self.qbf[i,:] = (1/(self.qw*(2*np.pi)**0.5))*np.exp(-((self.qx-self.q0[i])**2)/(2*self.qw**2)) 

		self.qA = np.zeros((self.n_bf,self.n_bf))
		for k in range(self.n_bf):
			for j in range(self.n_bf):
				self.qA[k,j] = np.sum(self.qbf[k,:]*self.qbf[j,:]/self.qsigma**2) 

		self.qAT = self.qA.T
		self.qAA = np.dot(self.qAT,self.qA)

		self.Msigma = 0.0001
		self.Mx = np.linspace(0.1,1.1,200)

		self.Mbf = np.zeros([self.n_bf,200])
		for i in range(self.n_bf):
			self.Mbf[i,:] = (1/(self.Mw*(2*np.pi)**0.5))*np.exp(-((self.Mx-self.M0[i])**2)/(2*self.Mw**2)) 

		self.MA = np.zeros((self.n_bf,self.n_bf))
		for k in range(self.n_bf):
			for j in range(self.n_bf):
				self.MA[k,j] = np.sum(self.Mbf[k,:]*self.Mbf[j,:]/self.Msigma**2) 

		self.MAT = self.MA.T
		self.MAA = np.dot(self.MAT,self.MA)

		D_ij = np.zeros([self.n_bf**2,2])
		S_ij = np.zeros([self.n_bf**2,2,2])
		width_matrix = np.zeros([2,2])
		for i in range(self.n_bf):
			for j in range(self.n_bf):
				jacob = self.jacobian(self.M0[i],self.q0[j]) 
				width_matrix[0,0] = self.Mw**2
				width_matrix[1,1] = self.qw**2
				mag, colour = self.iso.binary(self.M0[i],self.q0[j])
				D_ij[i+j*self.n_bf] = np.array([colour,mag])
				S_ij[i+j*self.n_bf] = np.dot(jacob,(np.dot(width_matrix,(jacob.T))))

		S_ij_shaped = S_ij.reshape(self.n_bf**2,4)


		# upload basis functions to GPU texture memory

		self.DMQ_CUDA = likelihood_functions.get_texref("DMQ")
		drv.matrix_to_texref(np.float32(D_ij),self.DMQ_CUDA,order='F')
		self.DMQ_CUDA.set_filter_mode(drv.filter_mode.POINT)

		self.SMQ_CUDA = likelihood_functions.get_texref("SMQ")
		drv.matrix_to_texref(np.float32(S_ij_shaped),self.SMQ_CUDA,order='F')
		self.SMQ_CUDA.set_filter_mode(drv.filter_mode.POINT)

		#

		self.likelihood = likelihood_functions.get_function("likelihood")

		self.emcee_walker_dispersion = 1.e-7

		return



	def M_distribution(self,x,log_k,x0,gamma,maxM):

		"""Evaluate mass function at x."""

		k = 10.0**log_k
		m = np.linspace(0.1,1.1,1000)
		dm = m[1]-m[0]
		y = m**(-gamma) / (1.0 + np.exp(-k*(m-x0)))
		y[m>maxM] = 0.0
		normalM = 1.0 / (np.sum(y)*dm)
		y = normalM * x**(-gamma) / (1.0 + np.exp(-k*(x-x0)))
		y[x>maxM] = 0.0
		return y


	def M_distribution_sampler(self,log_k,x0,gamma,maxM):

		"""Return a function that maps the range (0,1) onto the mass function.""" 

		m = np.linspace(0.1,maxM,1000)

		y = self.M_distribution(m,log_k,x0,gamma,maxM)

		y_cumulative = np.cumsum(y)/np.sum(y)
		pts = np.where(y_cumulative>1.e-50)[0]
		return PchipInterpolator(y_cumulative[pts],m[pts])


	def q_distribution(self,x,params):

		"""Evaluate the binary mass-ratio distribution function at x."""

		assert self.q_model in ['power','legendre']

		if self.q_model == 'power':
			beta,alpha,B,M = params
			power = alpha + beta*(M-self.M_ref)
			return (alpha + beta*(M-self.M_ref) + 1.0)*(1.0-B)*x**power + B

		if self.q_model == 'legendre':
			a1, a2, a3, a1_dot, a2_dot,a3_dot, M = params
			dM = M-self.M_ref
			return self.sl_0(x) + (a1+a1_dot*dM)*self.sl_1(x) + (a2+a2_dot*dM)*self.sl_2(x) + (a3+a3_dot*dM)*self.sl_3(x)


	def q_distribution_sampler(self,params):

		"""Return a function that maps the range (0,1) onto the binary mass-ratio distribution function.""" 

		assert self.q_model in ['power','legendre']

		q = np.linspace(0,1,1001)

		y = self.q_distribution(q,params)			

		y_cumulative = np.cumsum(y)+np.arange(len(y))*1.e-6

		return PchipInterpolator(y_cumulative/y_cumulative[-1],q)


	def compute_observational_scatter(self,mag,nbins=10):

		"""Return an average covariance at magnitude mag, based on the data."""

		from scipy.optimize import curve_fit

		x1 = np.linspace(self.data.magnitude_min,self.data.magnitude_max,nbins+1)

		xm = np.zeros(nbins)

		ycov = np.zeros((nbins,2,2))

		cov = np.zeros((len(mag),2,2))

		for i in range(nbins):
			pts = np.where((self.data.magnitude >= x1[i]) & (self.data.magnitude < x1[i+1]))
			ycov[i] = np.median(self.data.cov[pts],axis=0)
			xm[i] = 0.5*(x1[i]+x1[i+1])

		for i in range(2):
			for j in range(2):
				yoffset = 0.99*np.min(ycov[:,i,j])
				f, _ = curve_fit(lambda t,a,b: a*np.exp(b*t),  xm,  ycov[:,i,j]-yoffset)
				cov[:,i,j] = f[0]*np.exp(f[1]*mag) + yoffset

		return cov


	def model_realisation(self,p,n,add_observational_scatter=True):

		"""Compute a random (magnitude, colour) realisation of the model paramterised by p (excluding outliers) for n stars.""" 

		assert self.q_model in ['power','legendre']

		if self.q_model == 'power':
			log_k, M0, gamma, beta, alpha, B, fb, fo, h0, h1 = p

		if self.q_model == 'legendre':
			log_k, M0, gamma, a1, a2, a3, a1_dot, a2_dot,a3_dot, fb, fo, h0, h1 = p

		fraction_good = 1.0 - fo
		fraction_single = (1.0-fb-fo)/fraction_good
		fraction_binary = 1.0 - fraction_single

		n_single = int(round(fraction_single * n))
		n_binary = int(n - n_single)

		M_sampler = self.M_distribution_sampler(log_k,M0,gamma,self.mass_slice[1])
		M1 = M_sampler(np.random.rand(n))

		q = np.zeros_like(M1)

		for i in range(n_binary):

			if self.q_model == 'power':
				q_sampler = self.q_distribution_sampler([alpha,beta,B,M1[n_single+i]])
			if self.q_model == 'legendre':
				q_sampler = self.q_distribution_sampler([a1,a2,a3,a1_dot,a2_dot,a3_dot,M1[n_single+i]])

			q[n_single+i] = q_sampler(np.random.rand())
			
		mag, colour = self.iso.binary(M1,q)

		if add_observational_scatter:

			h = h0 + h1*(mag-self.h_magnitude_ref)

			cov = self.compute_observational_scatter(mag)

			for i in range(len(mag)):
				z = np.random.multivariate_normal(mean=np.zeros(2), cov=h[i]**2*cov[i], size=1)
				colour[i] += z[0][0]
				mag[i] += z[0][1]

		return mag, colour


	def jacobian(self,M,q):

		jacob = np.zeros((2,2))

		mag_fixq, colour_fixq = self.iso.binary(self.M0,q)
		mag_fixM, colour_fixM = self.iso.binary(M,self.q0)

		mag_q = PchipInterpolator(self.q0,mag_fixM)
		mag_M = PchipInterpolator(self.M0,mag_fixq)
		colour_q = PchipInterpolator(self.q0,colour_fixM)
		colour_M = PchipInterpolator(self.M0,colour_fixq)

		jacob[0,0] = colour_M(M,1)
		jacob[0,1] = colour_q(q,1)
		jacob[1,0] = mag_M(M,1)
		jacob[1,1] = mag_q(q,1)

		return jacob



	def M_gauss(self,Mparams):
		
		"""Return the (positive) coefficients for mapping the mass basis functions onto the mass function."""

		log_k, M0, gamma = Mparams

		My = self.M_distribution(self.Mx,log_k, M0, gamma, self.mass_slice[1])
		
		Mb = np.zeros(self.n_bf)
		for k in range(self.n_bf):
			Mb[k] = np.sum(My*self.Mbf[k,:]/self.Msigma**2)
		
		Ma, resid = nnls(self.MA,Mb)

		norm_c = np.sum(Ma)

		return Ma/norm_c


	def q_gauss(self,params):
				
		"""Return the (positive) coefficients for mapping the mass-ratio basis functions onto the mass-ratio distribution function."""

		qy = self.q_distribution(self.qx,params)

		qb = np.zeros(self.n_bf)
		for k in range(self.n_bf):
			qb[k] = np.sum(qy*self.qbf[k,:]/self.qsigma**2)
		
		qa, resid = nnls(self.qA,qb)

		norm_c = np.sum(qa)

		return qa/norm_c


	def precalc(self,params):
		
		"""Return the grid of (M,q) bais function coefficients."""

		assert self.q_model in ['power','legendre']


		PMQ = np.zeros(self.n_bf**2)
		Ma = self.M_gauss(params[:3])

		if (np.abs(params[4]) < 1.e-5) or (self.q_model == 'legendre'):

			if self.q_model == 'power':
				args = params[3:6].tolist()
				args.append(self.M0[0])
				qa = self.q_gauss(args)
			if self.q_model == 'legendre':
				args = params[3:9].tolist()
				args.append(self.M0[0])
				qa = self.q_gauss(args)

			for i in range (self.n_bf):
				for j in range(self.n_bf):
					PMQ[i+j*self.n_bf] = Ma[i]*qa[j]

		else:

			for i in range (self.n_bf):
				qa = self.q_gauss(params[3:6],self.M0[i])
				for j in range(self.n_bf):
					PMQ[i+j*self.n_bf] = Ma[i]*qa[j]

		return Ma, PMQ


	def lnlikelihood(self,params):

		"""Call the external CUDA function to evaluate the likelihood function."""
		
		p = self.default_params.copy()
		p[self.freeze==0] = params

		assert self.q_model in ['power','legendre']

		if self.q_model == 'power':

			log_k, M0, gamma, beta, alpha, B, fb, fo, h0, h1 = p

		if self.q_model == 'legendre':

			log_k, M0, gamma, a1, a2, a3, a1_dot, a2_dot, a3_dot, fb, fo, h0, h1 = p

			# Check that the parameters generate a positive q distribution for all masses
			for MM in np.linspace(self.mass_slice[0],self.mass_slice[1],31).tolist():
				args = p[3:9].tolist()
				args.append(MM)
				q_dist_test = self.q_distribution(np.linspace(0.0,1.0,101),args)
				if np.min(q_dist_test) < 0.0:
					return self.neginf


		P_i, PMQ = self.precalc(p)


		c_P_i = np.ascontiguousarray(P_i.astype(np.float64))
		c_PMQ = np.ascontiguousarray(PMQ.astype(np.float64))
		
		blockshape = (int(256),1, 1)
		gridshape = (len(self.data.magnitude), 1)
		#gridshape = (1, 1)

		lnP_k = np.zeros(len(self.data.magnitude)).astype(np.float64)

		likelihood(drv.In(c_P_i), drv.In(c_PMQ), drv.In(self.outlier_description),np.float64(h0), np.float64(h1), np.float64(self.h_magnitude_ref),np.float64(fo), np.float64(fb), drv.InOut(lnP_k), block=blockshape, grid=gridshape)

		lnP = np.sum(lnP_k)

		self.lnP_k = lnP_k


		if not(np.isfinite(lnP)):
			with open(self.prefix+'.err', 'a') as f:
				f.write(np.array2string(params,max_line_width=1000).strip('[]\n')+'\n')
			return self.neginf

		return lnP


	def neglnlikelihood(self,params):
		return -self.lnlikelihood(params)


	def ln_prior(self,params):

		from scipy.stats import norm, truncnorm

		p = self.default_params.copy()
		p[self.freeze==0] = params

		assert self.q_model in ['power','legendre']

		if self.q_model == 'power':

			log_k, M0, gamma, beta, alpha, B, fb, fo, h0, h1 = p

			if fb < 0.02 or fb > 0.95 or fo < 0.0 or fo > 0.05 or B < 0.0 or B > (1.0 + 1.0/(alpha+beta*self.delta_M)):
				return self.neginf 

			log_h = np.log10(h0)

			prior = norm.pdf(log_k,loc=2.0,scale=0.3) * norm.pdf(M0,loc=self.mass_slice[0]+0.1*self.delta_M, scale=0.1*self.delta_M) * norm.pdf(gamma,loc=0.0,scale=1.0) * norm.pdf(beta,loc=0.0,scale=2.0) * \
						truncnorm.pdf(alpha + np.abs(beta)*self.delta_M,0.0,20.0,loc=0.0,scale=3.0) * norm.pdf(log_h,loc=0.1,scale=0.3) * \
						truncnorm.pdf(h1,0.0,2.0,loc=0.0,scale=0.4*h0)

		if self.q_model == 'legendre':

			log_k, M0, gamma, a1, a2, a3,  a1_dot, a2_dot, a3_dot, fb, fo, h0, h1 = p

			if fb < 0.02 or fb > 0.95 or fo < 0.0 or fo > 0.05:
				return self.neginf 

			log_h = np.log10(h0)

			prior = norm.pdf(log_k,loc=2.0,scale=0.3) * norm.pdf(M0,loc=self.mass_slice[0]+0.1*self.delta_M, scale=0.1*self.delta_M) * norm.pdf(gamma,loc=0.0,scale=1.0) * \
							norm.pdf(a1,loc=0.0,scale=2.0) * \
							norm.pdf(a2,loc=0.0,scale=2.0) * norm.pdf(a3,loc=0.0,scale=2.0) *  \
							norm.pdf(a1_dot,loc=0.0,scale=0.1/self.delta_M) * \
							norm.pdf(a2_dot,loc=0.0,scale=0.1/self.delta_M) * norm.pdf(a3_dot,loc=0.0,scale=0.1/self.delta_M) * \
							norm.pdf(log_h,loc=0.1,scale=0.3) * truncnorm.pdf(h1,0.0,2.0*h0,loc=0.0,scale=0.4*h0)

		return np.log(prior)


	def prior_transform(self,u):

		from scipy.stats import norm, truncnorm

		assert self.q_model in ['power','legendre']

		x = self.default_params.copy()

		if self.q_model == 'power':


			# params are log k, M0, gamma, beta, alpha, B, fb, fo, h0, h1

			i = 0

			if not self.freeze[0]:
				# log k
				x[0] = norm.ppf(u[i], loc=2.0, scale=0.3)
				i += 1
			if not self.freeze[1]:
				# M0
				x[1] = norm.ppf(u[i], loc=0.55, scale=0.01)
				i += 1
			if not self.freeze[2]:
				# gamma
				x[2] = norm.ppf(u[i], loc=0.0, scale=1.0)
				i += 1

			if not self.freeze[3]:
				# beta
				x[3] = norm.ppf(u[i], loc=0.0, scale=2.0)
				i += 1
			if not self.freeze[4]:
				# alpha + beta * delta M
				abm = truncnorm.ppf(u[i], 0.0, 20.0, loc=0.0, scale=3.0)
				x[4] = abm - np.abs(x[3])*self.delta_M
				i += 1
			if not self.freeze[5]:
				# B
				x[5] = (1.0 + 1.0/(x[4]+x[3]*self.delta_M))*u[i]
				i += 1

			if not self.freeze[6]:
				# f_B
				x[6] = 0.93*u[i] + 0.02
				i += 1
			if not self.freeze[7]:
				# f_O
				x[7] = 0.05*u[i]
				i += 1

			if not self.freeze[8]:
				# log h0
				logh = norm.ppf(u[i], loc=0.1, scale=0.3)
				x[8] = 10.0**logh
				i += 1
			if not self.freeze[9]:
				# h1
				x[9] = truncnorm.ppf(u[i], 0.0, 2.0*x[8], loc=0.0, scale=0.4*x[8])
				i += 1

		if self.q_model == 'legendre':


			# params are log k, M0, gamma, a1, a2, a3, a1_dot, a2_dot, a3_dot, fb, fo, h0, h1

			i = 0

			if not self.freeze[0]:
				# log k
				x[0] = norm.ppf(u[i], loc=2.0, scale=0.3)
				i += 1
			if not self.freeze[1]:
				# M0
				x[1] = norm.ppf(u[i], loc=self.mass_slice[0]+0.5*self.delta_M, scale=0.5*self.delta_M)
				i += 1
			if not self.freeze[2]:
				# gamma
				x[2] = norm.ppf(u[i], loc=0.0, scale=1.0)
				i += 1

			if not self.freeze[3]:
				# a1
				x[3] = norm.ppf(u[i], loc=0.0, scale=2.0)
				i += 1
			if not self.freeze[4]:
				# a2
				x[4] = norm.ppf(u[i], loc=0.0, scale=2.0)
				i += 1
			if not self.freeze[5]:
				# a3
				x[5] = norm.ppf(u[i], loc=0.0, scale=2.0)
				i += 1

			if not self.freeze[6]:
				# a1
				x[6] = norm.ppf(u[i], loc=0.0, scale=0.1/self.delta_M)
				i += 1
			if not self.freeze[7]:
				# a2
				x[7] = norm.ppf(u[i], loc=0.0, scale=0.1/self.delta_M)
				i += 1
			if not self.freeze[8]:
				# a3
				x[8] = norm.ppf(u[i], loc=0.0, scale=0.1/self.delta_M)
				i += 1

			if not self.freeze[9]:
				# f_B
				x[9] = 0.93*u[i] + 0.02
				i += 1
			if not self.freeze[10]:
				# f_O
				x[10] = 0.05*u[i]
				i += 1

			if not self.freeze[11]:
				# log h0
				logh = norm.ppf(u[i], loc=0.1, scale=0.3)
				x[11] = 10.0**logh
				i += 1
			if not self.freeze[12]:
				# h1
				x[12] = truncnorm.ppf(u[i], 0.0, 2.0, loc=0.0, scale=0.4*x[11])
				i += 1

		y = x[self.freeze==0]

		return y


	def lnprob(self,params):

		lp = self.ln_prior(params)

		if not np.isfinite(lp):
			return self.neginf

		lnp = self.lnlikelihood(params)

		if not np.isfinite(lnp):
			return self.neginf

		return lp + lnp


	def neglnprob(self,params):
		return -self.lnprob(params)



	def emcee_sample(self,params_guess,n_steps_per_save=20,n_saves=100,n_walkers=100,prefix='em_',target='lnprob'):

		import emcee 
		import corner

		assert target in ['lnprob','lnlikelihood']

		emcee_target = self.lnprob
		if target == 'lnlikelihood':
			emcee_target = self.lnlikelihood

		self.prefix = prefix

		ndim = int(self.ndim - np.sum(self.freeze))

		labels = [self.labels[i] for i in range(self.ndim) if self.freeze[i] == 0]

		# Set the initial state for the walkers to be a tight distribution around the guess
		state = [params_guess[self.freeze==0] + self.emcee_walker_dispersion*np.random.randn(ndim) for i in range(n_walkers)]

		# Run the sampler    
		n_steps = n_saves * n_steps_per_save

		sampler = emcee.EnsembleSampler(n_walkers, ndim, emcee_target)

		for save in range(n_saves):

			state, lnp , _ = sampler.run_mcmc(state, n_steps_per_save, progress=True, skip_initial_state_check=True)

			samples = sampler.chain
			np.save(prefix+'sampleschain.npy',np.asarray(samples))
			samples = sampler.flatchain
			np.save(prefix+'samples.npy',np.asarray(samples))
			ln_prob = sampler.get_log_prob()
			np.save(prefix+'lnprob.npy',np.asarray(ln_prob))
			flatlnprob = sampler.get_log_prob(flat=True)
			np.save(prefix+'flatlnprob.npy',np.asarray(flatlnprob))


			plt.figure(figsize=(8,11))
			subplots_adjust(hspace=0.0001)

			for i in range(ndim):
				plt.subplot(ndim+1,1,i+1)
				plt.plot(sampler.chain[:,:,i].T, '-', color='k', alpha=0.3)
				plt.ylabel(labels[i])
			plt.subplot(ndim+1,1,ndim+1)
			plt.plot(ln_prob, '-', color='r', alpha=0.3)
			plt.ylabel('ln P')
			plt.savefig(prefix+'chains.png')

			plt.close()

		# The result
		good_samples = sampler.chain[:,-n_steps//2:,:].reshape(n_walkers*n_steps//2,ndim)
		params_out = np.median(good_samples,axis=0)
		uncertainty_params_out = 0.5*(np.percentile(good_samples,84,axis=0) - np.percentile(good_samples,16,axis=0))

		print("paramaters =", params_out)
		print("Uncertainty in parameters =", uncertainty_params_out)

		# Covariance plots

		corner.corner(good_samples, labels=labels,show_titles=True,levels=[0.68,0.95],
							quantiles=[0.16, 0.5, 0.84])
		plt.savefig(prefix+'corner.png')


	def dynesty_sample(self,prefix='dy_'):

		from dynesty import NestedSampler
		from dynesty import plotting as dyplot

		self.prefix = prefix

		ndim = int(self.ndim - np.sum(self.freeze))

		labels = [self.labels[i] for i in range(self.ndim) if self.freeze[i] == 0]

		sampler = NestedSampler(self.lnlikelihood, self.prior_transform, ndim)

		sampler.run_nested()

		res = sampler.results

		samples, weights = res.samples, np.exp(res.logwt - res.logz[-1])

		np.save(prefix+'samples.npy',samples)
		np.save(prefix+'weights.npy',weights)

		res.summary()

		fig, axes = dyplot.runplot(res)
		plt.savefig(prefix+'summary.png')


		fig, axes = dyplot.traceplot(res, show_titles=True,trace_cmap='viridis',
		                         connect=True,connect_highlight=range(5),labels=labels)
		plt.savefig(prefix+'trace.png')


		fig, axes = plt.subplots(ndim, ndim, figsize=(15, 15))
		axes = axes.reshape((ndim, ndim))  # reshape axes
		fg, ax = dyplot.cornerplot(res, color='blue',show_titles=True,max_n_ticks=3,labels=labels,
		                        quantiles=None,fig=(fig,axes))
		plt.savefig(prefix+'corner.png')


	def ultranest_sample(self,prefix='un_',stepsampler=False):

		import ultranest
		import ultranest.stepsampler

		self.neginf = sys.float_info.min

		self.prefix = prefix

		labels = [self.labels[i] for i in range(self.ndim) if self.freeze[i] == 0]

		output_dir = prefix+'output'

		sampler = ultranest.ReactiveNestedSampler(labels, self.lnlikelihood, self.prior_transform,log_dir=output_dir,resume='overwrite')

		if stepsampler:

			nsteps = 5*(len(self.freeze) - sum(self.freeze))
			sampler.stepsampler = ultranest.stepsampler.SliceSampler(nsteps=nsteps,generate_direction=ultranest.stepsampler.generate_mixture_random_direction)


		result = sampler.run(min_num_live_points=400)

		sampler.print_results()

		plt.figure()
		sampler.plot_run()
		plt.savefig(prefix+'plot_run.png')
		plt.close()

		plt.figure()
		sampler.plot_trace()
		plt.savefig(prefix+'plot_trace.png')
		plt.close()

		plt.figure()
		sampler.plot_corner()
		plt.savefig(prefix+'plot_corner.png')
		plt.close()





