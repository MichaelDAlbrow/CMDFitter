import sys
import os
import numpy as np
from scipy.interpolate import PchipInterpolator


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Data import Data

class Isochrone():

	"""Class to contain an isochrone and its methods."""

	def __init__(self,isochrone_dict,isochrone_correction_data=None,correction_type='colour'):

		"""
		Set up an Isochrone instance.


		Inputs:

			isochrone_dict:				(dictionary) with entries:
										file : (string) data file name
										column_mag: (int) column number in file corresponding to magnitude
										column_colour_blue: (int) column number in file for blue component of colour  
										column_colour_red: (int) column number in file for red component of colour  
										column_initial_mass: (int) column number in file for initial mass
										column_mass: (int) column number in file for mass
										magnitude_offset: (float) add to isochrone magnitude to match data
										colour_offset: (float) add to isochrone colour to match data
										magnitude_min : (float) minimum magnitude for main sequence
										magnitude_max : (float) maximum magnitude for main sequence

			isochrone_correction_data:		(Data) Data instance used to correct isochrone colour or magnitude

		"""

		print('isochrone definition:',isochrone_dict)

		iso_data = np.loadtxt(isochrone_dict['file'])

		if 'initial_mass' in isochrone_dict:
			q = np.where(iso_data[:,isochrone_dict['initial_mass']] - iso_data[:,isochrone_dict['star_mass']] < 1.e-2)[0]
			iso_data = iso_data[q]

		# cut to exclude white dwarfs
		cut = np.where((iso_data[:,isochrone_dict['column_blue']]-iso_data[:,isochrone_dict['column_red']])>0.0)[0]
		iso_data = iso_data[cut]

		self.magnitude = iso_data[:,isochrone_dict['column_mag']] + isochrone_dict['magnitude_offset']
		self.colour = iso_data[:,isochrone_dict['column_blue']] - iso_data[:,isochrone_dict['column_red']] + isochrone_dict['colour_offset']
		self.colour_offset = isochrone_dict['colour_offset']

		self.magnitude_min = isochrone_dict['magnitude_min']
		self.magnitude_max = isochrone_dict['magnitude_max']


		if isochrone_correction_data is not None:

			self.colour_correction = self.colour_correction_interpolator(isochrone_correction_data)
			self.magnitude_correction = self.magnitude_correction_interpolator(isochrone_correction_data)

		else:

			self.colour_correction = lambda x: x*0.0
			self.magnitude_correction = lambda x: x*0.0


		iso_red = iso_data[:,isochrone_dict['column_red']]
		iso_blue = iso_data[:,isochrone_dict['column_blue']]
		iso_M = iso_data[:,isochrone_dict['column_mass']]

		pts = np.where((self.magnitude > isochrone_dict['magnitude_min']) & (self.magnitude < isochrone_dict['magnitude_max']))[0]

		self.magnitude[pts] += self.magnitude_correction(self.colour[pts])

		ind = np.argsort(self.magnitude[pts])
		self.mag_M_interp = PchipInterpolator(self.magnitude[pts][ind],iso_M[pts][ind])

		iso_M_increasing = np.hstack((0.0,iso_M[pts]+1e-6*np.arange(len(pts))))

		self.M_mag_interp = PchipInterpolator(iso_M_increasing,np.hstack((self.magnitude[0]+10,self.magnitude[pts])))
		self.M_red_interp = PchipInterpolator(iso_M_increasing,np.hstack((iso_red[0]+10,iso_red[pts])))
		self.M_blue_interp = PchipInterpolator(iso_M_increasing,np.hstack((iso_blue[0]+16,iso_blue[pts])))


		assert correction_type in ['colour','magnitude']
		self.correction_type = correction_type

		if correction_type == 'colour':
			colour = self.colour[pts]+self.colour_correction(self.magnitude[pts])
		else:
			# This dummy call is to make the luminosity function
			colour = self.colour[pts]
			_ = self.colour_correction(self.magnitude[pts])

		mag = self.magnitude[pts]

		ind = np.argsort(colour)
		self.colour_mag_interp = PchipInterpolator(colour[ind],mag[ind])
		ind = np.argsort(mag)
		self.mag_colour_interp = PchipInterpolator(mag[ind],colour[ind])

		self.plot_luminosity_mass_functions(isochrone_correction_data)

		return

	@staticmethod
	def hist_peak(x):

		"""Make a histogram of x, fit a parabola to the peak, and return the location of the maximum."""

		med_x = np.median(x)


		bin_edges = np.linspace(med_x-0.1,med_x+0.1,21)

		h, h_edges = np.histogram(x,bins=bin_edges)
		if np.max(h) < 5:
			return -9999.0

		j = np.argmax(h)
		htop = h[j-1:j+2]
		try:
			xtop = 0.5*(h_edges[j-1:j+2]+h_edges[j:j+3])
		except:
			print('j,h_edges,h:',j,h_edges,h)
			raise

		A = np.vstack((xtop**2,xtop,np.ones_like(xtop))).T
		c = np.linalg.solve(A,htop)

		print('x, median, std, result', x, med_x, np.std(x),-0.5*c[1]/c[0])

		return -0.5*c[1]/c[0]


	@staticmethod
	def median_peak(x,std_cut=1.0,n_passes=3):

		xx = x.copy()
		for i in range(n_passes):
			mx = np.median(xx)
			sx = np.std(xx)
			p = np.where((xx > mx-std_cut*sx) & (xx<mx+std_cut*sx))[0]
			xx = xx[p]

		return np.mean(xx)


	def magnitude_correction_interpolator(self,data,plot=True,plot_file='magnitude_correction.png',plot_binary_sequence=True,plot_triple_sequence=True):

		"""Return a function that computes a magnitude-correction (as a function of colour) to be added to the
		isochrone in order to match the main-sequence ridge line."""

		from Data import Data

		assert isinstance(data,Data)

		index = np.argsort(self.magnitude)
		iso_colour_interp = PchipInterpolator(self.magnitude[index],self.colour[index])

		q = np.where(self.magnitude > data.magnitude_min)[0]
		index = np.argsort(self.colour[q])
		iso_mag_interp = PchipInterpolator(self.colour[q][index]+1.e-6*np.arange(len(index)),self.magnitude[q][index])
		data_delta = data.colour - iso_colour_interp(data.magnitude)

		nbins = np.int(1*(data.magnitude_max - data.magnitude_min+2) + 0.5)

		y = -9999*np.ones(nbins)
		luminosity_function = np.empty(nbins)

		edges = np.linspace(data.magnitude_min-1.0,data.magnitude_max+1.0,nbins+1)
		centres = 0.5*(edges[1:]+edges[:-1])

		print('edges',edges)

		for i in range(nbins):

			pts = np.where((data.magnitude > edges[i]) & (data.magnitude <= edges[i+1]))[0]

			print('centre mag, npts:',centres[i],len(pts))

			luminosity_function[i] = len(pts)

			if len(pts) > 4:

				#y[i] = self.hist_peak(data_delta[pts])
				y[i] = self.median_peak(data_delta[pts])

		good = y > -100
		iso_col = iso_colour_interp(centres[good])
		col = iso_col + y[good]

		print('col',col)

		delta_mag = centres[good] - iso_mag_interp(col)

		print('mag, iso_col, col, delta_mag')
		print(centres[good])
		print(iso_col)
		print(col)
		print(delta_mag)

		delta_interp = PchipInterpolator(col,delta_mag)

		self.lf_centres = centres
		self.lf_n = luminosity_function

		if plot:

			plt.figure(figsize=(4.5,6))
			ax = plt.axes()
			ax.scatter(data.colour,data.magnitude,marker='.',c='k',s=0.2)
			xmag = np.linspace(self.magnitude_min,self.magnitude_max,1001)
			ax.plot(iso_colour_interp(xmag),xmag,'b--',alpha=0.7)
			xcol = iso_colour_interp(xmag)
			ax.plot(xcol,xmag+delta_interp(xcol),'r-',alpha=0.7)
			print('xcol,xmag,delta_interp',xcol[-5:],xmag[-5:],delta_interp(xcol[-5:]))
			if plot_binary_sequence:
				ax.plot(xcol,xmag+delta_interp(xcol)-0.753,'r-',alpha=0.7)
			if plot_triple_sequence:
				ax.plot(xcol,xmag+delta_interp(xcol)-1.193,'r-',alpha=0.7)
			ax.set_xlabel(data.colour_label)
			ax.set_ylabel(data.magnitude_label)
			ax.set_ylim([data.magnitude_max+1,data.magnitude_min-1])
			ax.set_xlim([np.min(xcol)-0.25,np.max(xcol)+0.5])
			plt.savefig(plot_file)

		return delta_interp


	def colour_correction_interpolator(self,data,plot=True,plot_file='colour_correction.png',plot_binary_sequence=True):

		"""Return a function that computes a colour-correction (as a function of magnitude) to be added to the
		isochrone in order to match the main-sequence ridge line."""

		assert isinstance(data,Data)

		index = np.argsort(self.magnitude)

		iso_colour_interp = PchipInterpolator(self.magnitude[index],self.colour[index])
		data_delta = data.colour - iso_colour_interp(data.magnitude)

		nbins = np.int(1*(data.magnitude_max - data.magnitude_min) + 0.5)

		y = -9999*np.ones(nbins)
		luminosity_function = np.empty(nbins)

		edges = np.linspace(data.magnitude_min,data.magnitude_max,nbins+1)
		centres = 0.5*(edges[1:]+edges[:-1])


		for i in range(nbins):

			pts = np.where((data.magnitude > edges[i]) & (data.magnitude <= edges[i+1]))[0]
			luminosity_function[i] = len(pts)

			if len(pts) > 4:
				#y[i] = self.hist_peak(data_delta[pts])
				y[i] = self.median_peak(data_delta[pts])

		good = y > -100

		delta_interp = PchipInterpolator(centres[good],y[good])

		self.lf_centres = centres
		self.lf_n = luminosity_function

		if plot:

			plt.figure(figsize=(4.5,6))
			ax = plt.axes()
			ax.scatter(data.colour,data.magnitude,marker='.',c='k',s=0.2)
			xmag = np.linspace(self.magnitude_min,self.magnitude_max,1001)
			ax.plot(iso_colour_interp(xmag),xmag,'b--',alpha=0.7)
			ax.plot(iso_colour_interp(xmag)+delta_interp(xmag),xmag,'r-',alpha=0.7)
			if plot_binary_sequence:
				ax.plot(iso_colour_interp(xmag)+delta_interp(xmag),xmag-0.75,'r-',alpha=0.7)
			ax.set_xlabel(data.colour_label)
			ax.set_ylabel(data.magnitude_label)
			ax.set_ylim([data.magnitude_max+1,data.magnitude_min-1])
			ax.set_xlim([np.min(iso_colour_interp(xmag))-0.25,np.max(iso_colour_interp(xmag)+delta_interp(xmag))+0.5])
			plt.savefig(plot_file)

		return delta_interp




	def plot_luminosity_mass_functions(self,magnitude_label='Magnitude',plot_file='ML_functions.png'):

			fig, ax = plt.subplots(3,2,figsize=(8,10))

			ax[0,0].scatter(self.lf_centres,self.lf_n,marker='.',c='b',s=20)
			ax[0,0].set_xlabel(magnitude_label)
			ax[0,0].set_ylabel(r'$N$')
			ylim = ax[0,0].get_ylim()
			ax[0,0].set_ylim((0,ylim[1]))

			ax[1,0].scatter(self.mag_M_interp(self.lf_centres),self.lf_n,marker='.',c='b',s=20)
			ax[1,0].set_xlabel(r'$Mass (M_\odot)$')
			ax[1,0].set_ylabel(r'$N$')
			ax[1,0].set_ylim((0,ylim[1]))

			ax[2,0].scatter(np.log10(self.mag_M_interp(self.lf_centres)),self.lf_n,marker='.',c='b',s=20)
			ax[2,0].set_xlabel(r'$\log_{10} Mass (M_\odot)$')
			ax[2,0].set_ylabel(r'$N$')
			ax[2,0].set_ylim((0,ylim[1]))

			ax[0,1].scatter(self.lf_centres,np.log10(self.lf_n),marker='.',c='b',s=20)
			ax[0,1].set_xlabel(magnitude_label)
			ax[0,1].set_ylabel(r'$\log_{10} N$')
			ylim = ax[0,1].get_ylim()
			ax[0,1].set_ylim((0,ylim[1]))

			ax[1,1].scatter(self.mag_M_interp(self.lf_centres),np.log10(self.lf_n),marker='.',c='b',s=20)
			ax[1,1].set_xlabel(r'$Mass (M_\odot)$')
			ax[1,1].set_ylabel(r'$\log_{10} N$')
			ax[1,1].set_ylim((0,ylim[1]))

			ax[2,1].scatter(np.log10(self.mag_M_interp(self.lf_centres)),np.log10(self.lf_n),marker='.',c='b',s=20)
			ax[2,1].set_xlabel(r'$\log_{10} Mass (M_\odot)$')
			ax[2,1].set_ylabel(r'$\log_{10} N$')
			ax[2,1].set_ylim((0,ylim[1]))

			plt.tight_layout()

			plt.savefig(plot_file)
	


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

		if self.correction_type == 'colour':
			return mag, blue - red + self.colour_correction(mag) + self.colour_offset
		else:
			return mag, blue - red + self.colour_offset






