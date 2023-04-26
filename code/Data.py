import sys
import os
import numpy as np
#import cunumeric as np

from scipy.interpolate import PchipInterpolator

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Data():

	"""Container class to hold CMD data"""

	def __init__(self,data_dict,data_field=None):

		"""
		Set up a Data instance.


		Inputs:

			data_dict:			(dictionary) with entries:
								file : (string) data file name
								name: (string) data set name
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

		if data_field is None:
			data_file = data_dict['file']
		else:
			data_file = data_dict[data_field]

		if 'error_multiplier' not in data_dict:
			data_dict['error_multiplier'] = 1.0


		rawdata = np.loadtxt(data_file)
		data = rawdata[~np.isnan(rawdata).any(axis=1),:]

		self.magnitude = data[:,data_dict['column_mag']]

		self.colour = data[:,data_dict['column_blue']] - data[:,data_dict['column_red']]

		self.magnitude_err = data_dict['error_multiplier']*data[:,data_dict['column_mag_err']]
		self.colour_err = data_dict['error_multiplier']*np.sqrt(data[:,data_dict['column_blue_err']]**2 + data[:,data_dict['column_red_err']]**2)

		self.magnitude_min = data_dict['magnitude_min']
		self.magnitude_max = data_dict['magnitude_max']

		self.colour_label = data_dict['colour_label']
		self.magnitude_label = data_dict['magnitude_label']


		self.trim_left = 0.05
		self.trim_right = 0.15
		if 'trim_left' in data_dict:
			self.trim_left = data_dict['trim_left']
		if 'trim_right' in data_dict:
			self.trim_right = data_dict['trim_right']

		self.name = ""
		if 'name' in data_dict:
			self.name = data_dict['name']


		# set up data covariance matrices

		if 'column_mag_equiv' not in data_dict:
			data_dict['column_mag_equiv'] = data_dict['column_mag']

		self.cov = np.empty((len(self.magnitude),2,2),dtype='float32')

		if data_dict['column_mag'] == data_dict['column_blue'] or data_dict['column_mag_equiv'] == data_dict['column_blue']:

			self.cov[:,0,0] = data_dict['error_multiplier']**2*(data[:,data_dict['column_blue_err']]**2 + data[:,data_dict['column_red_err']]**2)
			self.cov[:,0,1] = data_dict['error_multiplier']**2*data[:,data_dict['column_blue_err']]**2
			self.cov[:,1,0] = data_dict['error_multiplier']**2*data[:,data_dict['column_blue_err']]**2
			self.cov[:,1,1] = data_dict['error_multiplier']**2*data[:,data_dict['column_blue_err']]**2
			self.magnitude_type = 'blue'

		elif data_dict['column_mag'] == data_dict['column_red'] or data_dict['column_mag_equiv'] == data_dict['column_red']:

			self.cov[:,0,0] = data_dict['error_multiplier']**2*(data[:,data_dict['column_blue_err']]**2 + data[:,data_dict['column_red_err']]**2)
			self.cov[:,0,1] = data_dict['error_multiplier']**2*data[:,data_dict['column_red_err']]**2
			self.cov[:,1,0] = data_dict['error_multiplier']**2*data[:,data_dict['column_red_err']]**2
			self.cov[:,1,1] = data_dict['error_multiplier']**2*data[:,data_dict['column_red_err']]**2
			self.magnitude_type = 'red'

		else:

			self.cov[:,0,0] = data_dict['error_multiplier']**2*(data[:,data_dict['column_blue_err']]**2 + data[:,data_dict['column_red_err']]**2)
			self.cov[:,0,1] = 0.0
			self.cov[:,1,0] = 0.0
			self.cov[:,1,1] = data_dict['error_multiplier']**2*data[:,data_dict['column_mag_err']]**2
			self.magnitude_type = 'independent'

		return



	def upload_to_GPU(self,drv,likelihood_functions):

		"""Upload data to GPU texture memory."""


		c_cov = self.cov.reshape((len(self.magnitude),4),order='F')
		self.cov_CUDA = likelihood_functions.get_texref("cov")
		drv.matrix_to_texref(np.float32(c_cov),self.cov_CUDA,order='F')
		self.cov_CUDA.set_filter_mode(drv.filter_mode.POINT)

		col_mag = np.vstack((self.colour,self.magnitude))

		self.col_mag_CUDA = likelihood_functions.get_texref("data")
		drv.matrix_to_texref(np.float32(col_mag),self.col_mag_CUDA,order='C')
		self.col_mag_CUDA.set_filter_mode(drv.filter_mode.POINT)

		return


	def trim(self,isochrone,plot=True,plot_file='data_trim.png',return_axis=False):

		"""Apply some cuts to the data, based on the provided isochrone."""

		from Isochrone import Isochrone

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

		print('self.magnitude_min',self.magnitude_min)
		print('self.magnitude_max',self.magnitude_max)
		print('M_min',M_min)
		print('M_max',M_max)
		print('B_min_mag',B_min_mag)
		print('B_min_colour',B_min_colour)
		print('B_max_mag',B_max_mag)
		print('B_max_colour',B_max_colour)
		print('k_max_flag',k_max_flag)
		print('k_max',k_max)


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

			plt.figure(figsize=(3.3,4.5))
			ax = plt.axes()
			ax.scatter(self.colour,self.magnitude,c='k',s=0.2,alpha=0.6,marker='.')


		self.raw_magnitude = self.magnitude.copy()
		self.raw_colour = self.colour.copy()

		bad_points = np.setdiff1d(np.arange(len(self.magnitude)),good_points,assume_unique=True)
		self.cut_magnitude = self.magnitude[bad_points]
		self.cut_colour = self.colour[bad_points]

		self.magnitude = self.magnitude[good_points]
		self.colour = self.colour[good_points]
		self.cov = self.cov[good_points]

		data_iso_colour = isochrone.mag_colour_interp(self.magnitude)
		good_points = np.where( (self.colour - data_iso_colour > -self.trim_left) & (self.colour - data_iso_colour < self.trim_right) )[0]

		self.magnitude = self.magnitude[good_points]
		self.colour = self.colour[good_points]
		self.cov = self.cov[good_points]

		if plot:

			ax.scatter(self.colour,self.magnitude,c='b',s=0.5,marker='.')
			ax.scatter(-100,-100,c='w',s=0.001,marker='.',label=self.name)

			xmag = np.linspace(self.magnitude_min-0.5,self.magnitude_max+0.5,1001)
			ax.plot(isochrone.mag_colour_interp(xmag),xmag,'r-',alpha=0.5)

			ax.plot(isochrone.colour,isochrone.magnitude,'g--',alpha=0.5)

			xmag = np.linspace(self.magnitude_min+0.5,self.magnitude_max+0.5,1001)
			ax.plot(isochrone.mag_colour_interp(xmag)-self.trim_left,xmag,'c-',alpha=1.0)
			xmag = np.linspace(self.magnitude_min-0.55,self.magnitude_max-0.55,1001)
			ax.plot(isochrone.mag_colour_interp(xmag)+self.trim_right,xmag,'c-',alpha=1.0)

			ax.plot(B_min_colour,B_min_mag,'c-')
			ax.plot(B_max_colour,B_max_mag,'c-')

			ax.set_xlabel(self.colour_label)
			ax.set_ylabel(self.magnitude_label)
			ax.set_ylim([self.magnitude_max+1,self.magnitude_min-1])
			ax.set_xlim([np.min(isochrone.mag_colour_interp(xmag))-0.25,np.max(isochrone.mag_colour_interp(xmag))+0.5])
			ax.legend(frameon=False)

			if return_axis:
				
				return ax

			else:
				
				plt.tight_layout()
				plt.savefig(plot_file)


		return

