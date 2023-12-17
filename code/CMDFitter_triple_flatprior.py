import sys
import os
import numpy as np
#import cunumeric as np

from scipy.interpolate import PchipInterpolator
from scipy.optimize import nnls
from scipy.stats import norm, truncnorm
from scipy.special import gamma

from matplotlib.tri import CubicTriInterpolator, Triangulation


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pylab import subplots_adjust

import json

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from likelihood_gaussbf_triple_CUDA import likelihood_functions
likelihood = likelihood_functions.get_function("likelihood")

from Data import Data
from Isochrone import Isochrone

from pytntnn import tntnn

class CMDFitter():

	"""Main class for the CMDFitter code."""

	def __init__(self,json_file=None,data=None,isochrone=None,iso_correction_type='colour',trim_data=True,q_model='legendre',m_model='power',outlier_scale=2.0,q_min=0.0,error_scale_type='both',include_triples=False,model_isochrone_correction=False,parameters_dict={'n_q_hist_bins':5}):


		"""
		Set up a CMDFitter instance. This can be defined by providing a json-format definition file, or separate Data and Isochrone objects.


		Inputs:

			json_file:		(string) the name of a json-format file with the data and isochrone defintions

			data:			(Data) input CMD data instance

			isochrone:		(Isochrone) input Isochrone instance

			trim_data:		(boolean) data will be filtered if true

			q_model:		(string) functional form for q distribution. Must be "power" or "legendre"

			outlier_scale:  (float) multiplicative constant to scale the bivariate gaussian data distribution to make the outlier distribution

			q_min:			(float) The q distribution function is defined over (q_min,1). fb means fraction of binaries with q > q_min

			error_scale_type: (string) Must be "colour" or "both", indicating how the data error scaling parameters are to be applied

			include_triples (boolean) Whether to model triples stars as well as singles and binaries

			model_isochrone_correction (boolean) Whether to model (rather than just initially fit) the isochrone main-sequence ridge line

		"""

		self.version = 7.0

		self.citation = "Albrow, M.D., Ulusele, I.H., 2022, MNRAS, 515, 730"

		self.drv = drv
		self.likelihood_functions = likelihood_functions
		self.likelihood = likelihood_functions.get_function("likelihood")

		self.model_isochrone_correction = model_isochrone_correction
		self.include_triples = include_triples

		assert error_scale_type in ['colour','both']
		self.error_scale_type = error_scale_type

		assert q_model in ['quadratic','single_power','power','legendre','piecewise','hist']
		assert m_model in ['power']

		self.q_model = q_model
		self.m_model = m_model

		self.prefix = 'out_'

		assert q_min >= 0.0 and q_min < 1.0
		self.q_min = q_min

		# Miscellaneous attributes
		self.sp_log_d_s = np.log10(0.25)
		self.sp_d = 0.1
		self.neginf = -np.inf
		self.scale_fo = 0.15
		self.emcee_walker_dispersion = 1.e-7
		self.isochrone_correction_scale = 0.1

		# Exit now if no input provided
		if json_file is None and (data is None or isochrone is None):
			return

		if json_file is not None:

			data, isochrone = self.set_up_from_json(json_file,iso_correction_type)

		else:

			assert isinstance(data, Data)
			assert isinstance(isochrone, Isochrone)

		self.data = data
		self.iso = isochrone

		self.define_data_outlier_model(outlier_scale)

		self.set_up_model_parameters(parameters_dict)

		self.freeze = np.zeros(self.ndim)

		if trim_data:
			self.data.trim(self.iso)

		self.data.upload_to_GPU(self.drv,self.likelihood_functions)

		self.set_up_data_attributes()

		self.set_up_basis_functions()

		self.upload_basis_functions_to_GPU()

		print('Default parameters:', self.default_params)
		print('Likelihood test for default parameters:',self.lnlikelihood(self.default_params))

		return


	def set_up_from_json(self,json_file,iso_correction_type):

		with open(json_file) as file:
			data = json.load(file)
		
		data_description = data['data_description']
		iso_description = data['iso_description']
		
		print()
		print('data',data)
		print()
		print('data_description',data_description)
		print()


		if 'isochrone_correction_file' in data_description:
			data = Data(data_description)
			isochrone_correction_data = Data(data_description,data_field='isochrone_correction_file')
		else:
			data = Data(data_description)
			isochrone_correction_data = data

		isochrone = Isochrone(iso_description, isochrone_correction_data=isochrone_correction_data, correction_type=iso_correction_type)

		return data, isochrone


	def set_up_data_attributes(self):

		self.mass_slice = np.array([self.iso.mag_M_interp(self.data.magnitude_max),self.iso.mag_M_interp(self.data.magnitude_min)])
		self.mass_range = self.mass_slice[1] - self.mass_slice[0]
		print('Mass range:',self.mass_slice,self.mass_range)

		self.M = self.iso.mag_M_interp(self.data.magnitude),

		self.M_ref = np.mean(self.mass_slice)
		self.delta_M = np.abs(self.mass_slice[0] - self.M_ref)
		print('self.delta_M',self.delta_M)

		self.magnitude_ref = 0.5*(self.data.magnitude_min+self.data.magnitude_max)
		self.delta_mag = self.data.magnitude_max - self.magnitude_ref

		self.h_magnitude_ref = self.data.magnitude_min


	def define_data_outlier_model(self,outlier_scale):

		d = np.column_stack((self.data.colour,self.data.magnitude))
		d_med = np.median(d,axis=0)
		d_cov = outlier_scale**2*np.cov(d,rowvar=False)
		print('Outlier median:', d_med)
		print('Outlier covariance:', d_cov)
		self.outlier_description = np.ascontiguousarray([d_med[0],d_med[1],d_cov[0,0],d_cov[1,1],d_cov[0,1]],dtype=np.float64)


	def upload_basis_functions_to_GPU(self):

		self.bf_dim = 2

		if self.include_triples:
			self.bf_dim = 3

		D_ij = np.zeros([self.n_bf**self.bf_dim,2])
		S_ij = np.zeros([self.n_bf**self.bf_dim,2,2])

		width_matrix = np.zeros([self.bf_dim,self.bf_dim])
		width_matrix[0,0] = self.Mw**2
		width_matrix[1,1] = self.qw**2

		if self.include_triples:

			width_matrix[2,2] = self.qw**2

			for i in range(self.n_bf):
				for j in range(self.n_bf):
					for k in range(self.n_bf):
						jacob = self.jacobian_triple(self.M0[i],self.q0[j],self.q0[k]) 
						mag, colour = self.iso.triple(self.M0[i],self.q0[j],self.q0[k])
						D_ij[i+j*self.n_bf+k*self.n_bf**2] = np.array([colour,mag])
						S_ij[i+j*self.n_bf+k*self.n_bf**2] = np.dot(jacob,(np.dot(width_matrix,(jacob.T))))

		else:

			mag, colour = self.iso.binary_mesh(self.M0,self.q0)
			jacob = self.jacobian_binary_mesh().reshape(self.n_bf**2,2,2)

			D_ij = np.stack((colour,mag),axis=-1).reshape(self.n_bf**2,2)
			xx = np.einsum('ij,klj->kil', width_matrix, jacob)
			S_ij = np.einsum('ijk,ikl->ijl',jacob, xx)
			
			# for i in range(self.n_bf):
			# 	for j in range(self.n_bf):
			# 		jacob = self.jacobian_binary(self.M0[i],self.q0[j]) 
			# 		width_matrix[0,0] = self.Mw**2
			# 		width_matrix[1,1] = self.qw**2
			# 		mag, colour = self.iso.binary(self.M0[i],self.q0[j])
			# 		D_ij[i+j*self.n_bf] = np.array([colour,mag])
			# 		S_ij[i+j*self.n_bf] = np.dot(jacob,(np.dot(width_matrix,(jacob.T))))
			# sys.exit()

		S_ij_shaped = S_ij.reshape(self.n_bf**self.bf_dim,4).astype(np.float64)
		D_ij = D_ij.astype(np.float64)

		print()
		print('transformed basis functions')
		for i in range(self.n_bf):
			for j in range(self.n_bf):
				if (np.abs(self.M0[i]-0.99) < 0.02) and (np.abs(self.q0[j]-0.49) < 0.02):
					print(self.M0[i],self.q0[j],colour[i,j],mag[i,j],S_ij[i+j*self.n_bf])
		print()


		self.DMQ_gpu = drv.mem_alloc(D_ij.nbytes)
		drv.memcpy_htod(self.DMQ_gpu, D_ij)

		self.SMQ_gpu = drv.mem_alloc(S_ij_shaped.nbytes)
		drv.memcpy_htod(self.SMQ_gpu, S_ij_shaped)

		#print("DMQ",D_ij[0,0],D_ij[0,1])
		#print("DMQ",S_ij_shaped[0,0],S_ij_shaped[0,1],S_ij_shaped[0,2],S_ij_shaped[0,3])

		# self.DMQ_CUDA = self.likelihood_functions.get_texref("DMQ")
		# self.drv.matrix_to_texref(np.float32(D_ij),self.DMQ_CUDA,order='F')
		# self.DMQ_CUDA.set_filter_mode(self.drv.filter_mode.POINT)

		# self.SMQ_CUDA = self.likelihood_functions.get_texref("SMQ")
		# self.drv.matrix_to_texref(np.float32(S_ij_shaped),self.SMQ_CUDA,order='F')
		# self.SMQ_CUDA.set_filter_mode(self.drv.filter_mode.POINT)


	def jacobian_triple(self,M,q1,q2):

		jacob = np.zeros((2,3))

		mag_fix_q1_q2, colour_fix_q1_q2 = self.iso.triple(self.M0,q1,q2)
		mag_fix_M_q1, colour_fix_M_q1 = self.iso.triple(M,q1,self.q0)
		mag_fix_M_q2, colour_fix_M_q2 = self.iso.triple(M,self.q0,q2)

		mag_q1 = PchipInterpolator(self.q0,mag_fix_M_q2)
		mag_q2 = PchipInterpolator(self.q0,mag_fix_M_q1)
		mag_M = PchipInterpolator(self.M0,mag_fix_q1_q2)

		colour_q1 = PchipInterpolator(self.q0,colour_fix_M_q2)
		colour_q2 = PchipInterpolator(self.q0,colour_fix_M_q1)
		colour_M = PchipInterpolator(self.M0,colour_fix_q1_q2)

		jacob[0,0] = colour_M(M,1)
		jacob[0,1] = colour_q1(q1,1)
		jacob[0,2] = colour_q2(q2,1)
		jacob[1,0] = mag_M(M,1)
		jacob[1,1] = mag_q1(q1,1)
		jacob[1,2] = mag_q2(q2,1)

		return jacob


	def jacobian_binary_mesh(self):

		jacob = np.zeros((2,2))

		M_grid, q_grid = np.meshgrid(self.M0,self.q0)
		mag, colour = self.iso.binary_mesh(self.M0,self.q0)

		tri = Triangulation(q_grid.ravel(),M_grid.ravel())
		#tci_mag = CubicTriInterpolator(tri, mag.ravel(),kind='geom')
		#tci_colour = CubicTriInterpolator(tri, colour.ravel(),kind='geom')
		tci_mag = CubicTriInterpolator(tri, mag.ravel())
		tci_colour = CubicTriInterpolator(tri, colour.ravel())
		dmag_dq, dmag_dM = tci_mag.gradient(tri.x, tri.y)
		dcolour_dq, dcolour_dM = tci_colour.gradient(tri.x, tri.y)

		xx = q_grid[24:27,24:27]
		yy = M_grid[24:27,24:27]

		#print('xx(q)',xx)
		#print('yy(M)',yy)
		#print('mag',tci_mag(xx,yy))
		#print('grad',tci_mag.gradient(xx[1],yy[1]))


		jacob = np.stack((np.stack((dcolour_dM, dcolour_dq), axis=1),np.stack((dmag_dM, dmag_dq), axis=1)), axis=1).reshape(self.n_bf,self.n_bf,2,2)

		#print('jacob',jacob[25,25])

		#sys.exit()

		return jacob


	def jacobian_binary(self,M,q):

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



	def set_up_basis_functions(self):

		self.n_bf = 50

		self.Mw = 0.8*self.mass_range / self.n_bf
		self.M0 = np.linspace(self.mass_slice[0],self.mass_slice[1],self.n_bf)

		self.qw = 0.8*1.0/self.n_bf
		self.q0 = np.linspace(self.q_min,1.0,self.n_bf)

		self.qsigma = 0.0001
		self.qx = np.linspace(self.q_min,1.0,500)
		self.qbf = np.zeros([self.n_bf,500])
		for i in range(self.n_bf):
			self.qbf[i,:] = (1/(self.qw*(2*np.pi)**0.5))*np.exp(-((self.qx-self.q0[i])**2)/(2*self.qw**2)) 

		self.qbf_sigma2  = self.qbf/self.qsigma**2

		self.qA = np.zeros((self.n_bf,self.n_bf))
		for k in range(self.n_bf):
			for j in range(self.n_bf):
				self.qA[k,j] = np.sum(self.qbf[k,:]*self.qbf[j,:]/self.qsigma**2) 

		self.qAT = self.qA.T
		self.qAA = np.dot(self.qAT,self.qA)

		self.Msigma = 0.0001
		self.Mx = np.linspace(self.mass_slice[0],self.mass_slice[1],500)

		self.Mbf = np.zeros([self.n_bf,500])
		for i in range(self.n_bf):
			self.Mbf[i,:] = (1/(self.Mw*(2*np.pi)**0.5))*np.exp(-((self.Mx-self.M0[i])**2)/(2*self.Mw**2))

		self.Mbf_sigma2  = self.Mbf/self.Msigma**2

		self.MA = np.zeros((self.n_bf,self.n_bf))
		for k in range(self.n_bf):
			for j in range(self.n_bf):
				self.MA[k,j] = np.sum(self.Mbf[k,:]*self.Mbf[j,:]/self.Msigma**2) 

		self.MAT = self.MA.T
		self.MAA = np.dot(self.MAT,self.MA)


	def set_up_model_parameters(self,parameters_dict):

		"""
			Parameters are arranged as:

				0 - q_index:  mass function
				q_index+1 - b_index:	mass-ratio distribution function
				b_index+1 - i_index:	binary and triple fractions, error scaling
				i_index+1 - end:		offsets in colour or magnitude to be applied to the isochrone (evaluated at magnitudes isochrone.initial_colour_correction_offsets[0] or isochrone.initial_magnitude_correction_offsets[0])

		"""

		# Mass function

		if self.m_model == 'legendre':
			self.labels = [r"$b_1$", r"$b_2$", r"$b_3$", r"$b_4$"]
			self.default_params = np.array([0.0, 0.0, 0.0, 0.0])
		else:
			self.labels = [r"$\log_{10} k$", r"$M_0$", r"$\gamma$",  r"$c_0$",  r"$\dot{c}_0$"]
			self.default_params = np.array([4.0, 0.0, 0.0006, 0.0, 0.0])

		self.q_index = len(self.default_params)

		# Mass-ratio distribution function

		if self.q_model == 'quadratic':
			self.labels += [r"$q_0$",r"$a_1$",r"$\dot{q_0}$",r"$\dot{a_1}$"]
			self.default_params = np.hstack((self.default_params, np.array([0.0, 0.0, 0.0, 0.0])))

		if self.q_model == 'single_power':
			self.labels += [r"$\beta$",  r"$q_1$", r"$\dot{\beta}$",  r"$\dot{q_1}$"]
			self.default_params = np.hstack((self.default_params, np.array([np.pi/6, 0.5, 0.0, 0.0])))

		if self.q_model == 'power':
			self.labels += [r"$\alpha_1$",  r"$\alpha_2$", r"$q_0$",  r"$a_1$", r"$a_2$"]
			self.default_params = np.hstack((self.default_params, np.array([2.0, 2.0, 0.5, 1.0, 1.0])))

		if self.q_model == 'legendre':
			try:
				self.n_legendre = parameters_dict['n_legendre']
			except KeyError:
				self.n_legendre = 3
			self.set_up_legendre_functions()
			dot = r'\dot'
			#self.labels += [r"$\log_{10} \delta$"]
			self.labels += [r"$\delta$"]
			self.labels += [f"$a_{i}$" for i in range(1,self.n_legendre+1)]
			self.labels += [f"${dot}{{a_{i}}}$" for i in range(1,self.n_legendre+1)]
			self.default_params = np.hstack((self.default_params, 0.25, np.zeros(self.n_legendre),np.zeros(self.n_legendre)))
			#self.labels += [r"$a_1$", r"$a_2$", r"$a_3$", r"$\dot{a}_1$", r"$\dot{a}_2$", r"$\dot{a}_3$"]
			#self.default_params = np.hstack((self.default_params, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])))

		if self.q_model == 'piecewise':
			self.labels += [r"$pq_4$",  r"$pq_3$",  r"$pq_2$", r"$pq_1$"]
			self.default_params = np.hstack((self.default_params, np.array([1.0, 1.0, 1.0, 1.0])))

		if self.q_model == 'hist':
			try:
				self.n_q_hist_bins = parameters_dict['n_q_hist_bins']
			except KeyError:
				self.n_q_hist_bins = 5
			dot = r'\dot'
			self.labels += [f"$pq_{{{i}}}$" for i in range(self.n_q_hist_bins)]
			self.labels += [f"${dot}{{pq_{{{i}}}}}$" for i in range(self.n_q_hist_bins)]
			self.default_params = np.hstack((self.default_params, np.ones(self.n_q_hist_bins),np.zeros(self.n_q_hist_bins)))

		self.b_index = len(self.default_params)

		# Binaries and triples

		self.labels += [r"$f_B$", r"$\dot{f_B}$"]
		self.default_params = np.hstack((self.default_params,np.array([0.35,0.0])))

		self.labels += [r"$f_T$"]
		self.default_params = np.hstack((self.default_params,np.array([0.005])))

		# Outliers and error scaling

		self.labels += [r"$o_c$",r"$o_m$",r"$o_{cv}$",r"$o_{mv}$",r"$o_{cov}$",r"$f_O$", r"$log h$", r"$\dot{h}$"]
		self.default_params = np.hstack((self.default_params,self.outlier_description,np.array([0.0,0.0,0.0])))
		self.i_index = len(self.default_params)

		# Isochrone correction

		if self.model_isochrone_correction:

			if self.iso.correction_type == 'colour':
				self.labels += [f'io{j}' for j in range(len(self.iso.initial_colour_correction_offsets[0]))]
				self.default_params = np.hstack((self.default_params,self.iso.initial_colour_correction_offsets[1]))
			else:
				self.labels += [f'io{j}' for j in range(len(self.iso.initial_magnitude_correction_offsets[0]))]
				self.default_params = np.hstack((self.default_params,self.iso.initial_magnitude_correction_offsets[1]))

		print('q_index, b_index, i_index:',self.q_index, self.b_index, self.i_index)

		self.ndim = len(self.default_params)


	def set_up_legendre_functions(self):

		assert self.n_legendre < 6

		# Shifted Legendre polynomials
		#self.sl_0 = lambda x: x*0.0 + 1.0
		#self.sl_1 = lambda x: 2.0*x - 1.0
		#self.sl_2 = lambda x: 6.0*x**2 - 6.0*x + 1.0
		#self.sl_3 = lambda x: 20.0*x**3 - 30.0*x**2 + 12.0*x - 1.0
		#self.sl_4 = lambda x: 70.0*x**4 - 140.0*x**3 + 90.0*x**2 - 20.0*x + 1.0
		self.sl = [lambda x: x*0.0 + 1.0,
				lambda x: 2.0*x - 1.0,
				lambda x: 6.0*x**2 - 6.0*x + 1.0,
				lambda x: 20.0*x**3 - 30.0*x**2 + 12.0*x - 1.0,
				lambda x: 70.0*x**4 - 140.0*x**3 + 90.0*x**2 - 20.0*x + 1.0,
				lambda x: 252.0*x**5 - 630.0*x**4 + 560.0*x**3 - 210.0*x**2 + 30.0*x - 1.0]

		# Derivatives of shifted Legendre polynomials
		#self.der_sl_0 = lambda x: 0.0
		#self.der_sl_1 = lambda x: 2.0
		#self.der_sl_2 = lambda x: 12.0*x - 6.0
		#self.der_sl_3 = lambda x: 60.0*x**2 - 60.0*x + 12.0
		#self.der_sl_4 = lambda x: 280.0*x**3 - 420.0*x**2 + 180.0*x - 20.0
		self.der_sl = [lambda x: 0.0,
				lambda x: 2.0,
				lambda x: 12.0*x - 6.0,
				lambda x: 60.0*x**2 - 60.0*x + 12.0,
				lambda x: 280.0*x**3 - 420.0*x**2 + 180.0*x - 20.0,
				lambda x: 1260.0*x**4 - 2520.0*x**3 + 1680.0*x**2 - 4200*x + 30.0]

		# Integrals of shifted Legendre polynomials
		#self.int_sl_0 = lambda x: x
		#self.int_sl_1 = lambda x: x**2 - x
		#self.int_sl_2 = lambda x: 2.0*x**3 - 3.0*x**2 + x
		#self.int_sl_3 = lambda x: 5.0*x**4 - 10.0*x**3 + 6.0*x**2 - x
		#self.int_sl_4 = lambda x: 14.0*x**5 - 35.0*x**4 + 30.0*x**3 - 10.0*x**2 + x
		self.int_sl = [lambda x: x,
				lambda x: x**2 - x,
				lambda x: 2.0*x**3 - 3.0*x**2 + x,
				lambda x: 5.0*x**4 - 10.0*x**3 + 6.0*x**2 - x,
				lambda x: 14.0*x**5 - 35.0*x**4 + 30.0*x**3 - 10.0*x**2 + x,
				lambda x: 42.0*x**6 - 126.0*x**5 + 140.0*x**4 - 70.0*x**3 + 15.0*x**2 -x]


	def M_distribution(self,x,params):

		"""Evaluate mass function at x."""

		assert self.m_model in ['power']

		if self.m_model == 'power':
			log_k,x0,gamma,c0, c1 = params
			k = 10.0**log_k
			m = np.linspace(self.mass_slice[0],self.mass_slice[1],1000)
			#c_scale = np.max([self.mass_slice[0],x0])**(-gamma)
			c_scale = self.mass_slice[0]**(-gamma)
			#y = ((c0 + c1*(m-self.mass_slice[0]))*c_scale + m**(-gamma) )/ (1.0 + np.exp(-k*(m-x0)) )
			y = ((c0 + c1*(m-self.mass_slice[0]))*c_scale + m**(-gamma) ) * np.tanh(-k*(m-x0)) 
			normalM = 1.0 / (np.sum(y)*(m[1]-m[0]))

			#y =  normalM * ((c0 + c1*(x-self.mass_slice[0]))*c_scale + x**(-gamma)) / (1.0 + np.exp(-k*(x-x0)))
			y =  normalM * ((c0 + c1*(x-self.mass_slice[0]))*c_scale + x**(-gamma)) * np.tanh(-k*(x-x0)) 
			y[x<self.mass_slice[0]] = 0.0
			y[x>self.mass_slice[1]] = 0.0
			return y



	def M_distribution_sampler(self,params):

		"""Return a function that maps the range (0,1) onto the mass function.""" 

		m = np.linspace(self.mass_slice[0],self.mass_slice[1],10000)

		y = self.M_distribution(m,params)+np.arange(10000)*1.e-8
		pts = np.where(y>0.0)[0]
		y_cumulative = np.cumsum(y[pts])/np.sum(y[pts])
		pts = np.where(y_cumulative>1.e-50)[0]
		return PchipInterpolator(y_cumulative[pts],m[pts])


	def q_distribution_quadratic(self,x,params):

		q0, a1, q0_dot, a1_dot, M = params

		dM = M-self.M_ref

		a0 = 1.0 - (a1+a1_dot*dM)*(1.0-(q0+q0_dot*dM))**3/3.0

		xarr = np.atleast_1d(x)

		qdist = a0*np.ones_like(xarr)

		p = np.where(xarr > q0 + q0_dot*dM)[0]

		qdist[p] += (a1+a1_dot*dM)*(xarr[p]-(q0+q0_dot*dM))**2

		return qdist
	

	def q_distribution_single_power(self,x,params):

		beta, q1, beta_dot, q1_dot, M = params

		dM = M-self.M_ref

		tan_beta = np.tan(beta+beta_dot*dM)

		alpha = 1.0001 + self.sp_log_d_s/np.log10(q1+q1_dot*dM)

		a1 = tan_beta/alpha

		alpha1 = alpha + 1.0

		a0 = 1.0 - a1/alpha1

		xarr = np.atleast_1d(x)
		qdist = a0 + a1*xarr**alpha

		return qdist


	def q_distribution_power(self,x,params):

		alpha1, alpha2, q0, a1, a2, M = params
		xarr = np.atleast_1d(x)
		qdist = np.zeros_like(xarr)

		c = (1.0 - a1*(q0-self.q_min)**(alpha1+1.0)/(alpha1+1.0) - a2*(1.0-q0)**(alpha2+1.0)/(alpha2+1.0)) / (1.0-self.q_min)
		
		qdist[xarr<q0] = c + a1*(q0 - xarr[xarr<q0])**alpha1
		qdist[xarr>=q0] = c + a2*(xarr[xarr>=q0] - q0)**alpha2

		return qdist


	def q_distribution_legendre(self,x,params):

		a = np.array(params[1:self.n_legendre+1])
		adot = np.array(params[self.n_legendre+1:2*self.n_legendre+1])
		M = params[-1]
		#a1, a2, a3, a1_dot, a2_dot,a3_dot, M = params
		dM = M-self.M_ref

		x_arr = np.atleast_1d(x)
		
		# Map q to 0 - 1 domain for legendre functions
		x1 = (x_arr-self.q_min)/(1.0-self.q_min) 

		#return self.sl_0(x1) + (a1+a1_dot*dM)*self.sl_1(x1) + (a2+a2_dot*dM)*self.sl_2(x1) + (a3+a3_dot*dM)*self.sl_3(x1)
		f = self.sl[0](x1) 
		for i in range(self.n_legendre):
			f += (a[i]+adot[i]*dM)*self.sl[i+1](x1)
		return f


	def q_distribution_piecewise(self,x,params):

		pnode = np.zeros(5)
		pnode[4], pnode[3], pnode[2], pnode[1], M = params

		delta_q = 1.0/4.0
		qnode = np.linspace(self.q_min,1.0,5)

		pnode[0] = 2.0/delta_q - 2*(pnode[1]+pnode[2]+pnode[3]) - pnode[4]

		x_arr = np.atleast_1d(x)
		qdist = np.zeros_like(x_arr)

		for i in range(4):
			ind = np.where(x_arr >= qnode[i])[0]
			if ind.any():
				qdist[ind] = pnode[i] + (x_arr[ind]-qnode[i])*(pnode[i+1]-pnode[i])/(qnode[i+1]-qnode[i]) 

		return qdist


	def q_distribution_hist(self,x,params):

		pnode = params[:self.n_q_hist_bins]
		pnode_dot = params[self.n_q_hist_bins:2*self.n_q_hist_bins]
		M = params[-1]

		dM = M-self.M_ref

		delta_q = (1.0-self.q_min)/self.n_q_hist_bins
		qnode = self.q_min + np.arange(self.n_q_hist_bins)*delta_q

		x_arr = np.atleast_1d(x)
		qdist = np.zeros_like(x_arr)

		for i in range(self.n_q_hist_bins):
			ind = np.where(x_arr >= qnode[i])[0]
			if ind.any():
				qdist[ind] = pnode[i] + pnode_dot[i]*dM

		return qdist


	def q_distribution(self,x,params):

		"""
		Evaluate the binary mass-ratio distribution function at x.

		"""

		assert self.q_model in ['quadratic','single_power','power','legendre','piecewise','hist']

		if self.q_model == 'quadratic':

			qdist = self.q_distribution_quadratic(x,params)

		if self.q_model == 'single_power':

			qdist = self.q_distribution_single_power(x,params)

		if self.q_model == 'power':

			qdist = self.q_distribution_power(x,params)

		if self.q_model == 'legendre':

			qdist = self.q_distribution_legendre(x,params)

		if self.q_model == 'piecewise':

			qdist = self.q_distribution_piecewise(x,params)

		if self.q_model == 'hist':

			qdist = self.q_distribution_hist(x,params)

		if type(x) == np.ndarray:
			return qdist
		else:
			return qdist[0]


	def q_distribution_sampler(self,params):

		"""Return a function that maps the range (0,1) onto the binary mass-ratio distribution function.""" 

		q = np.linspace(self.q_min,1.0,1001)

		y = self.q_distribution(q,params)			

		y[y<0.0] = 0.0
		y_cumulative = np.cumsum(y)+np.arange(len(y))*1.e-6

		return PchipInterpolator(y_cumulative/y_cumulative[-1],q)


	def integrate_legendre(self,params,q1,q2):

		# Map q to 0 - 1 domain for legendre functions
		q1 = (q1-self.q_min)/(1.0-self.q_min) 
		q2 = (q2-self.q_min)/(1.0-self.q_min) 

		y = self.int_sl[0](q2) - self.int_sl[0](q1)
		for i in range(self.n_legendre):
			y += params[i]*(self.int_sl[i+1](q2) - self.int_sl[i+1](q1))

		return y


	def integrate_single_power(self,params,q1,q2):

		beta, q_u = params

		tan_beta = np.tan(beta)

		alpha = 1.0001 + self.sp_log_d_s/np.log10(q1)

		a1 = tan_beta/alpha

		alpha1 = alpha + 1.0

		a0 = 1.0 - a1/alpha1

		y = a0*(q2 - q1) + a1*(q2**alpha1 - q1**alpha1)/alpha1

		return y


	def integrate_quadratic(self,params,q1,q2):

		q0, a1 = params

		a0 = 1.0 - a1*(1.0-q0)**3/3.0

		y = a0*(q2 - q1)

		if q2 > q0:
			y += a1*(q2-q0)**3/3.0
		if q1 > q0:
			y -= a1*(q1-q0)**3/3.0

		return y


	def integrate_power(self,params,q1,q2):

		alpha1, alpha2, q0, a1, a2 = params

		c = (1.0 - a1*(q0-self.q_min)**alpha1/(2*(alpha1+1.0)) - a2*(1.0-q0)**alpha2/(2*(alpha2+1.0))) / (1.0-self.q_min)

		y = 0.0
		if q2 < q0:
			y += c*(q2-self.q_min) + a1*((q0-self.q_min)**(alpha1+1.0) - (q0-q2)**(alpha1+1.0))/(alpha1+1.0)
		else:
			y += c*(q2-self.q_min) + a1*(q0-self.q_min)**(alpha1+1.0)/(alpha1+1.0) + a2*(q2-q0)**(alpha2+1.0)/(alpha2+1.0)
		if q1 < q0:
			y -= c*(q1-self.q_min) + a1*((q0-self.q_min)**(alpha1+1.0) - (q0-q1)**(alpha1+1.0))/(alpha1+1.0)
		else:
			y -= c*(q1-self.q_min) + a1*(q0-self.q_min)**(alpha1+1.0)/(alpha1+1.0) + a2*(q1-q0)**(alpha2+1.0)/(alpha2+1.0)

		return y


	def integrate_piecewise(self,params,q1,q2):

		def integral_within_bin(qa,qb,qlow,qhigh,plow,phigh):
			s = (phigh-plow)/(qhigh-qlow)
			return (plow-s*qlow)*(qb-qa) + 0.5*s*(qb**2 - qa**2)

		pnode = np.zeros(5)
		pnode[4], pnode[3], pnode[2], pnode[1] = params

		delta_q = 1.0/4.0
		qnode = np.linspace(self.q_min,1.0,5)

		pnode[0] = 2.0/delta_q -2*np.sum(pnode[1:-1]) - pnode[-1]

		ind = np.where(q1 - qnode >= 0.0)[0][-1]
		q1low = qnode[ind]
		p1low = pnode[ind]
		q1high = qnode[ind+1]
		p1high = pnode[ind+1]
		ind = np.where(q2 - qnode > 0.0)[0][-1]
		q2low = qnode[ind]
		p2low = pnode[ind]
		q2high = qnode[ind+1]
		p2high = pnode[ind+1]

		if q1low == q2low:

			y = integral_within_bin(q1,q2,q1low,q1high,p1low,p1high)

		else:

			y = integral_within_bin(q1,q1high,q1low,q1high,p1low,p1high)

			ind = np.where(qnode == q1high)[0]
			while qnode[ind] < q2low:
				s = (pnode[ind+1]-pnode[ind])/(qnode[ind+1]-qnode[ind])
				y += (pnode[ind]-s*qnode[ind])*(qnode[ind+1]-qnode[ind]) + 0.5*s*(qnode[ind+1]**2 - qnode[ind]**2)
				ind += 1

			y += integral_within_bin(q2low,q2,q2low,q2high,p2low,p2high)

		return y


	def integrate_hist(self,params,q1,q2):

		pnode = np.zeros(self.n_q_hist_bins)
		pnode[self.n_q_hist_bins-1:0:-1] = params

		delta_q = self.q_min/self.n_q_hist_bins
		qnode = self.q_min + np.arange(self.n_q_hist_bins)*delta_q

		pnode[0] = self.n_q_hist_bins - np.sum(pnode[1:])

		q_int = np.zeros(2)

		for i, q in enumerate([q1,q2]):

			ind = np.where(qnode<=q)[0][-1]

			if ind > 0:
				q_int[i] = np.sum(pnode[:ind])*delta_q

			q_int[i] += pnode[ind]*(q-qnode[ind])

		y = q_int[1] - q_int[0]

		return y


	def q_distribution_integral(self,params,q1,q2):

		if q1 < q2:
			x = np.linspace(q1,q2,1000)
			y = self.q_distribution(x,params)
			return np.sum(y)*(x[1]-x[0])
		return 0.0


	def q_distribution_integral_explicit(self,params,q1,q2):

		assert self.q_model in ['quadratic','single_power','power','legendre','piecewise','hist']

		assert np.min([q1,q2]) >= self.q_min
		assert np.max([q1,q2]) <= 1.0
		assert q2 >= q1

		if q2 == q1:
			return 0.0

		if self.q_model == 'legendre':

			return self.integrate_lendre(params,q1,q2)

		if self.q_model == 'quadratic':

			return self.integrate_quadratic(params,q1,q2)

		if self.q_model == 'single_power':

			return self.integrate_single_power(params,q1,q2)

		if self.q_model == 'power':

			return self.integrate_power(params,q1,q2)

		if self.q_model == 'piecewise':

			return self.integrate_piecewise(params,q1,q2)

		if self.q_model == 'hist':

			return self.integrate_hist(params,q1,q2)



	def compute_observational_scatter(self,mag,nbins=10):

		"""Return an average covariance at magnitude mag, based on the data."""

		from scipy.optimize import curve_fit

		x1 = np.linspace(self.data.magnitude_min,self.data.magnitude_max,nbins+1)
		xmag = np.linspace(self.data.magnitude_min,self.data.magnitude_max,1001)

		xm = np.zeros(nbins)

		ycov = np.zeros((nbins,2,2))

		cov = np.zeros((len(mag),2,2))

		for i in range(nbins):
			pts = np.where((self.data.magnitude >= x1[i]) & (self.data.magnitude <= x1[i+1]))
			cov[pts,:,:] = np.median(self.data.cov[pts],axis=0)
			print('mag',0.5*(x1[i]+x1[i+1]))
			print('cov',np.median(self.data.cov[pts],axis=0))
		print(self.data.magnitude_type)

		# for i in range(nbins):
		# 	pts = np.where((self.data.magnitude >= x1[i]) & (self.data.magnitude <= x1[i+1]))
		# 	ycov[i] = np.median(self.data.cov[pts],axis=0)
		# 	xm[i] = 0.5*(x1[i]+x1[i+1])

		# fig, ax = plt.subplots(2,2,figsize=(8,8))

		# for i in range(2):
		# 	for j in range(2):
		# 		yoffset = 0.99*np.min(ycov[:,i,j])
		# 		f, _ = curve_fit(lambda t,a,b: a*np.exp(b*t),  xm,  ycov[:,i,j]-yoffset)
		# 		cov[:,i,j] = f[0]*np.exp(f[1]*mag) + yoffset
		# 		ax[i,j].scatter(xm,ycov[:,i,j],s=10)
		# 		ax[i,j].plot(xmag,f[0]*np.exp(f[1]*xmag) + yoffset,'r-')

		# plt.savefig('cov.png')


		return cov


	def model_realisation(self,p,n,add_observational_scatter=True,outliers=True):

		"""
		Compute a random (magnitude, colour) realisation of the model paramterised by p for n stars.

		Also return 		star_type = 0, 1, 2 for single stars, binaries, outliers.
		""" 

		fb0, fb1, ft0, oc, om, ocv, omv, ocov, fo, logh, hdot = p[self.b_index:self.i_index]

		# If we are modelling the isochrone offsets, then we need to recompute the isochrone and basis functions.
		if self.model_isochrone_correction:
			self.iso.recompute(p[self.i_index:])

		star_type = np.zeros(n)
		mag = np.zeros(n)
		colour = np.zeros(n)

		if np.sum(self.freeze[self.i_index-8:self.i_index-3]) < 5:
			outlier_description = np.ascontiguousarray([oc, om, ocv, omv, ocov],dtype=np.float64)
		else:
			outlier_description = self.outlier_description

		if outliers:

			n_outliers = int(round(fo*n))
			cov = np.zeros((2,2))
			cov[0,0] = outlier_description[2];
			cov[1,1] = outlier_description[3];
			cov[0,1] = outlier_description[4];
			cov[1,0] = outlier_description[4];

			d = np.random.multivariate_normal(outlier_description[:2], cov, n_outliers)

			colour[:n_outliers] = d[:,0]
			mag[:n_outliers] = d[:,1]
			star_type[:n_outliers] = 2

		else:

			n_outliers = 0
			fo = 0.0


		# Draw n primaries
		M_sampler = self.M_distribution_sampler(p[:self.q_index])
		M1 = M_sampler(np.random.rand(n))

		# Compute their binary probabilities and some random numbers
		fb = fb0 + fb1*(self.mass_slice[1] - M1)
		r = np.random.rand(n)

		q1 = np.zeros_like(M1)
		q2 = np.zeros_like(M1)

		for i in range(n_outliers,n):

			args = p[self.q_index:self.b_index].tolist() + [M1[i]]
			q_sampler = self.q_distribution_sampler(args)

			if r[i] <= fb[i]/(1.0-fo):

				star_type[i] = 1

				q1[i] = q_sampler(np.random.rand())

			if self.include_triples and r[i] > fb[i]/(1.0-fo) and r[i] - fb[i]/(1.0-fo) <= ft/(1.0-fo):

				star_type[i] = 3

				q1[i] = q_sampler(np.random.rand())
				q2[i] = q_sampler(np.random.rand())

		mag[n_outliers:], colour[n_outliers:] = self.iso.triple(M1[n_outliers:],q1[n_outliers:],q2[n_outliers:])

		if add_observational_scatter:

			h = 10.0**logh + hdot*(mag-self.h_magnitude_ref)

			cov = self.compute_observational_scatter(mag)

			for i in range(len(mag)):
				if self.error_scale_type == 'both':
					dcov = h[i]**2*cov[i]
				else:
					dcov = cov[i]
					dcov[0,0] *= h[i]**2
				z = np.random.multivariate_normal(mean=np.zeros(2), cov=dcov, size=1)
				colour[i] += z[0][0]
				mag[i] += z[0][1]


		print('Model realisation')
		print('Mass range:',np.min(M1),np.max(M1))
		print('Magnitude range',np.min(mag),np.max(mag))
		print(n,'total stars including',n_outliers,'outliers')

		return mag, colour, star_type


	def M_gauss(self,params):
		
		"""Return the (positive) coefficients for mapping the mass basis functions onto the mass function."""

		My = self.M_distribution(self.Mx,params)
		
		Mb = np.dot(self.Mbf_sigma2,My)
		#Mb = np.zeros(self.n_bf)
		#for k in range(self.n_bf):
		#		Mb[k] = np.sum(My*self.Mbf[k,:]/self.Msigma**2)
		
		Ma, resid = nnls(self.MA,Mb,maxiter=5000)
		#result = tntnn(self.MA,Mb)
		#Ma = result.x

		mfit = np.dot(Ma,self.Mbf)
		#mfit  = np.zeros_like(self.Mx)
		#for k in range(self.n_bf):
		#	mfit += Ma[k]*self.Mbf[k,:]

		norm_c = np.sum(mfit*(self.Mx[1]-self.Mx[0]))

		return Ma/norm_c


	def q_gauss(self,params):
				
		"""Return the (positive) coefficients for mapping the q basis functions onto the q distribution function."""

		qy = self.q_distribution(self.qx,params)

		qb = np.dot(self.qbf_sigma2,qy)
		# qb = np.zeros(self.n_bf)
		# for k in range(self.n_bf):
		# 	qb[k] = np.sum(qy*self.qbf[k,:]/self.qsigma**2)
		
		qa, resid = nnls(self.qA,qb,maxiter=5000)
		#result = tntnn(self.qA,qb)
		#qa = result.x

		qfit = np.dot(qa,self.qbf)
		# qfit  = np.zeros_like(self.qx)
		# for k in range(self.n_bf):
		# 	qfit += qa[k]*self.qbf[k,:]

		norm_c = np.sum(qfit*(self.qx[1]-self.qx[0]))

		return qa/norm_c


	def precalc(self,params):
		
		"""Return the vector of mass basis function coefficients (for single stars) and 
		the grids of (M,q) basis function coefficients for binary stars and for triple stars. These are multiplied
		by the single-star, binary-star and triple-star fractions respectively."""

		fb0, fb1, ft0, oc, om, ocv, omf, ocov, fo, logh, hdot = params[self.b_index:self.i_index]

		# fb and ft_fb are vectors here
		fb = fb0 + fb1*(self.mass_slice[1] - self.M0)
		ft_fb = ft0 / fb

		PMQ = np.zeros(self.n_bf**2)
		Ma = self.M_gauss(params[:self.q_index])

		if self.include_triples:

			PMQ_t = np.zeros(self.n_bf**3)

			if (self.q_model == 'legendre') and (np.sum(self.freeze[self.q_index+self.n_legendre:self.b_index]) < self.n_legendre) or \
						(self.q_model == 'single_power') and (np.sum(self.freeze[self.q_index+2:self.b_index]) < 2) or \
						(self.q_model == 'quadratic') and (np.sum(self.freeze[self.q_index+2:self.b_index]) < 2):

				# p(q) is a function of mass

				for i in range (self.n_bf):
					args = params[self.q_index:self.b_index].tolist() + [self.M0[i]]
					qa = self.q_gauss(args)
					for j in range(self.n_bf):
						PMQ[i+j*self.n_bf] = Ma[i]*qa[j]*fb[i]
						for k in range(self.n_bf):
							PMQ_t[i+j*self.n_bf+k*self.n_bf**2] = PMQ[i+j*self.n_bf]*ft_fb[i]*qa[k]

			else:

				args = params[self.q_index:self.b_index].tolist() + [self.M0[0]]
				qa = self.q_gauss(args)
				for i in range (self.n_bf):
					for j in range(self.n_bf):
						PMQ[i+j*self.n_bf] = Ma[i]*qa[j]*fb[i]
						for k in range(self.n_bf):
							PMQ_t[i+j*self.n_bf+k*self.n_bf**2] = PMQ[i+j*self.n_bf]*ft_fb[i]*qa[k]

		else:

			PMQ_t = None

			if (self.q_model == 'legendre') and (np.sum(self.freeze[self.q_index+self.n_legendre:self.b_index]) < self.n_legendre) or \
						(self.q_model == 'single_power') and (np.sum(self.freeze[self.q_index+2:self.b_index]) < 2) or \
						(self.q_model == 'quadratic') and (np.sum(self.freeze[self.q_index+2:self.b_index]) < 2):

				# p(q) is a function of mass

				for i in range (self.n_bf):
					args = params[self.q_index:self.b_index].tolist() + [self.M0[i]]
					qa = self.q_gauss(args)
					for j in range(self.n_bf):
						PMQ[i+j*self.n_bf] = Ma[i]*qa[j]*fb[i]

			else:

				args = params[self.q_index:self.b_index].tolist() + [self.M0[0]]
				qa = self.q_gauss(args)
				PMQ = np.dot(qa.reshape(self.n_bf,1),fb.reshape(1,self.n_bf)*Ma.reshape(1,self.n_bf)).reshape((self.n_bf**2,))


		return Ma*(1.0-fb-fo-ft0), PMQ, PMQ_t


	def lnlikelihood(self,params):

		"""Call the external CUDA function to evaluate the likelihood function."""

		#print('Computing likelihood for',params)
		
		p = self.default_params.copy()
		p[self.freeze==0] = params

		oc, om, ocv, omf, ocov, fo, logh, hdot = p[self.i_index-8:self.i_index]

		if np.sum(self.freeze[self.i_index-8:self.i_index-3]) < 5:
			outlier_description = np.ascontiguousarray([oc, om, ocv, omf, ocov],dtype=np.float64)
		else:
			outlier_description = self.outlier_description

		blob = np.zeros(3)
		blob[1] = self.ln_prior(params)
		blob[2] = params[3]

		# If we are modelling the isochrone offsets, then we need to recompute the isochrone and basis functions.
		if self.model_isochrone_correction:
			self.iso.recompute(p[self.i_index:])
			self.upload_basis_functions_to_GPU()

		assert self.q_model in ['quadratic','single_power','power','legendre','piecewise','hist']

		# Check that the parameters generate positive q distributions for all masses, and a positive M distribution.
		m_dist_test = self.M_distribution(np.linspace(self.mass_slice[0],self.mass_slice[1],101),p[:self.q_index])
		if np.min(m_dist_test) < 0.0:
			blob[0] = self.neginf
			return self.neginf, blob

		if self.q_model == 'legendre':

			qx = np.linspace(0.0,1.0,101)
			for MM in np.linspace(self.mass_slice[0],self.mass_slice[1],31).tolist():
				args = p[self.q_index:self.b_index].tolist() + [MM]
				q_dist_test = self.q_distribution(qx,args)
				if np.min(q_dist_test) < 0.0:
					with open(self.prefix+'.err', 'a') as f:
						#f.write('Negative q dist for:')
						f.write(np.array2string(params,max_line_width=1000).strip('[]\n')+'\n')
					blob[0] = self.neginf
					return self.neginf, blob

		try:
			P_i, PMQ, PMQ_t = self.precalc(p)
		except:
			print("Error in precalc for p =",p)
			raise


		c_P_i = np.ascontiguousarray(P_i.astype(np.float64))
		c_PMQ = np.ascontiguousarray(PMQ.astype(np.float64))

		if PMQ_t is not None:
			c_PMQ_t = np.ascontiguousarray(PMQ_t.astype(np.float64))
		else:
			c_PMQ_t = np.array([0.0]).astype(np.float64)

		n_pts = len(self.data.magnitude)
		
		blockshape = (int(256),1, 1)
		gridshape = (n_pts, 1)
		#gridshape = (1, 1)

		lnP_k = np.zeros(n_pts*5).astype(np.float64)

		htype = 0
		if self.error_scale_type == 'both':
			htype = 1

		mtype = 0
		if self.include_triples:
			mtype = 1

		likelihood(self.DMQ_gpu,self.SMQ_gpu,self.data.col_mag_gpu,self.data.c_cov_gpu,self.drv.In(c_P_i), self.drv.In(c_PMQ), self.drv.In(c_PMQ_t), self.drv.In(outlier_description),np.int32(htype),np.int32(mtype),np.float64(logh), np.float64(hdot), np.float64(self.h_magnitude_ref),np.float64(fo), self.drv.InOut(lnP_k), block=blockshape, grid=gridshape)

		lnP = np.sum(lnP_k[:n_pts])

		#sys.exit()

		#print('ln P =',lnP)
		#print()

		#sys.exit()


		self.lnP_k = lnP_k.reshape(n_pts,5,order='F')

		if not(np.isfinite(lnP)):
			with open(self.prefix+'.err', 'a') as f:
				f.write(np.array2string(params,max_line_width=1000).strip('[]\n')+'\n')
			blob[0] = self.neginf
			return self.neginf, blob

		blob[0] = lnP
		return lnP, blob


	def neglnlikelihood(self,params):
		lnl, _ =  self.lnlikelihood(params)
		return -lnl


	def ln_prior(self,params):

		from scipy.stats import norm, truncnorm

		p = self.default_params.copy()
		p[self.freeze==0] = params

		assert self.q_model in ['quadratic','single_power','power','legendre','piecewise','hist']
		assert self.m_model in ['power']

		lnp = self.ln_prior_mass_power(p[:self.q_index])

		if self.q_model == 'quadratic':

			lnp += self.ln_prior_q_quadratic(p[self.q_index:self.b_index])

		if self.q_model == 'single_power':

			lnp += self.ln_prior_q_single_power(p[self.q_index:self.b_index])

		if self.q_model == 'power':

			lnp += self.ln_prior_q_power(p[self.q_index:self.b_index])

		if self.q_model == 'legendre':

			lnp += self.ln_prior_q_legendre(p[self.q_index:self.b_index])

		if self.q_model == 'piecewise':

			lnp += self.ln_prior_q_piecewise(p[self.q_index:self.b_index])

		if self.q_model == 'hist':

			lnp += self.ln_prior_q_hist(p[self.q_index:self.b_index])

		lnp += self.ln_prior_q_general(p[self.b_index:self.i_index])

		if self.model_isochrone_correction:
			lnp += self.ln_prior_isochrone_correction(p[self.i_index:])

		return lnp



	def prior_transform_mass_power(self,u):

		# params are log k, M0, gamma, c0, c1

		x = self.default_params.copy()[:5]

		i = 0

		if not self.freeze[0]:
			# log k
			#x[0] = norm.ppf(u[i], loc=1.7, scale=0.2)
			x[0] = 0.5*u[i]+1.5
			i += 1
		if not self.freeze[1]:
			# M0
			#x[1] = norm.ppf(u[i], loc=self.mass_slice[0]+0.1,scale=0.5)
			x[1] = 0.2*u[i] - 0.1 + self.mass_slice[0]
			i += 1
		if not self.freeze[2]:
			# gamma
			#x[2] = truncnorm.ppf(u[i], -2.35, 6.0-2.35, loc=2.35, scale=1.0)
			x[2] = 6.0*u[i]
			i += 1
		if not self.freeze[3]:
			# c0
			#x[3] = truncnorm.ppf(u[i],0.0,1.0/0.5,loc=0.0,scale=0.5)
			x[3] = 13.0*u[i] - 1.0
			i += 1
		if not self.freeze[4]:
			# c1
			#x[4] = truncnorm.ppf(u[i],-1.0/0.2,1.0/0.2,loc=0.0,scale=0.2)
			x[4] = 12.0*u[i]-6.0
			i += 1

		return i, x


	def ln_prior_mass_power(self,p):

		log_k, M0, gamma, c0, c1 = p

		if not self.freeze[0]:
			if log_k < 1.5 or log_k > 2.0:
				return self.neginf
		if not self.freeze[1]:
			if M0 < self.mass_slice[0] - 0.1 or M0 > self.mass_slice[0] + 0.1:
				return self.neginf
		if gamma < 0.0 or gamma > 6.0 or c0 < -1.0 or c0 > 12.0 or c1 < -6.0 or c1 > 6.0:  
			return self.neginf

		return 0.0


	def prior_transform_q_quadratic(self,u,i):

		# params are q0, a1, q0_dot, a1_dot

		x = self.default_params.copy()

		if not self.freeze[self.q_index]:
			# q0
			x[self.q_index] = 0.6*u[i] + 0.2
			i += 1
		if not self.freeze[self.q_index+1]:
			# a1
			a1_min = -3.0/((1.0-x[self.q_index])**2*(2.0+x[self.q_index]))
			a1_max = 3.0/(1.0-x[self.q_index])**3
			x[self.q_index+1] = (a1_max-a1_min)*u[i] + a1_min
			i += 1
		if not self.freeze[self.q_index+2]:
			# q0_dot
			q0_dot_min = np.max([-(0.8-x[self.q_index])/self.delta_M,(0.2 - x[self.q_index])/self.delta_M])
			q0_dot_max = np.min([-(0.2-x[self.q_index])/self.delta_M,(0.8 - x[self.q_index])/self.delta_M])
			x[self.q_index+2] = (q0_dot_max-q0_dot_min)*u[i] + q0_dot_min
			i += 1
		if not self.freeze[self.q_index+3]:
			# a1_dot
			a1_dot_min = np.max([-(a1_max-x[self.q_index+1])/self.delta_M,(a1_min - x[self.q_index+1])/self.delta_M])
			a1_dot_max = np.min([-(a1_min-x[self.q_index+1])/self.delta_M,(a1_max - x[self.q_index+1])/self.delta_M])
			x[self.q_index+3] = (a1_dot_max-a1_dot_min)*u[i] + a1_dot_min
			i += 1

		return i, x[self.q_index:self.b_index]


	def ln_prior_q_quadratic(self,p):

		q0, a1, q0_dot, a1_dot = p

		if q0 < 0.2 or q0 > 0.8:
			return self.neginf

		a1_min = -3.0/((1.0-q0)**2*(2.0+q0))
		a1_max = 3.0/(1.0-q0)**3
		if a1 < a1_min or a1 > a1_max:
			return self.neginf

		q0_dot_min = np.max([-(0.8-q0)/self.delta_M,(0.3 - q0)/self.delta_M])
		q0_dot_max = np.min([-(0.2-q0)/self.delta_M,(0.8 - q0)/self.delta_M])
		if q0_dot < q0_dot_min  or q0_dot > q0_dot_max:
			return self.neginf

		a1_dot_min = np.max([-(a1_max-a1)/self.delta_M,(a1_min - a1)/self.delta_M])
		a1_dot_max = np.min([-(a1_min-a1)/self.delta_M,(a1_max - a1)/self.delta_M])
		if a1_dot < a1_dot_min  or a1_dot > a1_dot_max:
			return self.neginf

		return 0.0


	def prior_transform_q_single_power(self,u,i):

		# params are beta, q1, beta_dot, q1_dot

		x = self.default_params.copy()

		if not self.freeze[self.q_index]:
			# beta
			x[self.q_index] = np.pi*u[i] - np.pi/2.0
			i += 1
		if not self.freeze[self.q_index+1]:
			# q1
			x[self.q_index+1] = 0.50*u[i]
			i += 1
		if not self.freeze[self.q_index+2]:
			# beta_dot
			beta_dot_min = np.max([-(np.pi/2-x[self.q_index])/self.delta_M,(-np.pi/2 - x[self.q_index])/self.delta_M])
			beta_dot_max = np.min([-(-np.pi/2-x[self.q_index])/self.delta_M,(np.pi/2 - x[self.q_index])/self.delta_M])
			x[self.q_index+2] = (beta_dot_max-beta_dot_min)*u[i] + beta_dot_min
			i += 1
		if not self.freeze[self.q_index+3]:
			# q1_dot
			q1_dot_min = np.max([-(0.99-x[self.q_index+1])/self.delta_M,(0.01 - x[self.q_index+1])/self.delta_M])
			q1_dot_max = np.min([-(0.01-x[self.q_index+1])/self.delta_M,(0.99 - x[self.q_index+1])/self.delta_M])
			x[self.q_index+3] = (q1_dot_max-q1_dot_min)*u[i] + q1_dot_min
			i += 1

		return i, x[self.q_index:self.b_index]


	def ln_prior_q_single_power(self,p):

		beta, q1, beta_dot, q1_dot = p

		if beta < -np.pi/2 or beta > np.pi/2:
			return self.neginf
		if q1 > 0.5:
			return self.neginf
		beta_dot_min = np.max([-(np.pi/2-beta)/self.delta_M,(-np.pi/2 - beta)/self.delta_M])
		beta_dot_max = np.min([-(-np.pi/2-beta)/self.delta_M,(np.pi/2 - beta)/self.delta_M])
		if beta_dot < beta_dot_min or beta_dot > beta_dot_max:
			return self.neginf
		q1_dot_min = np.max([-(0.99-q1)/self.delta_M,(0.01 - q1)/self.delta_M])
		q1_dot_max = np.min([-(0.01-q1)/self.delta_M,(0.99 - q1)/self.delta_M])
		if q1_dot < q1_dot_min  or q1_dot > q1_dot_max:
			return self.neginf

		return 0.0


	def prior_transform_q_power(self,u,i):

		# params are alpha1, alpha2, q0, a1, a2

		x = self.default_params.copy()

		if not self.freeze[self.q_index]:
			# alpha1
			#x[self.q_index] = truncnorm.ppf(u[i], -0.9/10.0, 3.0/10.0, loc=2.0, scale=10.0)
			x[self.q_index] = 33.9*u[i] + 1.1
			i += 1
		if not self.freeze[self.q_index+1]:
			# alpha2
			#x[self.q_index+1] = truncnorm.ppf(u[i], -0.9/10.0, 3.0/10.0, loc=2.0, scale=10.0)
			x[self.q_index+1] = 33.9*u[i] + 1.1
			i += 1
		if not self.freeze[self.q_index+2]:
			# q0
			#x[self.q_index+2] = truncnorm.ppf(u[i], np.max([self.q_min,-0.49/(1.0-self.q_min)])/10.0, np.min([1.0,0.49/(1.0-self.q_min)])/10.0, loc=(0.5-self.q_min)/(1.0-self.q_min), scale=10.0)
			m1 = np.max([self.q_min,-0.49/(1.0-self.q_min)])
			m2 = np.min([1.0,0.49/(1.0-self.q_min)])
			x[self.q_index+2] = ((m2-m1)*u[i] + m1)
			i += 1
		if not self.freeze[self.q_index+3]:
			# a1
			alpha1 = x[self.q_index]
			alpha2 = x[self.q_index+1]
			q0 = x[self.q_index+2]
			fac = (1.0-self.q_min) / ( (1.0-q0)**(alpha2+1.0) * (self.q_min - (alpha2+q0)/(alpha2+1.0)) )
			a1min_list = [-50.0]
			a1max_list = [50.0]
			if fac > 0.0:
				a1min_list.append((alpha1+1.0)*(q0-self.q_min)**(-(alpha1+1.0)))
			if fac < 0.0:
				a1max_list.append((alpha1+1.0)*(q0-self.q_min)**(-(alpha1+1.0)))
			fac = (q0-self.q_min)**alpha1 * ( ( (alpha1+1.0)*(1.0-self.q_min) - (q0 - self.q_min) ) * ( 1.0 - q0 - (alpha2+1.0)*(1.0-self.q_min) ) + (1.0-q0)*(q0-self.q_min) ) / ( (alpha1+1.0)*(alpha2+1.0)*(1.0-self.q_min) )        
			if fac > 0.0:
				a1max_list.append(1.0/fac)
			if fac < 0.0:
				a1min_list.append(1.0/fac)
			a1min = np.max(a1min_list)
			a1max = np.min(a1max_list)
			#x[self.q_index+3] = truncnorm.ppf(u[i], a1min/50.0, a1max/50.0, loc=0.0, scale=50.0)
			x[self.q_index+3] = (a1max-a1min)*u[i] + a1min
			i += 1
		if not self.freeze[self.q_index+4]:
			# a2
			a1 = x[self.q_index+3]
			alpha1 = x[self.q_index]
			alpha2 = x[self.q_index+1]
			q0 = x[self.q_index+2]
			a2min =  np.max([(1.0 - a1*(q0-self.q_min)**(alpha1+1.0)/(alpha1+1.0)) / ((1.0-q0)**alpha2 * ( (1.0-q0)/(alpha2+1.0) - (1.0-self.q_min) ) ) ])
			a2max = np.min([(alpha2+1.0) * ( 1.0 + a1*(q0-self.q_min)**alpha1 * ( 1.0 - self.q_min - (q0-self.q_min)/(alpha1+1.0) )) / (1.0-q0)**(alpha2+1.0) , \
						(alpha2+1.0)*(1.0-q0)**(-(alpha2+1.0)) * (1.0 - a1*(q0-self.q_min)**(alpha1+1.0)/(alpha1+1.0)) ])
			#x[self.q_index+4] = truncnorm.ppf(u[i], a2min/50.0, a2max/50.0, loc=0.0, scale=50.0)
			x[self.q_index+4] = (a2max-a2min)*u[i] + a2min
			i += 1

		return i, x[self.q_index:self.b_index]


	def ln_prior_q_power(self,p):

		alpha1, alpha2, q0, a1, a2 = p

		fac = (1.0-self.q_min) / ( (1.0-q0)**(alpha2+1.0) * (self.q_min - (alpha2+q0)/(alpha2+1.0)) )
		a1min_list = [-50.0]
		a1max_list = [50.0]
		if fac > 0.0:
			a1min_list.append((alpha1+1.0)*(q0-self.q_min)**(-(alpha1+1.0)))
		if fac < 0.0:
			a1max_list.append((alpha1+1.0)*(q0-self.q_min)**(-(alpha1+1.0)))
		fac = (q0-self.q_min)**alpha1 * ( ( (alpha1+1.0)*(1.0-self.q_min) - (q0 - self.q_min) ) * ( 1.0 - q0 - (alpha2+1.0)*(1.0-self.q_min) ) + (1.0-q0)*(q0-self.q_min) ) / ( (alpha1+1.0)*(alpha2+1.0)*(1.0-self.q_min) )        
		if fac > 0.0:
			a1max_list.append(1.0/fac)
		if fac < 0.0:
			a1min_list.append(1.0/fac)
		a1min = np.max(a1min_list)
		a1max = np.min(a1max_list)

		a2min =  np.max([(1.0 - a1*(q0-self.q_min)**(alpha1+1.0)/(alpha1+1.0)) / ((1.0-q0)**alpha2 * ( (1.0-q0)/(alpha2+1.0) - (1.0-self.q_min))) ]) 
		a2max = np.min([(alpha2+1.0) * ( 1.0 + a1*(q0-self.q_min)**alpha1 * ( 1.0 - self.q_min - (q0-self.q_min)/(alpha1+1.0) )) / (1.0-q0)**(alpha2+1.0) , \
					(alpha2+1.0)*(1.0-q0)**(-(alpha2+1.0)) * (1.0 - a1*(q0-self.q_min)**(alpha1+1.0)/(alpha1+1.0)) ])

		m1 = np.max([self.q_min,-0.49/(1.0-self.q_min)])
		m2 = np.min([1.0,0.49/(1.0-self.q_min)])

		if alpha1  < 1.1 or alpha1 > 35.0 or alpha2 < 1.1 or alpha2 > 35.0 or q0 < m1 or q0 >  m2 or a1 < a1min or a1 > a1max or a2 < a2min or a2 > a2max:
			return self.neginf 

		return 0.0


	def prior_transform_q_legendre(self,u,i):

		from scipy.stats import expon, gamma

		# params are a1, a2, a3, ..., a1_dot, a2_dot, a3_dot, ...

		x = self.default_params.copy()

		# hyperparameter for regularisation
		#alpha = 2.0
		scale = 3.0
		if not self.freeze[self.q_index]:
			#x[self.q_index] = 4.0*u[i] + 1.0
			#x[self.q_index] = 3.0*u[i]
			#x[self.q_index]  = expon.ppf(u[i],scale=scale)
			x[self.q_index]  = gamma.ppf(u[i],2.0,scale=scale)
			i += 1

		#alpha = 10.0**x[self.q_index]
		alpha = x[self.q_index]



		for j in range(1, self.n_legendre+1):

			if not self.freeze[self.q_index+j]:

				#x[self.q_index+i] = 6.0*u[i] - 3.0
				#x[self.q_index+j] = norm.ppf(u[i], loc=0.0, scale=alpha**(-j+1))
				x[self.q_index+j] = norm.ppf(u[i], loc=0.0, scale=alpha)
				i += 1

		for j in range(1,self.n_legendre+1):

			if not self.freeze[self.q_index+self.n_legendre+j]:

				x[self.q_index+self.n_legendre+j] = (10.0*u[self.n_legendre+i]-5.0) * 0.1/self.delta_M
				i += 1


		return i, x[self.q_index:self.b_index]


	def ln_prior_q_legendre(self,p):

		from scipy.stats import expon, gamma

		scale = 3.0

		# hyperparameter for regularisation
		#alpha = 2.0
		#alpha = 10.0**p[0]
		alpha = p[0]

		a = p[1:self.n_legendre+1]
		adot = p[self.n_legendre+1:2*self.n_legendre+1]

		#if p[0] <= 0.0 or p[0] > 1.0:
		# if p[0] < 1.0 or p[0] > 5.0:
		# 	return self.neginf 

		for i in range(self.n_legendre):
			if adot[i] < -5.0*0.1/self.delta_M or adot[i] > 5.0*0.1/self.delta_M:
				return self.neginf

		lnp = 0.0
		lnp += expon.pdf(alpha,scale=scale)
		lnp += gamma.pdf(alpha,2.0,scale=scale)
		for i in range(self.n_legendre):
			#pp = norm.pdf(a[i], loc=0.0, scale=alpha**(-i))
			pp = norm.pdf(a[i], loc=0.0, scale=alpha)
			lnp += np.log(pp)

		return lnp


	def prior_transform_q_piecewise(self,u,i):

		# params are pq4, pq3, pq2, pq1 

		x = self.default_params.copy()

		if not self.freeze[self.q_index]:
			# pq4
			sc = 0.5
			#x[self.q_index] = truncnorm.ppf(u[i], -1.0/sc, 3.99/sc, loc=1.0, scale=sc)
			x[self.q_index] = 5.0*u[i] 
			i += 1
		if not self.freeze[self.q_index+1]:
			# pq3
			delta_q = 1.0/4.0
			pmax = 1.0/delta_q - x[self.q_index]/2.0 
			#x[self.q_index+1] = truncnorm.ppf(u[i], -1.0/sc, (pmax-1.0)/sc, loc=1.0, scale=sc)
			x[self.q_index+1] = pmax*u[i]
			q_int = delta_q * (x[self.q_index] + x[self.q_index+1])/2.0
			i += 1
		if not self.freeze[self.q_index+2]:
			# pq2
			pmax = (1.0-q_int)/delta_q - x[self.q_index+1]/2.0
			#x[self.q_index+2] = truncnorm.ppf(u[i], -1.0/sc, (pmax-1.0)/sc, loc=1.0, scale=sc)
			x[self.q_index+2] = pmax*u[i]
			q_int += delta_q * (x[self.q_index+1] + x[self.q_index+2])/2.0
			i += 1
		if not self.freeze[self.q_index+3]:
			# pq1
			pmax = (1.0-q_int)/delta_q - x[self.q_index+2]/2.0
			#x[self.q_index+3] = truncnorm.ppf(u[i], -1.0/sc, (pmax-1.0)/sc, loc=1.0, scale=sc)
			x[self.q_index+3] = pmax*u[i] 
			q_int += delta_q * (x[self.q_index+2] + x[self.q_index+3])/2.0
			i += 1

		return i, x[self.q_index:self.b_index]


	def ln_prior_q_piecewise(self,p):

		pq4, pq3, pq2, pq1 = p

		if pq4 < 0.0 or pq4 > 4.0:
			return self.neginf
		delta_q = 1.0/4.0
		pmax = 1.0/delta_q - pq4/2.0 
		if pq3 < 0.0 or pq3 > pmax - 1.0:
			return self.neginf
		q_int = delta_q * (pq4 + pq3)/2.0
		pmax = (1.0-q_int)/delta_q - pq3/2.0
		if pq2 < 0.0 or pq2 > pmax - 1.0:
			return self.neginf
		q_int = delta_q * (pq3 + pq2)/2.0
		pmax = (1.0-q_int)/delta_q - pq2/2.0
		if pq1 < 0.0 or pq1 > pmax - 1.0:
			return self.neginf
		return 0.0


	def prior_transform_q_hist_old(self,u,i):

		# params are pq4, pq3, pq2, pq1, pq4_dot, pq3_dot, pq2_dot, pq1_dot

		x = self.default_params.copy()

		if not self.freeze[self.q_index]:
			# pq4
			#sc = 0.5
			#x[self.q_index] = truncnorm.ppf(u[i], -1.0/sc, 3.99/sc, loc=1.0, scale=sc)
			delta_q = 1.0/5.0
			#pmax = 1.0/delta_q
			pmax = 2.0
			x[self.q_index] = u[i]*pmax
			i += 1
		if not self.freeze[self.q_index+1]:
			# pq3
			pmax = np.min([2.0,1.0/delta_q - x[self.q_index]]) 
			#x[self.q_index+1] = truncnorm.ppf(u[i], -1.0/sc, (pmax-1.0)/sc, loc=1.0, scale=sc)
			x[self.q_index+1] = pmax*u[i]
			i += 1
		if not self.freeze[self.q_index+2]:
			# pq2
			pmax = np.min([2.0,1.0/delta_q - x[self.q_index] - x[self.q_index+1]])
			#x[self.q_index+2] = truncnorm.ppf(u[i], -1.0/sc, (pmax-1.0)/sc, loc=1.0, scale=sc)
			x[self.q_index+2] = pmax*u[i]
			i += 1
		if not self.freeze[self.q_index+3]:
			# pq1
			pmax = np.min([2.0,1.0/delta_q - x[self.q_index] - x[self.q_index+1] - x[self.q_index+2]])
			#x[self.q_index+3] = truncnorm.ppf(u[i], -1.0/sc, (pmax-1.0)/sc, loc=1.0, scale=sc)
			x[self.q_index+3] = pmax*u[i]
			i += 1

		if not self.freeze[self.q_index+4]:
			# pq1_dot
			pmax = 1.0/delta_q
			p_dot_min = np.max([-(pmax-x[self.q_index])/self.delta_M,(0.00 - x[self.q_index])/self.delta_M])
			p_dot_max = np.min([-(0.00-x[self.q_index])/self.delta_M,(pmax - x[self.q_index])/self.delta_M])
			x[self.q_index+4] = (p_dot_max-p_dot_min)*u[i] + p_dot_min
			i += 1
		if not self.freeze[self.q_index+5]:
			# pq1_dot
			pmax = 1.0/delta_q - x[self.q_index] 
			p_dot_min = np.max([-(pmax-x[self.q_index])/self.delta_M,(0.00 - x[self.q_index])/self.delta_M])
			p_dot_max = np.min([-(0.00-x[self.q_index])/self.delta_M,(pmax - x[self.q_index])/self.delta_M])
			x[self.q_index+5] = (p_dot_max-p_dot_min)*u[i] + p_dot_min
			i += 1
		if not self.freeze[self.q_index+6]:
			# pq1_dot
			pmax = 1.0/delta_q - x[self.q_index] - x[self.q_index+1]
			p_dot_min = np.max([-(pmax-x[self.q_index])/self.delta_M,(0.00 - x[self.q_index])/self.delta_M])
			p_dot_max = np.min([-(0.00-x[self.q_index])/self.delta_M,(pmax - x[self.q_index])/self.delta_M])
			x[self.q_index+6] = (p_dot_max-p_dot_min)*u[i] + p_dot_min
			i += 1
		if not self.freeze[self.q_index+7]:
			# pq1_dot
			pmax = 1.0/delta_q - x[self.q_index] - x[self.q_index+1] - x[self.q_index+2]
			p_dot_min = np.max([-(pmax-x[self.q_index])/self.delta_M,(0.00 - x[self.q_index])/self.delta_M])
			p_dot_max = np.min([-(0.00-x[self.q_index])/self.delta_M,(pmax - x[self.q_index])/self.delta_M])
			x[self.q_index+7] = (p_dot_max-p_dot_min)*u[i] + p_dot_min
			i += 1

		return i, x[self.q_index:self.b_index]


	def prior_transform_q_hist(self,u,i):

		# params are pq4, pq3, pq2, pq1, pq4_dot, pq3_dot, pq2_dot, pq1_dot

		# Map 0 - 1 to inverse of cumulative beta distribution

		x = self.default_params.copy()
		N =self.n_q_hist_bins
		delta_q = 1.0/N

		for j in range(N):
			if not self.freeze[self.q_index+j]:
				# pq j
				#x[self.q_index+j] = 1.0 - (1.0 - u[i])**(1.0/(N-1))
				x[self.q_index+j] = 1.0 - (1.0 - u[i])**(1.0/(N-1))
				i += 1

		# This normalization ensures that the parameters are Dirichlet-distributed
		# rand_beta = 1.0 - (1.0 - np.random.rand())**(1.0/(N-1))
		# x[self.q_index:self.q_index+N-1] /= np.sum(x[self.q_index:self.q_index+N-1]) + rand_beta
		# x[self.q_index:self.q_index+N-1] *= N

		x[self.q_index:self.q_index+N] /= np.sum(x[self.q_index:self.q_index+N]) 
		x[self.q_index:self.q_index+N] *= N

		for j in range(N,2*N):
			if not self.freeze[self.q_index+j]:
				# pq j _dot
				pmax = 1.0/delta_q - np.sum(x[self.q_index:self.q_index+j-N+1])
				p_dot_min = np.max([-(pmax-x[self.q_index+j-N+1])/self.delta_M,(0.00 - x[self.q_index+j-N+1])/self.delta_M])
				p_dot_max = np.min([-(0.00-x[self.q_index+j-N+1])/self.delta_M,(pmax - x[self.q_index+j-N+1])/self.delta_M])
				x[self.q_index+j] = (p_dot_max-p_dot_min)*u[i] + p_dot_min
				i += 1

		return i, x[self.q_index:self.b_index]


	def ln_prior_q_hist(self,p):

		pnode = p[:self.n_q_hist_bins]
		pnode_dot = p[self.n_q_hist_bins:2*self.n_q_hist_bins]


		#pq4, pq3, pq2, pq1, pq4_dot, pq3_dot, pq2_dot, pq1_dot = p

		x = self.default_params.copy()
		N = self.n_q_hist_bins
		delta_q = 1.0/N

		#pnode[0] = N - np.sum(pnode[1:])

		pmax = 100*N
		for j in range(N):
			if pnode[j] < 0.0:
				return -1.e6*pnode[j]**2
			if pnode[j] > pmax:
				return self.neginf

		if np.sum(self.freeze[self.q_index+self.n_q_hist_bins:self.q_index+2*self.n_q_hist_bins]) < N:
			for j in range(N):
				# pq j _dot
				pmax = np.min([2,1.0/delta_q - np.sum(x[self.q_index+j:self.q_index+N])])
				p_dot_min = np.max([-(pmax-pnode[j])/self.delta_M,(0.0 - pnode[j])/self.delta_M])
				p_dot_max = np.min([-(0.0-pnode[j])/self.delta_M,(pmax - pnode[j])/self.delta_M])
				if pnode_dot[j] < p_dot_min  or pnode_dot[j]> p_dot_max:
					return self.neginf

		return N*np.log(N) + (N-2)*np.sum(np.log(1.0-pnode/N))


	def ln_prior_q_hist_old(self,p):

		pq4, pq3, pq2, pq1, pq4_dot, pq3_dot, pq2_dot, pq1_dot = p

		delta_q = 1.0/5.0
		#pmax = 1.0/delta_q
		pmax = 2.0
		if pq4 < 0.0 or pq4 > pmax:
			return self.neginf
		delta_q = 1.0/5.0
		pmax = np.min([2.0,1.0/delta_q - pq4]) 
		if pq3 < 0.0 or pq3 > pmax:
			return self.neginf
		pmax = np.min([2.0,1.0/delta_q - pq4 - pq3])
		if pq2 < 0.0 or pq2 > pmax:
			return self.neginf
		pmax = np.min([2.0,1.0/delta_q - pq4 - pq3 -pq2])
		if pq1 < 0.0 or pq1 > pmax:
			return self.neginf

		#pmax = 1.0/delta_q
		pmax = 2.0
		p_dot_min = np.max([-(pmax-pq1)/self.delta_M,(0.0 - pq1)/self.delta_M])
		p_dot_max = np.min([-(0.0-pq1)/self.delta_M,(pmax - pq1)/self.delta_M])
		if pq4_dot < p_dot_min  or pq4_dot > p_dot_max:
			return self.neginf
		pmax = np.min([2.0,1.0/delta_q - pq4]) 
		p_dot_min = np.max([-(pmax-pq1)/self.delta_M,(0.0 - pq1)/self.delta_M])
		p_dot_max = np.min([-(0.0-pq1)/self.delta_M,(pmax - pq1)/self.delta_M])
		if pq3_dot < p_dot_min  or pq3_dot > p_dot_max:
			return self.neginf
		pmax = np.min([2.0,1.0/delta_q - pq4 - pq3])
		p_dot_min = np.max([-(pmax-pq1)/self.delta_M,(0.0 - pq1)/self.delta_M])
		p_dot_max = np.min([-(0.0-pq1)/self.delta_M,(pmax - pq1)/self.delta_M])
		if pq2_dot < p_dot_min  or pq2_dot > p_dot_max:
			return self.neginf
		pmax = np.min([2.0,1.0/delta_q - pq4 - pq3 -pq2])
		p_dot_min = np.max([-(pmax-pq1)/self.delta_M,(0.0 - pq1)/self.delta_M])
		p_dot_max = np.min([-(0.0-pq1)/self.delta_M,(pmax - pq1)/self.delta_M])
		if pq1_dot < p_dot_min  or pq1_dot > p_dot_max:
			return self.neginf

		return 0.0


	def prior_transform_general(self,u,i):

		# params are fb0, fb1, ft, oc, om, ocv, omv, ocov, f_o, logh, hdot

		x = self.default_params.copy()

		if not self.freeze[self.b_index]:
			# fB0
			x[self.b_index] = (0.93)*u[i] + 0.02
			i += 1
		if not self.freeze[self.b_index+1]:
			# fB1
			fb1min = np.max([-0.1/self.delta_M,-(0.95  - x[self.b_index])/self.delta_M,(0.02 - x[self.b_index])/self.delta_M])
			fb1max = np.min([0.1/self.delta_M,-(0.02 - x[self.b_index])/self.delta_M,(0.95  - x[self.b_index])/self.delta_M])
			x[self.b_index+1] = (fb1max-fb1min)*u[i] + fb1min
			#x[self.b_index+1] = truncnorm.ppf(u[i], fb1min/0.1, fb1max/0.1, loc=0.0, scale=0.1)
			i += 1
		if not self.freeze[self.b_index+2]:
			# ft
			x[self.b_index+2] = (1.0-x[self.b_index])*u[i] + 0.00
			i += 1

		if not self.freeze[self.b_index+3]:
			# oc
			x[self.b_index+3] = 0.6*u[i] + self.outlier_description[0] - 0.3
			i += 1
		if not self.freeze[self.b_index+4]:
			# om
			x[self.b_index+4] = 12*u[i] + self.outlier_description[1] - 6.0
			i += 1
		if not self.freeze[self.b_index+5]:
			# ocv
			x[self.b_index+5] = 9*u[i]*self.outlier_description[2]
			i += 1
		if not self.freeze[self.b_index+6]:
			# omv
			x[self.b_index+6] = 9*u[i]*self.outlier_description[3]
			i += 1
		if not self.freeze[self.b_index+7]:
			# ocov
			x[self.b_index+7] = 9*u[i]*self.outlier_description[4]
			i += 1
		if not self.freeze[self.b_index+8]:
			# f_O
			#x[self.b_index+3] = truncnorm.ppf(u[i], 0.0, 6.0, loc=0.0, scale=self.scale_fo)
			x[self.b_index+8] = u[i]*self.scale_fo
			i += 1

		if not self.freeze[self.b_index+9]:
			# log h
			#logh = truncnorm.ppf(u[i], -0.3/0.2, 1.0/0.1, loc=0.0, scale=0.1)
			#logh = 1.06*u[i] - 0.06 
			#x[self.b_index+9] = 10.0**logh
			x[self.b_index+9] = 1.06*u[i] - 0.06 
			i += 1
		if not self.freeze[self.b_index+10]:
			# hdot
			scale = 0.4*x[self.b_index+9]/self.delta_M
			#x[self.b_index+5] = truncnorm.ppf(u[i], 0.0, 2.0*x[self.b_index+4]/scale, loc=0.0, scale=scale)
			x[self.b_index+10] = u[i]*2.0*scale
			i += 1

		return i, x[self.b_index:self.i_index]


	def ln_prior_q_general(self,p):

		fb0, fb1, ft, oc, om, ocv, omv, ocov, fo, logh, hdot = p

		fb1min = np.max([-0.1/self.delta_M,-(0.95 - fb0)/self.delta_M,(0.02 - fb0)/self.delta_M])
		fb1max = np.min([0.1/self.delta_M,-(0.02 - fb0)/self.delta_M,(0.95  - fb0)/self.delta_M])

		if oc < self.outlier_description[0] - 0.3 or oc > self.outlier_description[0] + 0.3 or om < self.outlier_description[1] - 3 or \
				om > self.outlier_description[1] + 3 or ocv < 0 or ocv > 9*self.outlier_description[2] or omv < 0 or omv > 9*self.outlier_description[3] or \
				ocov < 0 or ocov > 9*self.outlier_description[4]:
			return self.neginf

		if fb0 < 0.02 or fb0 > 0.95   or fb1 < fb1min or fb1 > fb1max or ft < 0.0 or ft > 1.0-fb0 or \
				fo < 0 or fo > self.scale_fo or logh < -0.06 or logh > 1.0 or hdot < 0.0 or hdot > 2.0*0.4*10**logh/self.delta_M:
			return self.neginf 

		return 0.0


	def prior_transform_isochrone_correction(self,u,i):

		x = self.default_params.copy()

		if self.iso.correction_type == 'color':
			for j in range(len(x)-self.i_index):
				x[self.i_index+j] = (u[i] - 0.5) * self.isochrone_correction_scale + self.iso.initial_colour_correction_offsets[1][j]
				#x[self.i_index+j] = truncnorm.ppf(u[i], -5.0/self.isochrone_correction_scale, 5.0/self.isochrone_correction_scale, loc=self.iso.initial_colour_correction_offsets[1][j], scale=self.isochrone_correction_scale)
				i += 1
		else:
			for j in range(len(x)-self.i_index):
				x[self.i_index+j] = (u[i] - 0.5) * self.isochrone_correction_scale + self.iso.initial_magnitude_correction_offsets[1][j]
				#x[self.i_index+j] = truncnorm.ppf(u[i], -5.0/self.isochrone_correction_scale, 5.0/self.isochrone_correction_scale, loc=self.iso.initial_magnitude_correction_offsets[1][j], scale=self.isochrone_correction_scale)
				i += 1

		return i, x[self.i_index:]


	def ln_prior_isochrone_correction(self,p):

		x = self.default_params

		if self.iso.correction_type == 'color':
			for j in range(len(x)-self.i_index):
				if p[j] < - 0.5 * self.isochrone_correction_scale + self.iso.initial_colour_correction_offsets[1][j]  or p[j] > 0.5 * self.isochrone_correction_scale + self.iso.initial_colour_correction_offsets[1][j]:
					return self.neginf
		else:
			for j in range(len(x)-self.i_index):
				if p[j] < - 0.5 * self.isochrone_correction_scale + self.iso.initial_magnitude_correction_offsets[1][j]  or p[j] > 0.5 * self.isochrone_correction_scale + self.iso.initial_magnitude_correction_offsets[1][j]:
					return self.neginf

		return 0.0


	def prior_transform(self,u):

		assert self.q_model in ['quadratic','single_power','power','legendre','piecewise','hist']

		assert self.m_model in ['power']

		x = self.default_params.copy()

		ind = 0

		ind, x[:self.q_index] = self.prior_transform_mass_power(u)

		if self.q_model == 'quadratic':

			ind, x[self.q_index:self.b_index] = self.prior_transform_q_quadratic(u,ind)

		if self.q_model == 'single_power':

			ind, x[self.q_index:self.b_index] = self.prior_transform_q_single_power(u,ind)

		if self.q_model == 'power':

			ind, x[self.q_index:self.b_index] = self.prior_transform_q_power(u,ind)

		if self.q_model == 'legendre':

			ind, x[self.q_index:self.b_index] = self.prior_transform_q_legendre(u,ind)

		if self.q_model == 'piecewise':

			ind, x[self.q_index:self.b_index] = self.prior_transform_q_piecewise(u,ind)

		if self.q_model == 'hist':

			ind, x[self.q_index:self.b_index] = self.prior_transform_q_hist(u,ind)

		ind, x[self.b_index:self.i_index] = self.prior_transform_general(u,ind)

		ind, x[self.i_index:] = self.prior_transform_isochrone_correction(u,ind)

		y = x[self.freeze==0]

		return y


	def lnprob(self,params):

		lp = self.ln_prior(params)

		if not np.isfinite(lp):
			return self.neginf

		lnp, _  = self.lnlikelihood(params)

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
		print('self.emcee_walker_dispersion.shape',self.emcee_walker_dispersion.shape)
		print('np.random.randn(ndim).shape',np.random.randn(ndim).shape)
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


	def dynesty_sample(self,prefix='dy_',jitter=False,bound='multi',sample='rwalk',nlive=500,n_parallel=1):

		from dynesty import NestedSampler, DynamicNestedSampler
		from dynesty import plotting as dyplot
		from dynesty import utils as dyfunc
		from dynesty.pool import Pool

		# import dill
		# dyfunc.pickle_module = dill
		
		assert n_parallel == 1, "Parallel processing not implemented yet."


		self.prefix = prefix

		ndim = int(self.ndim - np.sum(self.freeze))
		print('ndim:',self.ndim,np.sum(self.freeze),ndim)

		labels = [self.labels[i] for i in range(self.ndim) if self.freeze[i] == 0]


		if n_parallel == 1:

			#sampler = NestedSampler(self.lnlikelihood, self.prior_transform, ndim, bound=bound,sample=sample,nlive=nlive)
			sampler = DynamicNestedSampler(self.lnlikelihood, self.prior_transform, ndim, bound=bound, sample=sample, blob=True)
			sampler.run_nested(nlive_init=nlive,dlogz_init=0.01)
			res = sampler.results
			print(res)
			#sampler.run_nested(use_stop=False,wt_kwargs={'pfrac': 1.0})

		else:

			import multiprocessing
			multiprocessing.set_start_method('spawn')
			with Pool(n_parallel, self.lnlikelihood, self.prior_transform) as pool:
				print(pool.__dir__())
				sampler = DynamicNestedSampler(pool.loglike, pool.prior_transform, ndim, pool=pool, bound=bound, sample=sample, blob=True)
				sampler.run_nested(nlive_init=nlive)


		res = sampler.results

		#samples, weights = res.samples, np.exp(res.logwt - res.logz[-1])
		samples, weights = res.samples, res.importance_weights()

		np.save(prefix+'samples.npy',samples)
		np.save(prefix+'weights.npy',weights)
		np.save(prefix+'samples_lnl.npy',res['blob'])

		res.summary()

		try:
			fig, axes = dyplot.runplot(res)
			plt.savefig(prefix+'summary.png')
		except:
			print('dyplot.runplot failed')
			pass

		try:
			fig, axes = dyplot.traceplot(res, show_titles=True,trace_cmap='viridis',
		                         connect=True,connect_highlight=range(5),labels=labels)
			plt.savefig(prefix+'trace.png')
		except:
			print('dyplot.traceplot failed')
			pass

		try:
			fig, axes = plt.subplots(ndim, ndim, figsize=(15, 15))
			axes = axes.reshape((ndim, ndim))  # reshape axes
			fg, ax = dyplot.cornerplot(res, color='blue',show_titles=True,max_n_ticks=3,labels=labels,
			                        quantiles=None,fig=(fig,axes))
			plt.savefig(prefix+'corner.png')
		except:
			print('dyplot.cornerplot failed')
			pass

		try:
			fig, axes = plt.subplots(ndim, ndim, figsize=(15, 15))
			axes = axes.reshape((ndim, ndim))  # reshape axes
			fg, ax = dyplot.cornerpoints(res, cmap='viridis',kde=False,max_n_ticks=3,labels=labels,fig=(fig,axes))
			plt.savefig(prefix+'cornerpts.png')
		except:
			print('dyplot.cornerpoints failed')
			pass

		if jitter:

			lnzs = np.zeros((100, len(res.logvol)))
			for i in range(100):
				res_j = dyfunc.jitter_run(res)
				lnzs[i] = res_j.logz[-1]
			lnz_mean, lnz_std = np.mean(lnzs), np.std(lnzs)
			print('Jitter logz:        {:6.3f} +/- {:6.3f}'.format(lnz_mean, lnz_std))
			with open(f'{prefix}_lnZ.txt','w') as file:
				file.write(f'{lnz_mean:8.3f}    {lnz_std:8.3f}\n')



	def ultranest_sample(self,prefix='un_',stepsampler=True):

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


		result = sampler.run(min_num_live_points=400, min_ess=10000)

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


