import sys
import os
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import nnls
from scipy.stats import norm, truncnorm


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

from Data import Data
from Isochrone import Isochrone



class CMDFitter():

	"""Main class for the CMDFitter code."""

	def __init__(self,json_file=None,data=None,isochrone=None,iso_correction_type='colour',trim_data=True,q_model='legendre',m_model='power',outlier_scale=2.0,q_min=0.2,error_scale_type='both'):


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

		"""

		self.version = 7.0

		self.citation = "Albrow, M.D., Ulusele, I.H., 2022, MNRAS, 515, 730"


		assert error_scale_type in ['colour','both']
		self.error_scale_type = error_scale_type

		assert q_model in ['power','legendre','piecewise','hist']
		assert m_model in ['power']

		self.q_model = q_model
		self.m_model = m_model

		self.set_up_model_parameters()

		self.freeze = np.zeros(self.ndim)
		self.prefix = 'out_'

		self.set_up_lengdre_functions()

		assert q_min >= 0.0 and q_min < 1.0
		self.q_min = q_min

		# Miscellaneous attributes
		self.neginf = -np.inf
		self.scale_fo = 0.01
		self.emcee_walker_dispersion = 1.e-7


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

		if trim_data:
			self.data.trim(self.iso)

		self.data.upload_to_GPU(drv)

		self.set_up_data_attributes()

		self.define_data_outlier_model(outlier_scale)

		self.set_up_basis_functions()

		self.upload_basis_functions_to_GPU(drv)

		self.likelihood = likelihood_functions.get_function("likelihood")

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

		self.magnitude_ref = 0.5*(self.data.magnitude_min+self.data.magnitude_max)
		self.delta_mag = self.data.magnitude_max - self.magnitude_ref

		self.h_magnitude_ref = self.data.magnitude_min


	def define_data_outlier_model(self,outlier_scale):

		d = np.column_stack((self.data.colour,self.data.magnitude))
		d_med = np.median(d,axis=0)
		d_cov = outlier_scale**2*np.cov(d,rowvar=False)
		self.outlier_description = np.ascontiguousarray([d_med[0],d_med[1],d_cov[0,0],d_cov[1,1],d_cov[0,1]],dtype=np.float64)


	def upload_basis_functions_to_GPU(self,drv):

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

		self.DMQ_CUDA = likelihood_functions.get_texref("DMQ")
		drv.matrix_to_texref(np.float32(D_ij),self.DMQ_CUDA,order='F')
		self.DMQ_CUDA.set_filter_mode(drv.filter_mode.POINT)

		self.SMQ_CUDA = likelihood_functions.get_texref("SMQ")
		drv.matrix_to_texref(np.float32(S_ij_shaped),self.SMQ_CUDA,order='F')
		self.SMQ_CUDA.set_filter_mode(drv.filter_mode.POINT)


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

		self.MA = np.zeros((self.n_bf,self.n_bf))
		for k in range(self.n_bf):
			for j in range(self.n_bf):
				self.MA[k,j] = np.sum(self.Mbf[k,:]*self.Mbf[j,:]/self.Msigma**2) 

		self.MAT = self.MA.T
		self.MAA = np.dot(self.MAT,self.MA)


	def set_up_model_parameters(self):

		if self.q_model == 'power':
			self.ndim = 15
			self.labels =                  [r"$\log_{10} k$", r"$M_0$", r"$\gamma$",  r"$c_0$",  r"$\dot{c}_0$",  r"$\alpha_1$",  r"$\alpha_2$", r"$q_0$",  r"$a_1$", r"$a_2$", r"$f_B$", r"$\dot{f_B}$", r"$f_O$", r"$h_0$", r"$h_1$"]
			self.default_params = np.array([4.0,              0.0,     0.0006,       0.0,       0.0,             2.0,           2.0,            0.5,       1.0,      1.0,      0.35,     0.0,        0.01,     1.0,      0.00])
			self.q_index = 5
			self.b_index = 10

		if self.q_model == 'legendre':
			self.ndim = 16
			self.labels =                  [r"$\log_{10} k$", r"$M_0$", r"$\gamma$",  r"$c_0$",  r"$\dot{c}_0$",  r"$a_1$", r"$a_2$", r"$a_3$", r"$\dot{a}_1$", r"$\dot{a}_2$", r"$\dot{a}_3$", r"$f_B$", r"\dot{f_B}$", r"$f_O$", r"$h_0$", r"$h_1$"]
			self.default_params = np.array([4.0,              0.0,     0.0006,       0.0,       0.0,             0.0,      0.0,      0.0,      0.0,            0.0,            0.0,            0.35,       0.0,        0.01,     1.0,      0.00])
			self.q_index = 5
			self.b_index = 11

		if self.q_model == 'piecewise':
			self.ndim = 14
			self.labels =                  [r"$\log_{10} k$", r"$M_0$", r"$\gamma$", r"$c_0$",  r"$\dot{c}_0$", r"$pq_4$",  r"$pq_3$",  r"$pq_2$", r"$pq_1$",  r"$f_B$", r"$\dot{f_B}$", r"$f_O$", r"$h_0$", r"$h_1$"]
			self.default_params = np.array([4.0,              0.0,     0.0006,       0.0,       0.0,             1.0,        1.0,       1.0,      1.0,         0.35,        0.0,        0.01,     1.0,      0.00])
			self.q_index = 5
			self.b_index = 9

		if self.q_model == 'hist':
			self.ndim = 14
			self.labels =                  [r"$\log_{10} k$", r"$M_0$", r"$\gamma$", r"$c_0$",  r"$\dot{c}_0$", r"$pq_4$",  r"$pq_3$",  r"$pq_2$", r"$pq_1$",  r"$f_B$", r"$\dot{f_B}$", r"$f_O$", r"$h_0$", r"$h_1$"]
			self.default_params = np.array([4.0,              0.0,     0.0006,       0.0,       0.0,             1.0,        1.0,       1.0,      1.0,         0.35,        0.0,        0.01,     1.0,      0.00])
			self.q_index = 5
			self.b_index = 9

		if self.m_model == 'legendre':
			self.labels[4] =                              [r"$b_1$", r"$b_2$", r"$b_3$", r"$b_4$"] + self.labels[self.q_index:]
			self.default_params[:4] = np.hstack((np.array([0.0,      0.0,      0.0,      0.0]),self.default_params[self.q_index:]))
			self.ndim -= 1
			self.q_index -= 1
			self.b_index -= 1


	def set_up_lengdre_functions(self):

		# Shifted Legendre polynomials
		self.sl_0 = lambda x: x*0.0 + 1.0
		self.sl_1 = lambda x: 2.0*x - 1.0
		self.sl_2 = lambda x: 6.0*x**2 - 6.0*x + 1.0
		self.sl_3 = lambda x: 20.0*x**3 - 30.0*x**2 + 12.0*x - 1.0
		self.sl_4 = lambda x: 70.0*x**4 - 140.0*x**3 + 90.0*x**2 - 20.0*x + 1.0

		# Derivatives of shifted Legendre polynomials
		self.der_sl_0 = lambda x: 0.0
		self.der_sl_1 = lambda x: 2.0
		self.der_sl_2 = lambda x: 12.0*x - 6.0
		self.der_sl_3 = lambda x: 60.0*x**2 - 60.0*x + 12.0
		self.der_sl_4 = lambda x: 280.0*x**3 - 420.0*x**2 + 180.0*x - 20.0

		# Integrals of shifted Legendre polynomials
		self.int_sl_0 = lambda x: x
		self.int_sl_1 = lambda x: x**2 - x
		self.int_sl_2 = lambda x: 2.0*x**3 - 3.0*x**2 + x
		self.int_sl_3 = lambda x: 5.0*x**4 - 10.0*x**3 + 6.0*x**2 - x
		self.int_sl_4 = lambda x: 14.0*x**5 - 35.0*x**4 + 30.0*x**3 - 10.0*x**2 + x


	def M_distribution(self,x,params):

		"""Evaluate mass function at x."""

		assert self.m_model in ['power']

		if self.m_model == 'power':
			log_k,x0,gamma,c0, c1 = params
			k = 10.0**log_k
			m = np.linspace(self.mass_slice[0],self.mass_slice[1],1000)
			c_scale = np.max([self.mass_slice[0],x0])**(-gamma)
			y = ((c0 + c1*(m-self.mass_slice[0]))*c_scale + m**(-gamma) )/ (1.0 + np.exp(-k*(m-x0)) )
			normalM = 1.0 / (np.sum(y)*(m[1]-m[0]))

			y =  normalM * ((c0 + c1*(x-self.mass_slice[0]))*c_scale + x**(-gamma)) / (1.0 + np.exp(-k*(x-x0)))
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


	def q_distribution_power(self,x,params):

		alpha1, alpha2, q0, a1, a2, M = params
		xarr = np.atleast_1d(x)
		qdist = np.zeros_like(xarr)

		c = (1.0 - a1*(q0-self.q_min)**(alpha1+1.0)/(alpha1+1.0) - a2*(1.0-q0)**(alpha2+1.0)/(alpha2+1.0)) / (1.0-self.q_min)
		
		qdist[xarr<q0] = c + a1*(q0 - xarr[xarr<q0])**alpha1
		qdist[xarr>=q0] = c + a2*(xarr[xarr>=q0] - q0)**alpha2

		return qdist


	def q_distribution_legendre(self,x,params):

		a1, a2, a3, a1_dot, a2_dot,a3_dot, M = params
		dM = M-self.M_ref

		x_arr = np.atleast_1d(x)
		
		# Map q to 0 - 1 domain for legendre functions
		x1 = (x_arr-self.q_min)/(1.0-self.q_min) 

		return self.sl_0(x1) + (a1+a1_dot*dM)*self.sl_1(x1) + (a2+a2_dot*dM)*self.sl_2(x1) + (a3+a3_dot*dM)*self.sl_3(x1)


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

		pnode = np.zeros(5)
		pnode[4], pnode[3], pnode[2], pnode[1], M = params

		delta_q = (1.0-self.q_min)/5.0
		qnode = self.q_min + np.arange(5)*delta_q

		pnode[0] = 5.0 - np.sum(pnode[1:])

		x_arr = np.atleast_1d(x)
		qdist = np.zeros_like(x_arr)

		for i in range(5):
			ind = np.where(x_arr >= qnode[i])[0]
			if ind.any():
				qdist[ind] = pnode[i]

		return qdist


	def q_distribution(self,x,params):

		"""
		Evaluate the binary mass-ratio distribution function at x.

		"""

		assert self.q_model in ['power','legendre','piecewise','hist']

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


	def integrate_lendre(self,params,q1,q2):

		# Map q to 0 - 1 domain for legendre functions
		q1 = (q1-self.q_min)/(1.0-self.q_min) 
		q2 = (q2-self.q_min)/(1.0-self.q_min) 

		y = self.int_sl_0(q2) - self.int_sl_0(q1)
		y += params[0]*(self.int_sl_1(q2) - self.int_sl_1(q1))
		y += params[1]*(self.int_sl_2(q2) - self.int_sl_2(q1))
		y += params[2]*(self.int_sl_3(q2) - self.int_sl_3(q1))

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

		pnode = np.zeros(5)
		pnode[4], pnode[3], pnode[2], pnode[1] = params

		delta_q = self.q_min/5.0
		qnode = self.q_min + np.arange(5)*delta_q

		pnode[0] = 5.0 - np.sum(pnode[1:])

		q_int = np.zeros(2)

		for i, q in enumerate([q1,q2]):

			ind = np.where(qnode<=q)[0][-1]

			if ind > 0:
				q_int[i] = np.sum(pnode[:ind])*delta_q

			q_int[i] += pnode[ind]*(q-qnode[ind])

		y = q_int[1] - q_int[0]

		return y


	def q_distribution_integral(self,params,q1,q2):

		assert self.q_model in ['power','legendre','piecewise','hist']

		assert np.min([q1,q2]) >= self.q_min
		assert np.max([q1,q2]) <= 1.0
		assert q2 >= q1

		if q2 == q1:
			return 0.0

		if self.q_model == 'legendre':

			return self.integrate_lendre(params,q1,q2)

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


	def model_realisation(self,p,n,add_observational_scatter=True,outliers=True):

		"""
		Compute a random (magnitude, colour) realisation of the model paramterised by p for n stars.

		Also return 		star_type = 0, 1, 2 for single stars, binaries, outliers.
		""" 

		fb0, fb1, fo, h0, h1 = p[self.b_index:]

		star_type = np.zeros(n)
		mag = np.zeros(n)
		colour = np.zeros(n)

		if outliers:

			n_outliers = int(round(fo*n))
			cov = np.zeros((2,2))
			cov[0,0] = self.outlier_description[2];
			cov[1,1] = self.outlier_description[3];
			cov[0,1] = self.outlier_description[4];
			cov[1,0] = self.outlier_description[4];

			d = np.random.multivariate_normal(self.outlier_description[:2], cov, n_outliers)

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

		q = np.zeros_like(M1)

		for i in range(n_outliers,n):

			if r[i] <= fb[i]/(1.0-fo):

				star_type[i] = 1

				args = p[self.q_index:self.b_index].tolist() + [M1[i]]

				q_sampler = self.q_distribution_sampler(args)

				q[i] = q_sampler(np.random.rand())

		mag[n_outliers:], colour[n_outliers:] = self.iso.binary(M1[n_outliers:],q[n_outliers:])

		m0, c0 = self.iso.binary(M1,np.zeros_like(M1))


		if add_observational_scatter:

			h = h0 + h1*(mag-self.h_magnitude_ref)

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

		return mag, colour, star_type


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



	def M_gauss(self,params):
		
		"""Return the (positive) coefficients for mapping the mass basis functions onto the mass function."""

		My = self.M_distribution(self.Mx,params)
		
		Mb = np.zeros(self.n_bf)
		for k in range(self.n_bf):
			Mb[k] = np.sum(My*self.Mbf[k,:]/self.Msigma**2)
		
		Ma, resid = nnls(self.MA,Mb,maxiter=5000)

		mfit  = np.zeros_like(self.Mx)
		for k in range(self.n_bf):
			mfit += Ma[k]*self.Mbf[k,:]

		norm_c = np.sum(mfit*(self.Mx[1]-self.Mx[0]))

		return Ma/norm_c


	def q_gauss(self,params):
				
		"""Return the (positive) coefficients for mapping the q basis functions onto the q distribution function."""

		qy = self.q_distribution(self.qx,params)

		qb = np.zeros(self.n_bf)
		for k in range(self.n_bf):
			qb[k] = np.sum(qy*self.qbf[k,:]/self.qsigma**2)
		
		qa, resid = nnls(self.qA,qb,maxiter=5000)

		qfit  = np.zeros_like(self.qx)
		for k in range(self.n_bf):
			qfit += qa[k]*self.qbf[k,:]

		norm_c = np.sum(qfit*(self.qx[1]-self.qx[0]))

		return qa/norm_c


	def precalc(self,params):
		
		"""Return the vector of mass basis function coefficients (for single stars) and 
		the grid of (M,q) basis function coefficients (for binary stars). These are multiplied
		by the single-star and binary-star fractions respectively."""

		assert self.q_model in ['power','legendre','piecewise','hist']

		fb0, fb1, fo, h0, h1 = params[self.b_index:]

		# fb is a vector here
		fb = fb0 + fb1*(self.mass_slice[1] - self.M0)

		PMQ = np.zeros(self.n_bf**2)
		Ma = self.M_gauss(params[:self.q_index])

		if (self.q_model == 'legendre') and (np.sum(self.freeze[self.q_index+3:self.b_index]) < 3):

			# p(q) is a function of mass

			for i in range (self.n_bf):
				args = params[self.q_index:self.b_index].tolist() + [self.M0[i]]
				qa = self.q_gauss(args)
				for j in range(self.n_bf):
					PMQ[i+j*self.n_bf] = Ma[i]*qa[j]*fb[i]

		else:

			args = params[self.q_index:self.b_index].tolist() + [self.M0[0]]
			qa = self.q_gauss(args)
			for i in range (self.n_bf):
				for j in range(self.n_bf):
					PMQ[i+j*self.n_bf] = Ma[i]*qa[j]*fb[i]
	
		return Ma*(1.0-fb-fo), PMQ


	def lnlikelihood(self,params):

		"""Call the external CUDA function to evaluate the likelihood function."""
		
		p = self.default_params.copy()
		p[self.freeze==0] = params

		fo = p[-3]
		h0 = p[-2]
		h1 = p[-1]

		assert self.q_model in ['power','legendre','piecewise','hist']

		# Check that the parameters generate positive q distributions for all masses, and a positive M distribution.

		m_dist_test = self.M_distribution(np.linspace(self.mass_slice[0],self.mass_slice[1],101),p[:self.q_index])
		if np.min(m_dist_test) < 0.0:
			return self.neginf

		if self.q_model == 'legendre':

			qx = np.linspace(0.0,1.0,101)
			for MM in np.linspace(self.mass_slice[0],self.mass_slice[1],31).tolist():
				args = p[self.q_index:self.b_index].tolist() + [MM]
				q_dist_test = self.q_distribution(qx,args)
				if np.min(q_dist_test) < 0.0:
					with open(self.prefix+'.err', 'a') as f:
						#f.write('Negative q dist for:')
						f.write(np.array2string(params,max_line_width=1000).strip('[]\n')+'\n')
					return self.neginf

		try:
			P_i, PMQ = self.precalc(p)
		except:
			print("Error in precalc for p =",p)
			sys.exit()


		c_P_i = np.ascontiguousarray(P_i.astype(np.float64))
		c_PMQ = np.ascontiguousarray(PMQ.astype(np.float64))

		n_pts = len(self.data.magnitude)
		
		blockshape = (int(256),1, 1)
		gridshape = (n_pts, 1)

		lnP_k = np.zeros(n_pts*4).astype(np.float64)

		htype = 0
		if self.error_scale_type == 'both':
			htype = 1

		likelihood(drv.In(c_P_i), drv.In(c_PMQ), drv.In(self.outlier_description),np.int32(htype),np.float64(h0), np.float64(h1), np.float64(self.h_magnitude_ref),np.float64(fo), drv.InOut(lnP_k), block=blockshape, grid=gridshape)

		lnP = np.sum(lnP_k[:n_pts])

		self.lnP_k = lnP_k.reshape(n_pts,4,order='F')


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

		assert self.q_model in ['power','legendre','piecewise','hist']
		assert self.m_model in ['power']

		ln_p = self.ln_prior_mass_power(p[:q_index])

		if self.q_model == 'power':

			lnp += self.ln_prior_q_power(p[self.q_index:self.b_index])

		if self.q_model == 'legendre':

			lnp += self.ln_prior_q_legendre(p[self.q_index:self.b_index])

		if self.q_model == 'piecewise':

			lnp += self.ln_prior_q_piecewise(p[self.q_index:self.b_index])

		if self.q_model == 'hist':

			lnp += self.ln_prior_q_hist(p[self.q_index:self.b_index])

		lnp += self.ln_prior_q_general(p[self.b_index:])

		return lnp



	def prior_transform_mass_power(self,u):

		# params are log k, M0, gamma, c0, c1

		x = self.default_params.copy()[:5]

		i = 0

		if not self.freeze[0]:
			# log k
			x[0] = norm.ppf(u[i], loc=1.7, scale=0.2)
			i += 1
		if not self.freeze[1]:
			# M0
			x[1] = norm.ppf(u[i], loc=self.mass_slice[0]+0.1,scale=0.5)
			i += 1
		if not self.freeze[2]:
			# gamma
			x[2] = truncnorm.ppf(u[i], -2.35, 6.0-2.35, loc=2.35, scale=1.0)
			i += 1
		if not self.freeze[3]:
			# c0
			x[3] = truncnorm.ppf(u[i],0.0,1.0/0.5,loc=0.0,scale=0.5)
			i += 1
		if not self.freeze[4]:
			# c1
			x[4] = truncnorm.ppf(u[i],-1.0/0.2,1.0/0.2,loc=0.0,scale=0.2)
			i += 1

		return i, x


	def ln_prior_mass_power(self,p):

		log_k, M0, gamma, c0, c1 = p

		prior = norm.pdf(log_k,loc=1.7,scale=0.2) * \
				norm.pdf(M0,loc=self.mass_slice[0]+0.1, scale=0.5) * \
				truncnorm.pdf(gamma, -2.35, 6.0-2.35, loc=2.35, scale=1.0) * \
				truncnorm.pdf(c0,0.0,1.0/0.5,loc=0.0,scale=0.5) * \
				truncnorm.pdf(c1,-1.0/0.2,1.0/0.2,loc=0.0,scale=0.2)

		return np.log(prior)



	def prior_transform_q_power(self,u,i):

		# params are alpha1, alpha2, q0, a1, a2

		x = self.default_params.copy()

		if not self.freeze[self.q_index]:
			# alpha1
			x[self.q_index] = truncnorm.ppf(u[i], -0.9/10.0, 3.0/10.0, loc=2.0, scale=10.0)
			i += 1
		if not self.freeze[self.q_index+1]:
			# alpha2
			x[self.q_index+1] = truncnorm.ppf(u[i], -0.9/10.0, 3.0/10.0, loc=2.0, scale=10.0)
			i += 1
		if not self.freeze[self.q_index+2]:
			# q0
			x[self.q_index+2] = truncnorm.ppf(u[i], np.max([self.q_min,-0.49/(1.0-self.q_min)])/10.0, np.min([1.0,0.49/(1.0-self.q_min)])/10.0, loc=(0.5-self.q_min)/(1.0-self.q_min), scale=10.0)
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
			x[self.q_index+3] = truncnorm.ppf(u[i], a1min/50.0, a1max/50.0, loc=0.0, scale=50.0)
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
			x[self.q_index+4] = truncnorm.ppf(u[i], a2min/50.0, a2max/50.0, loc=0.0, scale=50.0)
			i += 1

		return i, x[self.q_index:self.b_index]


	def ln_prior_q_power(self,p):

		alpha1, alpha2, q0, a1, a2, = p

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

		if alpha1  < 1.0 or alpha2 < 1.0:
			return self.neginf 

		prior = truncnorm.pdf(alpha1, -0.9/10.0, 3.0/10.0, loc=2.0, scale=10.0) * \
				truncnorm.pdf(alpha2, -0.9/10.0, 3.0/10.0, loc=2.0, scale=10.0) * \
				truncnorm.pdf(q0, np.max([self.q_min,-0.49/(1.0-self.q_min)])/10.0, np.min([1.0,0.49/(1.0-self.q_min)])/10.0, loc=(0.5-self.q_min)/(1.0-self.q_min), scale=10.0) * \
				truncnorm.pdf(a1, a1min/50.0, a1max/50.0, loc=0.0, scale=50.0) * \
				truncnorm.pdf(a2, a2min/50.0, a2max/50.0, loc=0.0, scale=50.0)

		return np.log(prior)


	def prior_transform_q_legendre(self,u,i):

		# params are a1, a2, a3, a1_dot, a2_dot, a3_dot, 

		x = self.default_params.copy()

		if not self.freeze[self.q_index]:

			# a1
			x[self.q_index] = 2.0*u[i] - 1.0
			i += 1

		if not self.freeze[self.q_index+1]:
			# a2
			#x[self.q_index+1] = 2.0*u[i]
			x[self.q_index+1] = 3.0*u[i] - 1.0
			i += 1

		if not self.freeze[self.q_index+2]:
			# a3
			t1 = (5.0-np.sqrt(5.0))/10.0
			t2 = (5.0+np.sqrt(5.0))/10.0
			tmin = 0.5*(-0.8 - x[self.q_index]*self.sl_1(t1) - x[self.q_index+1]*self.sl_2(t1)) / self.sl_3(t1)
			tmax = 0.5*(-0.8 - x[self.q_index]*self.sl_1(t2) - x[self.q_index+1]*self.sl_2(t2)) / self.sl_3(t2)
			a3min = np.max([-1.0-x[self.q_index]-x[self.q_index+1],tmin,-1.0])
			a3max = np.min([1.0 - x[self.q_index] + x[self.q_index+1],tmax,-1.0])
			x[self.q_index+2] = a3min + (a3max - a3min)*u[i]
			i += 1

		if not self.freeze[self.q_index+3]:
			# a1_dot
			x[self.q_index+3] = norm.ppf(u[i], loc=0.0, scale=0.1/self.delta_M)
			i += 1
		if not self.freeze[self.q_index+4]:
			# a2_dot
			x[self.q_index+4] = norm.ppf(u[i], loc=0.0, scale=0.1/self.delta_M)
			i += 1
		if not self.freeze[self.q_index+5]:
			# a3_dot
			x[self.q_index+5] = norm.ppf(u[i], loc=0.0, scale=0.1/self.delta_M)
			i += 1

		return i, x[self.q_index:self.b_index]


	def ln_prior_q_legendre(self,p):

		a1, a2, a3,  a1_dot, a2_dot, a3_dot = p

		t1 = (5.0-np.sqrt(5.0))/10.0
		t2 = (5.0+np.sqrt(5.0))/10.0
		tmin = 0.5*(-0.8 - a1*self.sl_1(t1) - a2*self.sl_2(t1)) / self.sl_3(t1)
		tmax = 0.5*(-0.8 - a1*self.sl_1(t2) - a2*self.sl_2(t2)) / self.sl_3(t2)
		a3min = np.max([-1.0-a1-a2,tmin,-1.0])
		a3max = np.min([1.0 - a1 + a2,tmax,1.0])

		if a2 < -1.0 or a2 > 2.0 or a3 < a3min or a3 > a3max:
			return self.neginf 

		prior = norm.pdf(a1,loc=0.0,scale=2.0) * \
				norm.pdf(a1_dot,loc=0.0,scale=0.1/self.delta_M) * \
				norm.pdf(a2_dot,loc=0.0,scale=0.1/self.delta_M) * \
				norm.pdf(a3_dot,loc=0.0,scale=0.1/self.delta_M)

		return np.log(prior)


	def prior_transform_q_piecewise(self,u,i):

		# params are pq4, pq3, pq2, pq1 

		x = self.default_params.copy()

		if not self.freeze[self.q_index]:
			# pq4
			sc = 0.5
			x[self.q_index] = truncnorm.ppf(u[i], -1.0/sc, 3.99/sc, loc=1.0, scale=sc)
			i += 1
		if not self.freeze[self.q_index+1]:
			# pq3
			delta_q = 1.0/4.0
			pmax = 1.0/delta_q - x[self.q_index]/2.0 
			x[self.q_index+1] = truncnorm.ppf(u[i], -1.0/sc, (pmax-1.0)/sc, loc=1.0, scale=sc)
			q_int = delta_q * (x[self.q_index] + x[self.q_index+1])/2.0
			i += 1
		if not self.freeze[self.q_index+2]:
			# pq2
			pmax = (1.0-q_int)/delta_q - x[self.q_index+1]/2.0
			x[self.q_index+2] = truncnorm.ppf(u[i], -1.0/sc, (pmax-1.0)/sc, loc=1.0, scale=sc)
			q_int += delta_q * (x[self.q_index+1] + x[self.q_index+2])/2.0
			i += 1
		if not self.freeze[self.q_index+3]:
			# pq1
			pmax = (1.0-q_int)/delta_q - x[self.q_index+2]/2.0
			x[self.q_index+3] = truncnorm.ppf(u[i], -1.0/sc, (pmax-1.0)/sc, loc=1.0, scale=sc)
			q_int += delta_q * (x[self.q_index+2] + x[self.q_index+3])/2.0
			i += 1

		return i, x[self.q_index:self.b_index]


	def ln_prior_q_piecewise(self,p):

		pq4, pq3, pq2, pq1 = p

		sc = 0.5
		prior = truncnorm.pdf(pq4,-1.0/sc, 3.99/sc, loc=1.0, scale=sc)

		delta_q = 1.0/4.0
		pmax = 1.0/delta_q - pq4/2.0
		q_int = delta_q * (pq4 + pq3)/2.0
		prior *= truncnorm.pdf(pq3,-1.0/sc, (pmax-1.0)/sc, loc=1.0, scale=sc)

		pmax = (1.0-q_int)/delta_q - pq3/2.0
		q_int += delta_q * (pq3 + pq2)/2.0
		prior *= truncnorm.pdf(pq2,-1.0/sc, (pmax-1.0)/sc, loc=1.0, scale=sc)

		pmax = (1.0-q_int)/delta_q - pq2/2.0
		prior *= truncnorm.pdf(pq1,-1.0/sc, (pmax-1.0)/sc, loc=1.0, scale=sc)

		return np.log(prior)



	def prior_transform_q_hist(self,u,i):

		# params are pq4, pq3, pq2, pq1 

		x = self.default_params.copy()

		if not self.freeze[self.q_index]:
			# pq4
			sc = 0.5
			x[self.q_index] = truncnorm.ppf(u[i], -1.0/sc, 3.99/sc, loc=1.0, scale=sc)
			i += 1
		if not self.freeze[self.q_index+1]:
			# pq3
			delta_q = 1.0/5.0
			pmax = 1.0/delta_q - x[self.q_index] 
			x[self.q_index+1] = truncnorm.ppf(u[i], -1.0/sc, (pmax-1.0)/sc, loc=1.0, scale=sc)
			i += 1
		if not self.freeze[self.q_index+2]:
			# pq2
			pmax = 1.0/delta_q - x[self.q_index] - x[self.q_index+1]
			x[self.q_index+2] = truncnorm.ppf(u[i], -1.0/sc, (pmax-1.0)/sc, loc=1.0, scale=sc)
			i += 1
		if not self.freeze[self.q_index+3]:
			# pq1
			pmax = 1.0/delta_q - x[self.q_index] - x[self.q_index+1] - x[self.q_index+2]
			x[self.q_index+3] = truncnorm.ppf(u[i], -1.0/sc, (pmax-1.0)/sc, loc=1.0, scale=sc)
			i += 1

		return i, x[self.q_index:self.b_index]


	def ln_prior_q_hist(self,p):

		pq4, pq3, pq2, pq1 = p

		sc = 0.5
		delta_q = 1.0/5.0
		prior = truncnorm.pdf(fo, 0.0, 3.99, loc=0.0, scale=self.scale_fo) * truncnorm.pdf(pq4,-1.0/sc, 6/sc, loc=1.0, scale=sc)

		pmax = 1.0/delta_q - pq4
		prior *= truncnorm.pdf(pq3,-1.0/sc, (pmax-1.0)/sc, loc=1.0, scale=sc)

		pmax = 1.0/delta_q - pq4 - pq3
		prior *= truncnorm.pdf(pq2,-1.0/sc, (pmax-1.0)/sc, loc=1.0, scale=sc)

		pmax = 1.0/delta_q - pq4 - pq3 - pq1
		prior *= truncnorm.pdf(pq3,-1.0/sc, (pmax-1.0)/sc, loc=1.0, scale=sc)

		return np.log(prior)


	def prior_transform_general(self,u,i):

		# params are fb0, fb1, f_o, log h0, h1

		x = self.default_params.copy()

		if not self.freeze[self.b_index]:
			# fB0
			x[self.b_index] = 0.93*u[i] + 0.02
			i += 1
		if not self.freeze[self.b_index+1]:
			# fB1
			fb1max = np.min([0.1/self.mass_range,(0.95-x[self.b_index])/self.mass_range])
			fb1min = np.max([-0.1/self.mass_range,-(x[self.b_index]-0.02)/self.mass_range])
			x[self.b_index+1] = truncnorm.ppf(u[i], fb1min/0.1, fb1max/0.1, loc=0.0, scale=0.1)
			i += 1
		if not self.freeze[self.b_index+2]:
			# f_O
			x[self.b_index+2] = truncnorm.ppf(u[i], 0.0, 6.0, loc=0.0, scale=self.scale_fo)
			i += 1

		if not self.freeze[self.b_index+3]:
			# log h0
			logh = truncnorm.ppf(u[i], -0.3/0.2, 1.0/0.1, loc=0.0, scale=0.1)
			x[self.b_index+3] = 10.0**logh
			i += 1
		if not self.freeze[self.b_index+4]:
			# h1
			scale = 0.4*x[self.b_index+3]/self.delta_M
			x[self.b_index+4] = truncnorm.ppf(u[i], 0.0, 2.0*x[self.b_index+3]/scale, loc=0.0, scale=scale)
			i += 1

		return x[self.b_index:]


	def ln_prior_q_general(self,p):

		fb0, fb1, fo, h0, h1 = p

		fb_end = fb0 + fb1*self.mass_range

		if np.min([fb0,fb_end]) < 0.02 or np.max([fb0,fb_end]) > 0.95:
			return self.neginf 

		fb1max = np.min([0.1/self.mass_range,(0.95-x[self.b_index])/self.mass_range])
		fb1min = np.max([-0.1/self.mass_range,-(x[self.b_index]-0.02)/self.mass_range])
		x[self.b_index+1] = truncnorm.ppf(u[i], fb1min/0.1, fb1max/0.1, loc=0.0, scale=0.1)

		log_h = np.log10(h0)

		prior = truncnorm.pdf(fb1, fb1min/0.1, fb1max/0.1, loc=0.0, scale=0.1) * \
				truncnorm.pdf(fo, 0.0, 6.0, loc=0.0, scale=self.scale_fo) * \
				truncnorm.pdf(log_h,-0.3/0.2,1.0/0.2,loc=0.0,scale=0.2) * \
				truncnorm.pdf(h1,0.0,2.0/(0.4*h0),loc=0.0,scale=0.4*h0)

		return np.log(prior)


	def prior_transform(self,u):

		assert self.q_model in ['power','legendre','piecewise','hist']

		assert self.m_model in ['power']

		x = self.default_params.copy()

		ind = 0

		ind, x[:self.q_index] = self.prior_transform_mass_power(u)

		if self.q_model == 'power':

			ind, x[self.q_index:self.b_index] = self.prior_transform_q_power(u,ind)

		if self.q_model == 'legendre':

			ind, x[self.q_index:self.b_index] = self.prior_transform_q_legendre(u,ind)

		if self.q_model == 'piecewise':

			ind, x[self.q_index:self.b_index] = self.prior_transform_q_piecewise(u,ind)

		if self.q_model == 'hist':

			ind, x[self.q_index:self.b_index] = self.prior_transform_q_hist(u,ind)

		x[self.b_index:] = self.prior_transform_general(u,ind)

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


	def dynesty_sample(self,prefix='dy_',jitter=False,bound='multi',sample='rwalk',nlive=2000):

		from dynesty import NestedSampler, DynamicNestedSampler
		from dynesty import plotting as dyplot
		from dynesty import utils as dyfunc

		self.prefix = prefix

		ndim = int(self.ndim - np.sum(self.freeze))

		labels = [self.labels[i] for i in range(self.ndim) if self.freeze[i] == 0]

		#sampler = NestedSampler(self.lnlikelihood, self.prior_transform, ndim, bound=bound,sample=sample,nlive=nlive)
		sampler = DynamicNestedSampler(self.lnlikelihood, self.prior_transform, ndim, bound=bound,sample=sample)


		sampler.run_nested()

		res = sampler.results

		#samples, weights = res.samples, np.exp(res.logwt - res.logz[-1])
		samples, weights = res.samples, res.importance_weights()

		np.save(prefix+'samples.npy',samples)
		np.save(prefix+'weights.npy',weights)

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

		if jitter:

			lnzs = np.zeros((100, len(res.logvol)))
			for i in range(100):
				res_j = dyfunc.jitter_run(res)
				lnzs[i] = res_j.logz[-1]
			lnz_mean, lnz_std = np.mean(lnzs), np.std(lnzs)
			print('Jitter logz:        {:6.3f} +/- {:6.3f}'.format(lnz_mean, lnz_std))
			with open(f'{prefix}_lnZ.txt','w') as file:
				file.write(f'{lnz_mean:8.3f}    {lnz_std:8.3f}\n')



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


