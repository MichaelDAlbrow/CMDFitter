import sys
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pylab import subplots_adjust

from time import time

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from likelihood_gaussbf_CUDA import likelihood_functions
likelihood = likelihood_functions.get_function("likelihood")


class CMDFitter():


	def __init__(self,G_min,G_max,trim_data=True,q_model='power',scale_data_errors=None):


		assert q_model in ['power','legendre']

		if q_model == 'power':
			self.ndim = 10
			self.labels = [r"$\log_{10} k$", r"$M_0$", r"$\gamma$",  r"$\beta$", r"$\alpha$", r"$B$", r"$f_B$", r"$f_O$", r"$h_0$", r"$h_1$"]

		if q_model == 'legendre':
			self.ndim = 13
			self.labels = [r"$\log_{10} k$", r"$M_0$", r"$\gamma$",  r"$a_1$", r"$a_2$", r"$a_3$", \
						r"$\dot{a}_1$", r"$\dot{a}_2$", r"$\dot{a}_3$",r"$f_B$", r"$f_O$", r"$h_0$", r"$h_1$"]

		self.freeze = np.zeros(self.ndim)
		self.default_params = np.zeros(self.ndim)
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

		G_offset = 9.55
		R_offset = 9.538
		iso_file = 'MIST_iso_5GYr_06Fe.txt'
		iso_data = np.loadtxt(iso_file)
		cut = np.where((iso_data[:,30]-iso_data[:,32])>0.25)[0]
		iso_data = iso_data[cut]
		iso_G = iso_data[:,30] + G_offset
		iso_R = iso_data[:,32] + R_offset
		iso_M = iso_data[:,3]
		pts = np.where((iso_G>13.5) & (iso_G<19.5))[0]
		fg  = np.flip(iso_G[pts])

		interp_err = np.load('interp_err.npy')
		self.error_interp = PchipInterpolator(interp_err[:,0],interp_err[:,1])


		self.G_M_interp = PchipInterpolator(np.flip(iso_G[pts]),np.flip(iso_M[pts]))
		self.M_G_interp = PchipInterpolator(iso_M[pts],iso_G[pts])
		self.M_R_interp = PchipInterpolator(iso_M[pts],iso_R[pts])
		self.GR_G_interp = PchipInterpolator(np.flip(iso_G[pts]-(iso_R[pts]-self.error_interp(iso_G[pts]))),np.flip(iso_G[pts]))




		#self.data = np.load('binary_data.npy')
		self.data = np.loadtxt('CMD_data.txt')

		if trim_data:
			self.trim_data(G_min,G_max)


		self.mass_slice = np.array([self.G_M_interp(G_max),self.G_M_interp(G_min)])
		self.M_ref = np.mean(self.mass_slice)
		self.delta_M = np.abs(self.mass_slice[0] - self.M_ref)
		self.delta_G = np.max(np.abs(np.array([G_min,G_max])-16.0))

		#pts = np.where(interp_g>13.5)[0]
		#print(np.flip(interp_m[pts]))
		#fg  = np.flip(interp_g[pts])
		#print(fg)
		#print(fg[1:]-fg[:-1])
		#self.G_M_interp = PchipInterpolator(np.flip(interp_g[pts]),np.flip(interp_m[pts]))
		#self.M_G_interp = PchipInterpolator(interp_m,interp_g)
		#self.M_R_interp = PchipInterpolator(interp_m,interp_r)

		if scale_data_errors is not None:
			self.data[:,3] *= scale_data_errors
			self.data[:,4] *= scale_data_errors


		self.data_CUDA = likelihood_functions.get_texref("data")
		drv.matrix_to_texref(np.float32(self.data),self.data_CUDA,order='F')
		self.data_CUDA.set_filter_mode(drv.filter_mode.POINT)

		self.n_bf = 50

		self.Mw = 0.01
		self.M0 = np.linspace(self.mass_slice[0]-0.2,self.mass_slice[1]-(self.Mw/2),self.n_bf)
		self.qw = 0.012
		self.q0 = np.linspace(self.qw*2,1,self.n_bf)

		self.qsigma = 0.01
		self.qx = np.linspace(0.001,1,200)
		self.qbf = np.zeros([self.n_bf,200])
		for i in range(self.n_bf):
			self.qbf[i,:] = (1/(self.qw*(2*np.pi)**0.5))*np.exp(-((self.qx-self.q0[i])**2)/(2*self.qw**2)) 

		self.qA = np.zeros((self.n_bf,self.n_bf))
		for k in range(self.n_bf):
			for j in range(self.n_bf):
				self.qA[k,j] = np.sum(self.qbf[k,:]*self.qbf[j,:]/self.qsigma**2) 

		self.Msigma = 0.01
		self.Mx = np.linspace(0.1,1.1,200)

		self.Mbf = np.zeros([self.n_bf,200])
		for i in range(self.n_bf):
			self.Mbf[i,:] = (1/(self.Mw*(2*np.pi)**0.5))*np.exp(-((self.Mx-self.M0[i])**2)/(2*self.Mw**2)) 

		self.MA = np.zeros((self.n_bf,self.n_bf))
		for k in range(self.n_bf):
			for j in range(self.n_bf):
				self.MA[k,j] = np.sum(self.Mbf[k,:]*self.Mbf[j,:]/self.Msigma**2) 

		M_i = np.linspace(self.mass_slice[0],self.mass_slice[1],1024)
	
		D_i = np.zeros([1024,2])
		for i in range(len(M_i)):    
			G, R = self.binary(M_i[i],0)
			D_i[i] = np.array([G-R,G])

		D_ij = np.zeros([self.n_bf**2,2])
		S_ij = np.zeros([self.n_bf**2,2,2])
		width_matrix = np.zeros([2,2])
		for i in range(self.n_bf):
			for j in range(self.n_bf):
				jacob = self.jacobian(self.M0[i],self.q0[j]) 
				width_matrix[0,0] = self.Mw**2
				width_matrix[1,1] = self.qw**2
				G, R = self.binary(self.M0[i],self.q0[j])
				D_ij[i+j*self.n_bf] = np.array([G-R,G])
				S_ij[i+j*self.n_bf] = np.dot(jacob,(np.dot(width_matrix,(jacob.T))))

		S_ij_shaped = S_ij.reshape(self.n_bf**2,4)


		self.DM_CUDA = likelihood_functions.get_texref("DM")
		drv.matrix_to_texref(np.float32(D_i),self.DM_CUDA,order='F')
		self.DM_CUDA.set_filter_mode(drv.filter_mode.POINT)

		self.DMQ_CUDA = likelihood_functions.get_texref("DMQ")
		drv.matrix_to_texref(np.float32(D_ij),self.DMQ_CUDA,order='F')
		self.DMQ_CUDA.set_filter_mode(drv.filter_mode.POINT)

		self.SMQ_CUDA = likelihood_functions.get_texref("SMQ")
		drv.matrix_to_texref(np.float32(S_ij_shaped),self.SMQ_CUDA,order='F')
		self.SMQ_CUDA.set_filter_mode(drv.filter_mode.POINT)

		self.likelihood = likelihood_functions.get_function("likelihood")

		self.emcee_walker_dispersion = 1.e-7

		return


	def trim_data(self,G_min,G_max):
		M_min = self.G_M_interp(G_min)
		M_max = self.G_M_interp(G_max)
		q = np.linspace(0,1,11)
		B_min = self.binary(M_min,q)
		B_max = self.binary(M_max,q)
		B_min_interp = PchipInterpolator(np.flip(B_min[0]),np.flip(B_min[0]-B_min[1]))
		B_max_interp = PchipInterpolator(np.flip(B_max[0]),np.flip(B_max[0]-B_max[1]))
		data_GmR = self.data[:,0] - self.data[:,1]
		good_points = np.where( ( (self.data[:,0] > G_min) & (self.data[:,0] < G_max - 0.75) )  | \
								( (self.data[:,0] > G_min - 0.75) & (self.data[:,0] < G_min) & (data_GmR > B_min_interp(self.data[:,0]) - 0.005 ) ) | \
								( (self.data[:,0] < G_max) & (self.data[:,0] > G_max - 0.75) & (data_GmR < B_max_interp(self.data[:,0]) + 0.01 ) ) )[0]
		self.data = self.data[good_points]
		data_iso_G = self.GR_G_interp(self.data[:,0]-self.data[:,1])
		good_points = np.where( (self.data[:,0] - data_iso_G > -0.9) & (self.data[:,0] - data_iso_G < 0.15) )
		self.data = self.data[good_points]
		return


	def M_to_G(self,M):
		G = self.M_G_interp(M)
		return G


	def M_to_R(self,M):
		R = self.M_R_interp(M)
		return R


	def m_to_F(self,m,c=20):
		F = 10**(0.4*(c-m))
		return F


	def F_to_m(self,F,c=20):
		m = c - 2.5*np.log10(F)
		return m


	def M_distribution(self,x,log_k,x0,gamma,maxM):
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
		k = 10.0**log_k
		m = np.linspace(0.1,1.1,1000)
		dm = m[1]-m[0]
		y = m**(-gamma) / (1.0 + np.exp(-k*(m-x0)))
		y_cumulative = np.cumsum(y)/np.sum(y)
		pts = np.where(y_cumulative>1.e-50)[0]
		return PchipInterpolator(y_cumulative[pts],m[pts])


	def q_distribution(self,x,params):

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

		assert self.q_model in ['power','legendre']

		q = np.linspace(0,1,1001)
		dq = q[1]-q[0]

		if self.q_model == 'power':
			beta,alpha,B,M = params
			power = alpha + beta*(M-self.M_ref)
			y = (alpha + beta*(M-self.M_ref) + 1.0)*(1.0-B)*q**power + B

		if self.q_model == 'legendre':
			a1, a2, a3, a1_dot, a2_dot,a3_dot, M = params
			dM = M-self.M_ref
			y = self.sl_0(q) + (a1+a1_dot*dM)*self.sl_1(q) + (a2+a2_dot*dM)*self.sl_2(q) + (a3+a3_dot*dM)*self.sl_3(q)
			
		y_cumulative = np.cumsum(y)

		return PchipInterpolator(y_cumulative/y_cumulative[-1],q)


	def compute_observational_scatter(self,G,R):

		# This assumes obs data for G,R,G_err,R_err in self.data[0,1,3,4]

		from scipy.optimize import curve_fit

		nbins = 10
		x1 = np.linspace(np.min(self.data[:,0]),np.max(self.data[:,0]),nbins+1)
		xm = np.zeros(nbins)
		ym = np.zeros(nbins)
		for i in range(nbins):
			pts = np.where((self.data[:,0]>=x1[i]) & (self.data[:,0]<x1[i+1]))
			ym[i] = np.median(self.data[pts,3])
		xm[i] = 0.5*(x1[i]+x1[i+1])
		yoffset = 0.99*np.min(ym)
		f, cov = curve_fit(lambda t,a,b: a*np.exp(b*t),  xm,  ym-yoffset)
		G_err = f[0]*np.exp(f[1]*G) + yoffset

		x1 = np.linspace(np.min(self.data[:,1]),np.max(self.data[:,1]),nbins+1)
		for i in range(nbins):
			pts = np.where((self.data[:,1]>=x1[i]) & (self.data[:,1]<x1[i+1]))
			ym[i] = np.median(self.data[pts,4])
		xm[i] = 0.5*(x1[i]+x1[i+1])
		yoffset = 0.99*np.min(ym)
		f, cov = curve_fit(lambda t,a,b: a*np.exp(b*t),  xm,  ym-yoffset)
		R_err = f[0]*np.exp(f[1]*R) + yoffset

		return G_err, R_err


	def model_realisation(self,p,n,add_observational_scatter=True):

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
		print('mass range',M_sampler(np.array([0.01,0.99])))

		q = np.zeros_like(M1)
		G = np.zeros_like(M1)
		R = np.zeros_like(M1)

		for i in range(n_binary):

			if self.q_model == 'power':
				q_sampler = self.q_distribution_sampler([alpha,beta,B,M1[n_single+i]])
			if self.q_model == 'legendre':
				q_sampler = self.q_distribution_sampler([a1,a2,a3,a1_dot,a2_dot,a3_dot,M1[n_single+i]])

			q[n_single+i] = q_sampler(np.random.rand())
			
		G, R = self.binary(M1,q)
		GmR = G - R

		if add_observational_scatter:

			G_err, R_err = self.compute_observational_scatter(G,R)

			h = h0 + h1*(G-16.0)

			for i in range(len(G)):
				S = h[i]**2 * np.array([[G_err[i]**2 + R_err[i]**2, G_err[i]**2],[G_err[i]**2,G_err[i]**2]])
				z = np.random.multivariate_normal(mean=np.zeros(2), cov=S, size=1)
				GmR[i] += z[0][0]
				G[i] += z[0][1]

		return GmR, G



	def binary(self,M,q):
		#Returns the G,R values for a binary system with primary mass M and mass ratio q.
		M1 = M
		M2 = q*M
		G1 = self.M_to_G(M1)
		G2 = self.M_to_G(M2)
		R1 = self.M_to_R(M1)
		R2 = self.M_to_R(M2)
		G_flux = self.m_to_F(G1) + self.m_to_F(G2)
		R_flux = self.m_to_F(R1) + self.m_to_F(R2)
		G = self.F_to_m(G_flux)
		R = self.F_to_m(R_flux) - self.error_interp(G)
		return G, R


	def jacobian(self,M,q):
		jacob = np.zeros((2,2))
		Gfixq, Rfixq = self.binary(self.M0,q)
		GfixM, RfixM = self.binary(M,self.q0)
		G_q = PchipInterpolator(self.q0,GfixM)
		G_M = PchipInterpolator(self.M0,Gfixq)
		R_q = PchipInterpolator(self.q0,RfixM)
		R_M = PchipInterpolator(self.M0,Rfixq)
		jacob[0,0] = G_M(M,1)-R_M(M,1)
		jacob[0,1] = G_q(q,1)-R_q(q,1)
		jacob[1,0] = G_M(M,1)
		jacob[1,1] = G_q(q,1)
		return jacob


	def norm(self,x,A,b):
		return np.linalg.norm(np.dot(A,x)-b)


	def M_gauss(self,Mparams):
		
		log_k, M0, gamma = Mparams

		My = self.M_distribution(self.Mx,log_k, M0, gamma, self.mass_slice[1])
		
		Mb = np.zeros(self.n_bf)
		for k in range(self.n_bf):
			Mb[k] = np.sum(My*self.Mbf[k,:]/self.Msigma**2)
		
		Ma = np.linalg.solve(self.MA,Mb)
		#Ma[self.M0<(tanh0-2*tanh_width)] = 0.0
		Ma[self.M0 < (M0 - 4.0/k)] = 0.0

		if np.min(Ma) < 0:
			result = minimize(self.norm, np.zeros(self.n_bf), args=(self.MA,Mb), method='L-BFGS-B', bounds=[(0.,None) for x in range(self.n_bf)])
			Ma = result.x

		norm_c = np.sum(Ma)
		return Ma/norm_c


	def q_gauss(self,params):
				
		qy = self.q_distribution(self.qx,params)

		qb = np.zeros(self.n_bf)
		for k in range(self.n_bf):
			qb[k] = np.sum(qy*self.qbf[k,:]/self.qsigma**2)
		
		qa = np.linalg.solve(self.qA,qb)

		if np.min(qa) < 0:
			result = minimize(self.norm, np.zeros(self.n_bf), args=(self.qA,qb), method='L-BFGS-B', bounds=[(0.,None) for x in range(self.n_bf)])
			qa = result.x

		norm_c = np.sum(qa)
		return qa/norm_c


	def precalc(self,params):
		
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
		
		p = self.default_params
		p[self.freeze==0] = params

		assert self.q_model in ['power','legendre']

		if self.q_model == 'power':

			log_k, M0, gamma, beta, alpha, B, fb, fo, h0, h1 = p

		if self.q_model == 'legendre':

			log_k, M0, gamma, a1, a2, a3, a1_dot, a2_dot, a3_dot, fb, fo, h0, h1 = p

			# Check that the parameters generate a positive q distribution for all masses
			for MM in np.linspace(self.mass_slice[0],self.mass_slice[1],11).tolist():
				args = p[3:9].tolist()
				args.append(MM)
				q_dist_test = self.q_distribution(np.linspace(0.0,1.0,1001),args)
				if np.min(q_dist_test) < 0.0:
					return self.neginf


		P_i, PMQ = self.precalc(p)


		c_P_i = np.ascontiguousarray(P_i.astype(np.float64))
		c_PMQ = np.ascontiguousarray(PMQ.astype(np.float64))
		
		blockshape = (int(256),1, 1)
		gridshape = (len(self.data), 1)

		lnP_k = np.zeros(len(self.data)).astype(np.float64)

		likelihood(drv.In(c_P_i), drv.In(c_PMQ), np.float64(h0), np.float64(h1), np.float64(fo), np.float64(fb), drv.InOut(lnP_k), block=blockshape, grid=gridshape)

		lnP = np.sum(lnP_k)

		self.lnP_k = lnP_k


		if not(np.isfinite(lnP)):
			with open(self.prefix+'.err', 'a') as f:
				f.write(np.array2string(params,max_line_width=1000).strip('[]\n')+'\n')
			return self.neginf

		return lnP


	def ln_prior(self,params):

		from scipy.stats import norm, truncnorm

		p = self.default_params
		p[self.freeze==0] = params

		assert self.q_model in ['power','legendre']

		if self.q_model == 'power':

			log_k, M0, gamma, beta, alpha, B, fb, fo, h0, h1 = p

			if fb < 0.02 or fb > 0.95 or fo < 0.0 or fo > 0.05 or B < 0.0 or B > (1.0 + 1.0/(alpha+beta*self.delta_M)):
				return self.neginf 

			log_h = np.log10(h0)

			prior = norm.pdf(log_k,loc=2.0,scale=0.3) * norm.pdf(M0,loc=0.55,scale=0.01) * norm.pdf(gamma,loc=0.0,scale=1.0) * norm.pdf(beta,loc=0.0,scale=2.0) * \
						truncnorm.pdf(alpha + np.abs(beta)*self.delta_M,0.0,20.0,loc=0.0,scale=3.0) * norm.pdf(log_h,loc=0.1,scale=0.3) * \
						truncnorm.pdf(h1,0.0,2.0,loc=0.0,scale=0.4*h0)

		if self.q_model == 'legendre':

			log_k, M0, gamma, a1, a2, a3,  a1_dot, a2_dot, a3_dot, fb, fo, h0, h1 = p

			if fb < 0.02 or fb > 0.95 or fo < 0.0 or fo > 0.05:
				return self.neginf 


			log_h = np.log10(h0)

			prior = norm.pdf(log_k,loc=2.0,scale=0.3) * norm.pdf(M0,loc=0.55,scale=0.01) * norm.pdf(gamma,loc=0.0,scale=1.0) * \
							norm.pdf(a1,loc=0.0,scale=2.0) * \
							norm.pdf(a2,loc=0.0,scale=2.0) * norm.pdf(a3,loc=0.0,scale=2.0) *  \
							norm.pdf(a1_dot,loc=0.0,scale=0.1*np.abs(a1)/self.delta_M) * \
							norm.pdf(a2_dot,loc=0.0,scale=0.1*np.abs(a2)/self.delta_M) * norm.pdf(a3_dot,loc=0.0,scale=0.1*np.abs(a3)/self.delta_M) * \
							norm.pdf(log_h,loc=0.1,scale=0.3) * truncnorm.pdf(h1,0.0,2.0*h0,loc=0.0,scale=0.4*h0)

		return np.log(prior)


	def prior_transform(self,u):

		from scipy.stats import norm, truncnorm

		assert self.q_model in ['power','legendre']

		x = self.default_params

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
				x[1] = norm.ppf(u[i], loc=0.55, scale=0.01)
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
				x[6] = norm.ppf(u[i], loc=0.0, scale=0.1*np.abs(x[3])/self.delta_M)
				i += 1
			if not self.freeze[7]:
				# a2
				x[7] = norm.ppf(u[i], loc=0.0, scale=0.1*np.abs(x[4])/self.delta_M)
				i += 1
			if not self.freeze[8]:
				# a3
				x[8] = norm.ppf(u[i], loc=0.0, scale=0.1*np.abs(x[5])/self.delta_M)
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


	def ultranest_sample(self,prefix='un_'):

		import ultranest

		self.neginf = sys.float_info.min

		self.prefix = prefix

		labels = [self.labels[i] for i in range(self.ndim) if self.freeze[i] == 0]

		output_dir = prefix+'output'

		sampler = ultranest.ReactiveNestedSampler(labels, self.lnlikelihood, self.prior_transform,log_dir=output_dir,resume=True)

		result = sampler.run()

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





