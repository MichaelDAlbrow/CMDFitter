import pycuda.driver as drv
import pycuda.compiler
import pycuda.autoinit

from pycuda.compiler import SourceModule

likelihood_functions = SourceModule("""

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "texture_fetch_functions.h"
#include "texture_types.h"


#define THREADS_PER_BLOCK 256

#define MAX_GRID_SIZE 1000

#define _USE_MATH_DEFINES
#ifndef M_PI
#    define M_PI 3.1415926535897932
#endif

#define ZP 25.0


#define ind_G 0
#define ind_R 1
#define ind_sigma_G 3
#define ind_sigma_R 4

texture<float, 2> DMQ, SMQ;

//
// data should have 2 columns corresponding to colour and magnitude
//
// cov should be a [n,2,2] array, where n is the number of data points
//
texture<float, 2> data;
texture<float, 2> cov;
//texture<float, 3, cudaReadModeElementType> cov;

extern "C" {


// __device__ float random(curandState* global_state, int thread_id) {
// 	curandState local_state = global_state[thread_id];
// 	float num = curand_uniform(&local_state);
// 	global_state[thread_id] = local_state;
// 	return num;
// }



__device__ double linear_interp(double *x, double *y, int nx, double x0)
{

	int i;

	if (x0 >= x[nx-1]) return y[nx-1];

	for (i = 0; x[i] <= x0; i++);

	return y[i] + (x0 - x[i]) * (y[i+1] - y[i]) / (x[i+1] - x[i]);

}



__device__ double logaddexp(double x, double y)
{
	double xmy = x - y;

	if (xmy > 0.0) return x + log1p(exp(-xmy));

	return y + log1p(exp(xmy));

}





__device__ void parallel_logaddexp(double *A_local)
{
			
	__syncthreads();

	if ((int)(blockDim.x) >= 512)
	{ 
		if (threadIdx.x < 256)
		{
			A_local[threadIdx.x] = logaddexp(A_local[threadIdx.x],A_local[threadIdx.x + 256]);
		}
	}

	__syncthreads();
	
	if ((int)(blockDim.x) >= 256)
	{ 
		if (threadIdx.x < 128)
		{
			A_local[threadIdx.x] = logaddexp(A_local[threadIdx.x],A_local[threadIdx.x + 128]);
		}
	}

	__syncthreads();
	
	if ((int)(blockDim.x) >= 128)
	{ 
		if (threadIdx.x < 64)
		{
			A_local[threadIdx.x] = logaddexp(A_local[threadIdx.x],A_local[threadIdx.x + 64]);
		}
	}

	__syncthreads();
	
	if (threadIdx.x < 32)
	{
		if (blockDim.x >= 64) A_local[threadIdx.x] = logaddexp(A_local[threadIdx.x],A_local[threadIdx.x + 32]);
		if (blockDim.x >= 32) A_local[threadIdx.x] = logaddexp(A_local[threadIdx.x],A_local[threadIdx.x + 16]);
		if (blockDim.x >= 16) A_local[threadIdx.x] = logaddexp(A_local[threadIdx.x],A_local[threadIdx.x + 8]);
		if (blockDim.x >= 8) A_local[threadIdx.x] = logaddexp(A_local[threadIdx.x],A_local[threadIdx.x + 4]);
		if (blockDim.x >= 4) A_local[threadIdx.x] = logaddexp(A_local[threadIdx.x],A_local[threadIdx.x + 2]);
		if (blockDim.x >= 2) A_local[threadIdx.x] = logaddexp(A_local[threadIdx.x],A_local[threadIdx.x + 1]);
	}
	
	__syncthreads();

}


__device__ double outlier_likelihood(double *info, double *Dk, double f_outlier){
	
	double D0[2], D[2], S0[2][2], detS0, invS0[2][2];

	D0[0] = info[0];
	D0[1] = info[1];

	S0[0][0] = info[2]*info[2];
	S0[1][1] = info[3]*info[3];
	S0[0][1] = 0.0;
	S0[1][0] = 0.0;

    D[0] = Dk[0] - D0[0];
    D[1] = Dk[1] - D0[1];

    detS0 = S0[0][0]*S0[1][1] - S0[0][1]*S0[1][0];

    invS0[0][0] = S0[1][1] / detS0;
    invS0[1][1] = S0[0][0] / detS0;
    invS0[0][1] = -S0[0][1] / detS0;
    invS0[1][0] = -S0[1][0] / detS0;

    double DSD = D[0]*(invS0[0][0]*D[0] + invS0[0][1]*D[1]) + D[1]*(invS0[1][0]*D[0] + invS0[1][1]*D[1]);

    return -0.5*DSD - log(2.0*M_PI/f_outlier) - 0.5*log(detS0);

}


__device__ void single_likelihood(double h, double *Dk, double Sk[2][2], double *PM, double f_single, double *result){
	
	int nMB = 50;

	double D[2], detS, S[2][2], invS[2][2], DSD;

	__shared__ double lnp[THREADS_PER_BLOCK];

    lnp[threadIdx.x] = -1.e50;

    for (int i = threadIdx.x; i<nMB; i+= blockDim.x){

	    D[0] = Dk[0] - tex2D(DMQ,i,0);
    	D[1] = Dk[1] - tex2D(DMQ,i,1);

		S[0][0] = h*h*Sk[0][0] + tex2D(SMQ,i,0);
		S[0][1] = h*h*Sk[0][1] + tex2D(SMQ,i,1);
		S[1][0] = h*h*Sk[1][0] + tex2D(SMQ,i,2);
		S[1][1] = h*h*Sk[1][1] + tex2D(SMQ,i,3);

	    detS = S[0][0]*S[1][1] - S[0][1]*S[0][1];

		invS[0][0] = S[1][1] / detS;
		invS[1][1] = S[0][0] / detS;
		invS[0][1] = -S[1][0] / detS;
		invS[1][0] = -S[0][1] / detS;

		DSD = D[0]*(invS[0][0]*D[0] + invS[0][1]*D[1]) + D[1]*(invS[1][0]*D[0] + invS[1][1]*D[1]);

		lnp[threadIdx.x] = logaddexp(lnp[threadIdx.x],-0.5*DSD + log(PM[i]) - 0.5*log(detS));

		//printf("i, Dk, DMQ, D0, Sk, DSD, lnp: %d %f  %f %f %f %f %f %f %f %f %f %f %f\\n",i,Dk[0],Dk[1],tex2D(DMQ,i,0),tex2D(DMQ,i,1),D[0], D[1], Sk[0][0], Sk[0][1],Sk[1][0],Sk[1][1],DSD, lnp[threadIdx.x]);
		//printf("i, Dk, DMQ, D0, DSD, lnp: %d %f  %f %f %f %f %f %f %f\\n",i,Dk[0],Dk[1],tex2D(DMQ,i,0),tex2D(DMQ,i,1),D[0], D[1], DSD, lnp[threadIdx.x]);
		//printf("i, DMQ: %d, %f %f \\n",i,tex2D(DMQ,i,0),tex2D(DMQ,i,1));
		//printf("i, SMQ: %d %f %f \\n",i, tex2D(SMQ,i,0),tex2D(SMQ,i,1),tex2D(SMQ,i,2),tex2D(SMQ,i,3));


	}

	__syncthreads();

	parallel_logaddexp(lnp);

	if (threadIdx.x == 0){

		*result = lnp[0] - log(2.0*M_PI/f_single);

	}

	__syncthreads();

	return;

}


__device__ void binary_likelihood(double h, double *Dk, double Sk[2][2], double *PMQ, double f_binary, double *result){
	
	int nMB = 50;
	int nQB = 50;

	double D[2], detS, S[2][2], invS[2][2], DSD;

	__shared__ double lnp[THREADS_PER_BLOCK];

    lnp[threadIdx.x] = -1.e50;

    for (int i = threadIdx.x; i<nMB*nQB; i+= blockDim.x){

	    D[0] = Dk[0] - tex2D(DMQ,i,0);
    	D[1] = Dk[1] - tex2D(DMQ,i,1);

		S[0][0] = h*h*Sk[0][0] + tex2D(SMQ,i,0);
		S[0][1] = h*h*Sk[0][1] + tex2D(SMQ,i,1);
		S[1][0] = h*h*Sk[1][0] + tex2D(SMQ,i,2);
		S[1][1] = h*h*Sk[1][1] + tex2D(SMQ,i,3);

	    detS = S[0][0]*S[1][1] - S[0][1]*S[0][1];

		invS[0][0] = S[1][1] / detS;
		invS[1][1] = S[0][0] / detS;
		invS[0][1] = -S[1][0] / detS;
		invS[1][0] = -S[0][1] / detS;

		DSD = D[0]*(invS[0][0]*D[0] + invS[0][1]*D[1]) + D[1]*(invS[1][0]*D[0] + invS[1][1]*D[1]);

		lnp[threadIdx.x] = logaddexp(lnp[threadIdx.x],-0.5*DSD + log(PMQ[i]) - 0.5*log(detS));

	}

	__syncthreads();

	parallel_logaddexp(lnp);

	if (threadIdx.x == 0){

		*result = lnp[0] - log(2.0*M_PI/f_binary);

	}

	__syncthreads();

	return;

}




__global__ void likelihood(double *PM_single, double *PMQ_binary, double *outlier_info, double h0, double h1, double h_magnitude_ref, double f_outlier, double f_binary, double *lnp_k){


	int k = blockIdx.x;

	__syncthreads();

	double Sk[2][2], Dk[2];

    Dk[0] = tex2D(data,k,0);
    Dk[1] = tex2D(data,k,1);

    //Sk[0][0] = tex3D(cov,k,0,0);
    //Sk[0][1] = tex3D(cov,k,0,1);
    //Sk[1][0] = tex3D(cov,k,1,0);
    //Sk[1][1] = tex3D(cov,k,1,1);
    Sk[0][0] = tex2D(cov,k,0);
    Sk[0][1] = tex2D(cov,k,1);
    Sk[1][0] = tex2D(cov,k,2);
    Sk[1][1] = tex2D(cov,k,3);

	double h = h0 + h1*(Dk[1] - h_magnitude_ref);

	double l_outlier = 0.0;
    if (threadIdx.x == 0) {
	    l_outlier = outlier_likelihood(outlier_info, Dk, f_outlier);
	}

	__syncthreads();

	double l_single = 0.0;
	single_likelihood(h, Dk, Sk, PM_single, 1.0-f_outlier-f_binary, &l_single);

	__syncthreads();

	double l_binary;
	binary_likelihood(h, Dk, Sk, PMQ_binary, f_binary, &l_binary);

	__syncthreads();

    if (threadIdx.x == 0) {
		lnp_k[k] = logaddexp(l_outlier,l_single);
		lnp_k[k] = logaddexp(lnp_k[k],l_binary);
		//printf("%d %f %f %f %f %f %f\\n",k,Dk[0],Dk[1],l_outlier,l_single,l_binary,lnp_k[k]);
		//for (int m = 0; m < 10; m++) {
		//   printf("%d %12.9f %12.9f %12.9f %12.9f\\n",m,tex2D(cov,m,0),tex2D(cov,m,1),tex2D(cov,m,2),tex2D(cov,m,3));
		//}
		//for (int m = 0; m < 10; m++) {
		//   printf("%d %f %f %12.9f %12.9f %12.9f %12.9f\\n",m,tex2D(DMQ,m,0),tex2D(DMQ,m,1),tex2D(SMQ,m,0),tex2D(SMQ,m,1),tex2D(SMQ,m,2),tex2D(SMQ,m,3));
		//}
	}

 	__syncthreads();

   return;

} 

}

""",no_extern_c=True)


