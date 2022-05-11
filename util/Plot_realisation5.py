


import sys
import numpy as np

from statsmodels.stats.weightstats import DescrStatsW

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def get_cov_ellipse(cov, centre, nstd, **kwargs):
    """
    Return a matplotlib Ellipse patch representing the covariance matrix
    cov centred at centre and scaled by the factor nstd.

    """

    # Find and sort eigenvalues and eigenvectors into descending order
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # The anti-clockwise angle to rotate our ellipse by 
    vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
    theta = np.arctan2(vy, vx)

    # Width and height of ellipse to draw
    width, height = 2 * nstd * np.sqrt(eigvals)
    return Ellipse(xy=centre, width=width, height=height,
                   angle=np.degrees(theta), **kwargs)




from CMDFitter5 import CMDFitter

exts=[1,2,3,4]
models = ['A','B','C','D']
freeze = [(6,7,8),(),(6,7,8,12),(12,)]


fitter = CMDFitter(13.5,18,q_model='legendre')



for ext, model, f in zip(exts,models,freeze):

    fitter.freeze = np.zeros(13)
    for i in f:
        fitter.freeze[i] = 1

    try:
        s = np.load(f'NS5_t{ext}_samples.npy')
        w = np.load(f'NS5_t{ext}_weights.npy')
        samples = np.zeros([s.shape[0],13])
        samples[:,np.where(fitter.freeze==0)[0]] = s
        print(ext,s.shape,samples.shape)

    except:
        continue

    p = fitter.default_params

    for param in range(13):

        wq = DescrStatsW(data=samples[:,param],weights=w)
        p[param] = wq.quantile(probs=np.array([0.5]), return_pandas=False)[0]

    print(p)

    fraction_good = 1.0 - p[10]
    fraction_single = (1.0-p[10]-p[9])/fraction_good

    mean_sig_G = np.mean(fitter.data[:,3])
    mean_sig_R = np.mean(fitter.data[:,4])
    mean_sig_GR = np.sqrt(mean_sig_G**2 + mean_sig_R**2)

    plt.figure(figsize=(16,6))

    plt.subplot(1,4,1)
    G = fitter.data[:,0]
    R = fitter.data[:,1]
    n = len(fitter.data)
    plt.scatter(G-R,G,s=1,color='k')
    plt.ylim((18.5,12.5))
    plt.xlim((0.4,1.1))
    plt.grid()
    plt.xlabel(r'$G - R$')
    plt.ylabel(r'$G$')

    #plt.title(model)

    n_single = int(round(n*fraction_single))

    #log_k, M0, gamma, a1, a2, a3, a1_dot, a2_dot,a3_dot, fb, fo, h0, h1 = p
    #pp = [a1, a2, a3, a1_dot, a2_dot,a3_dot,fitter.M_ref]
    #qdi = fitter.q_distribution_sampler(pp)
    #for j in range(10):
    #    x = 0.0 + 0.1*j
    #    print(x,qdi(x))

    #sys.exit()

    for j in range(3):
        plt.subplot(1,4,2+j)
        colour, mag = fitter.model_realisation(p,n,add_observational_scatter=True)
        plt.scatter(colour[:n_single:],mag[:n_single],s=1,color='b')
        plt.scatter(colour[n_single:],mag[n_single:],s=1,color='r')
        plt.ylim((18.5,12.5))
        plt.xlim((0.4,1.1))
        plt.grid()
        plt.xlabel(r'$G - R$')
        plt.ylabel(r'$G$')

        G = np.array([19,18,17,16])
        R = np.array([17.88,17,16.75,15.31])
        #G_err, R_err = fitter.compute_observational_scatter(G,R)
        #print('G_err',G_err)
        #print('R_err',R_err)
        #h = p[8] + p[9]*(G-16.0)
        #for i in range(len(G)):
        #    S = h[i]**2 * np.array([[G_err[i]**2 + R_err[i]**2, G_err[i]**2],[G_err[i]**2,G_err[i]**2]])
        #    ell = get_cov_ellipse(S, np.array([0.6,G[i]]), 1.0, fc='m', ec='k')
        #    plt.gca().add_patch(ell)

    plt.savefig(f'CMDs_t{model}.png')




