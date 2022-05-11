import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from CMDFitter5 import CMDFitter



fitter = CMDFitter(13.5,18,q_model='legendre')

M = np.linspace(fitter.mass_slice[0],fitter.mass_slice[1],101)

plt.figure(figsize=(4,5))

G = fitter.data[:,0]
R = fitter.data[:,1]
n = len(fitter.data)
plt.scatter(G-R,G,s=1,color='k')
plt.ylim((18.5,12.5))
plt.xlim((0.4,1.1))
plt.grid()
plt.xlabel(r'$G - R$')
plt.ylabel(r'$G$')

for q in [0.0,0.4,0.6,0.8,1.0]:
    G, R = fitter.binary(M,q)
    GmR = G - R
    plt.plot(GmR,G,'-')

plt.savefig('data_iso.png')


