


import sys
import numpy as np
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from statsmodels.stats.weightstats import DescrStatsW

from CMDFitter5 import CMDFitter



fitter = CMDFitter(13.5,18,q_model='legendre')


exts=[1,2,3,4]
models = ['A','B','C','D']
freeze = [(6,7,8),(),(6,7,8,12),(12,)]


q = np.linspace(0,1,101)

yq1 = np.zeros(101)
yq2 = np.zeros(101)
yq3 = np.zeros(101)
yq4 = np.zeros(101)
yq5 = np.zeros(101)

sig1 = 0.5 * 68.27
sig2 = 0.5 * 95.45

q_tab = np.arange(10)*0.1
fb_tab = np.zeros((10,4))
fb_sigma_p = np.zeros((10,4))
fb_sigma_m = np.zeros((10,4))

fig, ax = plt.subplots(3,4,figsize=(7,5),sharex='col',sharey='row')

col = 0
for ext, model, f in zip(exts,models,freeze):

    fitter.freeze = np.zeros(13)
    for i in f:
        fitter.freeze[i] = 1

    try:
        s = np.load(f'NS5_t{ext}_samples.npy')
        w = np.load(f'NS5_t{ext}_weights.npy')
    except:
        continue

    samples = np.zeros([s.shape[0],13])
    samples[:,np.where(fitter.freeze==0)[0]] = s

    ns = s.shape[0]

    for i in range(1000):
        p = fitter.default_params.copy()
        ind = random.choices(range(ns),k=1,weights=w.tolist())
        p[np.where(fitter.freeze == 0)] = s[ind]
        args = p[3:9].tolist()
        args.append(fitter.M_ref)
        ax[0,col].plot(q,fitter.q_distribution(q,args),'b-',alpha=0.01)
        deriv = p[3]*fitter.der_sl_1(q) + p[4]*fitter.der_sl_2(q) + p[5]*fitter.der_sl_3(q)
        ax[1,col].plot(q,deriv,'b-',alpha=0.01)

    for j in range(101):

        y = np.zeros(len(samples))

        for i in range(len(samples)):

            p = samples[i]
            y[i] = p[9] * ((1.0  - fitter.int_sl_0(q[j])) + \
                           p[3]*(0.0 - fitter.int_sl_1(q[j])) + \
                           p[4]*(0.0 - fitter.int_sl_2(q[j])) + \
                           p[5]*(0.0 - fitter.int_sl_3(q[j])) )

        wq  = DescrStatsW(data=y,weights=w)
        qq = wq.quantile(probs=np.array(0.01*np.array([50.0-sig2,50.0-sig1,50.0,50.0+sig1,50.0+sig2])),\
                    return_pandas=False)

        yq1[j] = qq[0]
        yq2[j] = qq[1]
        yq3[j] = qq[2]
        yq4[j] = qq[3]
        yq5[j] = qq[4]

    for j in range(10):

        y = np.zeros(len(samples))

        for i in range(len(samples)):

            p = samples[i]
            y[i] = p[9] * ((1.0  - fitter.int_sl_0(q_tab[j])) + \
                           p[3]*(0.0 - fitter.int_sl_1(q_tab[j])) + \
                           p[4]*(0.0 - fitter.int_sl_2(q_tab[j])) + \
                           p[5]*(0.0 - fitter.int_sl_3(q_tab[j])) )

        wq  = DescrStatsW(data=y,weights=w)
        qq = wq.quantile(probs=np.array(0.01*np.array([50.0-sig1,50.0,50.0+sig1])),\
                    return_pandas=False)

        fb_sigma_m[j,ext-1] = qq[1]-qq[0]
        fb_tab[j,ext-1] = qq[1]
        fb_sigma_p[j,ext-1] = qq[2] - qq[1]

    ax[2,col].fill_between(q,y1=yq1,y2=yq5,color='b',alpha=0.1)
    ax[2,col].fill_between(q,y1=yq2,y2=yq4,color='b',alpha=0.4)

    ax[2,col].plot(q,yq3,'r-')

    ax[2,col].set_xlabel(r'$q$')

    ax[0,col].text(0.96,2.8,f"Model {model}",horizontalalignment='right',verticalalignment='top')

    ax[0,col].set_ylim((0,3))

    ax[0,col].set_xlim((0,1))
    ax[1,col].set_xlim((0,1))
    ax[2,col].set_xlim((0,1))
    ax[1,col].set_ylim((-15,15))
    ax[2,col].set_ylim((0,0.9))

    ax[0,col].tick_params(axis='y',which='both',direction='in',right=True)
    ax[1,col].tick_params(axis='y',which='both',direction='in',right=True)
    ax[2,col].tick_params(axis='y',which='both',direction='in',right=True)
    ax[0,col].tick_params(axis='x',which='both',direction='in',top=True)
    ax[1,col].tick_params(axis='x',which='both',direction='in',top=True)
    ax[2,col].tick_params(axis='x',which='both',direction='in',top=True)

    ax[1,col].plot([0,1],[0,0],'k--')

    col += 1

ax[0,0].set_ylabel(r'$P(q)$')
ax[1,0].set_ylabel(r'$dP(q)\,/\,dq$')
ax[2,0].set_ylabel(r"$f_B \, (q'>q)$")

fig.tight_layout()

plt.savefig('fb_q_dist_NS5.png')


print('\\begin{tabular}{lcccc}')
print('\\hline')

line = '$q$ '

for mod in range(len(exts)):

    line += f' & {models[mod]} '

line += '\\\\'

print(line)

print('\hline')

for i in range(10):

    line = f'${q_tab[i]:0.1f}$ '

    for j in range(4):

            line += f' & $ {fb_tab[i,j]:5.3f}_{{-{fb_sigma_m[i,j]:5.3f}}}^{{+{fb_sigma_p[i,j]:5.3f}}} $'

    line += '\\\\'

    print(line)
    print('\\\\')

print('\hline')
print('\end{tabular}')






