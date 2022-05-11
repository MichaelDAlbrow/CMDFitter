import sys
import numpy as np
from sigfig import round
from statsmodels.stats.weightstats import DescrStatsW

from CMDFitter5 import CMDFitter


def get_last_line(filepath):
    try:
        with open(filepath,'rb') as f:
            f.seek(-1,os.SEEK_END)
            text = [f.read(1)]
            while text[-1] != '\n'.encode('utf-8') or len(text)==1:
                f.seek(-2, os.SEEK_CUR)
                text.append(f.read(1))
    except Exception as e:
        pass
    return ''.join([t.decode('utf-8') for t in text[::-1]]).strip()


fitter = CMDFitter(13.5,18,q_model='legendre')


exts=[1,2,3,4]
models = ['A','B','C','D']
freeze = [(6,7,8),(),(6,7,8,12),(12,)]

table = np.zeros([len(exts),13,3])
logz = np.ones(len(exts))*572
lnp = np.zeros(len(exts))
max_lnprob = np.zeros(len(exts))

print('\\begin{tabular}{lcccc}')
print('\\hline')



line = ''
for mod in range(len(exts)):

    line += f' & {models[mod]} '

line += '\\\\'

print(line)

print('\hline')


for ext, model, f in zip(exts,models,freeze):

    fitter.freeze = np.zeros(13)
    for i in f:
        fitter.freeze[i] = 1

    s = np.load(f'NS5_t{ext}_samples.npy')
    w = np.load(f'NS5_t{ext}_weights.npy')
    samples = np.zeros([s.shape[0],13])
    samples[:,np.where(fitter.freeze==0)[0]] = s

    log_file = f'log.NS5-t{ext}.1'
    with open(log_file,'r') as f:
        for line in f:
            elements = line.split()
            if len(elements) > 2:
                if elements[0] == 'logz:':
                    logz[ext-1] = float(elements[1])


    emlp = np.load(f'EM5_l{ext}_flatlnprob.npy')

    max_lnprob[ext-1] = np.max(emlp)


    for param in range(13):

        wq = DescrStatsW(data=samples[:,param],weights=w)
        res = wq.quantile(probs=np.array([0.5,0.16,0.84]), return_pandas=False)
        table[ext-1,param,0]  = res[0]
        table[ext-1,param,1]  = res[0] - res[1]
        table[ext-1,param,2]  = res[2] - res[0]

    lnp[ext-1] = fitter.lnlikelihood(table[ext-1,:,0][fitter.freeze==0])

    #print(table[ext-1,:,0])
    #print(fitter.ln_prior(table[ext-1,:,0][fitter.freeze==0]))
    #print(fitter.lnlikelihood(table[ext-1,:,0][fitter.freeze==0]))

for param in range(13):

    line = fitter.labels[param]
    for mod in range(len(exts)):
        s0,  _, sm = round(table[mod,param,0],uncertainty=np.min(table[mod,param,1:]),cutoff=30).split()
        s1 = round(table[mod,param,1],uncertainty=sm,cutoff=30).split()[0]
        s2 = round(table[mod,param,2],uncertainty=sm,cutoff=30).split()[0]
        #print(param,freeze[mod],param+1 in freeze[mod])
        if param in freeze[mod]:
            line += f' & $0$'
        else:
            line += f' & $ {s0}_{{-{s1}}}^{{+{s2}}} $'

    line += '\\\\'

    print(line)
    print('\\\\')

print('\hline')

line = '$\ln P (NS)$'
for mod in range(len(exts)):
    
    line += f' & $ {lnp[mod]:.2f} $'

line += '\\\\'

print(line)

line = '$\ln P (EM)$'
for mod in range(len(exts)):
    
    line += f' & $ {max_lnprob[mod]:.2f} $'

line += '\\\\'

print(line)

dlnp = lnp-np.max(lnp)

line = '$\Delta \ln P (NS)$'
for mod in range(len(exts)):
    
    line += f' & $ {dlnp[mod]:.2f} $'

line += '\\\\'

print(line)

dlnp = max_lnprob-np.max(max_lnprob)

line = '$\Delta \ln P (EM)$'
for mod in range(len(exts)):
    
    line += f' & $ {dlnp[mod]:.2f} $'

line += '\\\\'

print(line)

dlogz = logz-np.max(logz)

line = '$\Delta \log_{10} Z$'
for mod in range(len(exts)):
    
    line += f' & $ {dlogz[mod]:.2f} $'

line += '\\\\'

print(line)

print('\hline')

print('\\end{tabular}')



