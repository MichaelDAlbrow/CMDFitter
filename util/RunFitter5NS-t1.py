import numpy as np


from CMDFitter5 import CMDFitter

fitter = CMDFitter(13.5,18,q_model='legendre')

fitter.freeze[6] = 1
fitter.freeze[7] = 1
fitter.freeze[8] = 1

fitter.dynesty_sample(prefix='NS5_t1_')

