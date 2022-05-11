import numpy as np


from CMDFitter5 import CMDFitter

fitter = CMDFitter(13.5,18,q_model='legendre')

fitter.freeze[12] = 1

fitter.dynesty_sample(prefix='NS5_t4_')

