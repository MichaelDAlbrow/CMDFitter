import numpy as np


from CMDFitter5 import CMDFitter

fitter = CMDFitter(13.5,18,q_model='legendre')


fitter.dynesty_sample(prefix='NS5_t2_')

