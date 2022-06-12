import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__))+'/../code')


from CMDFitter6 import CMDFitter


definition_file = sys.argv[1]

fitter = CMDFitter(definition_file)

fitter.freeze[6] = 1
fitter.freeze[7] = 1
fitter.freeze[8] = 1

p = fitter.default_params.copy()

fitter.emcee_sample(p,prefix='EM_test_')


