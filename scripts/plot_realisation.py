import sys
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.dirname(__file__))+'/../code')


from CMDFitter6 import CMDFitter, PlotUtils

definition_file = sys.argv[1]

fitter = CMDFitter(definition_file)

fitter.freeze[6] = 1
fitter.freeze[7] = 1
fitter.freeze[8] = 1

s = np.load(sys.argv[2])

params = fitter.default_params

params[fitter.freeze==0] = s

PlotUtils.plot_realisation(fitter,params,plot_file=sys.argv[3])





