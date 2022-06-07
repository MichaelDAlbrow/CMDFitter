import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__))+'/../code')


from CMDFitter6 import CMDFitter

definition_file = sys.argv[1]

fitter = CMDFitter(definition_file)


