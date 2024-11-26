# CMDFitter
Fit a probablistic generative model to a colour-magnitude diagram.

Please cite the following papers if you use this code:

  Albrow, M.D., Ulusele, I.H., 2022, MNRAS, 515, 730
  Albrow, M.D., 2024, MNRAS, 528, 6211
  Alexander, J.S., Albrow, M.D., 2025, MNRAS, in press


Basic usage:

```

from CMDFitter import CMDFitter

definition_file = 'M67.json'

fitter = CMDFitter(definition_file)

fitter.dynesty_sample(prefix='DY_test_')

```

See examples for running the code in the scripts folder, and a sample data file, isochrone and json definition file in the data folder.


 




  
