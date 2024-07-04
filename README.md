# The `GlitchPy` package

The package takes oscillation frequencies of the Sun-like main-sequence stars as the input. The main output is the set of parameters characterising the helium ionisation zone (or the He glitch) and the base of the envelope convection zone (or the BCZ glitch). Additionally, if needed, the code provides the large frequency separation and the frequency ratios. The output further includes the corresponding full covariance matrix.  

The details can be found in Verma et al. (2019) [https://ui.adsabs.harvard.edu/abs/2019MNRAS.483.4678V/abstract].


## Installation and Setup

The code can be downloaded using the following commands,

       git clone https://github.com/kuldeepv89/GlitchPy.git
or,

       git clone git@github.com:kuldeepv89/GlitchPy.git

It requires standard python packages including `numpy` (version 1.21.4 or later), `scipy` (version 1.7.3 or later), `h5py` (version 3.6.0 or later) and `sklearn`/`scikit-learn` (version 0.21.3 or later). 


## Input data

In the `stars.xml` file, the input parameters can be set appropriately. The parameters are briefly defined and explained below.

### Frequency Group 

`path : str` 
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; path to the `starid`.txt file containing the oscillation frequencies (for `starid`, see the physical group below) <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; output data and plots are stored in `path/starid_method/` (for method, see the numerical group below)

`num_of_l : int` 
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; number of harmonic degrees starting from `l = 0` to use in calculations 

`rtype : str` 
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; type of frequency ratios (choose from `r01`, `r10`, `r02`, `r010`, `r012`, `r102`) <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; if `rtype = None`, ratios are not calculated

`include_dnu : bool`
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; if `include_dnu = True`, calculate large frequency separation (otherwise don't)

### Numerical Group 

`method : str` 
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; fitting method (either `FQ` to fit frequencies or `SD` to fit second differences)

`npoly_params : int` 
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; number of parameters in the polynomial (smooth component) <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `npoly_params = 5 (3)` is tested and works well for `method = FQ (SD)`

`nderiv : int` 
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; order of the derivative used in the regularization <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `nderiv = 3 (1)` is tested and works well for `method = FQ (SD)`

`regu_param : float` 
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; regularization parameter <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `regu_param = 7 (1000)` is tested and works well for `method = FQ (SD)` (assuming above values of `npoly_params` and `nderiv`)

`tot_grad : float` 
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; absolute tolerance on the gradient of the cost function during its minimisation

`n_guess : int` 
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; number of initial guesses to search the global minimum

`n_rln : int` 
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; number of data realisations to estimate the covariance matrix

### Physical Group 

`starid : str` 
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; star name/ID 

`delta_nu : float` 
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; large frequency separation (muHz) <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; if `delta_nu = None`, it is internally calculated using radial modes 

`nu_max : float` 
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; frequency of maximum power (muHz) to be used in `delta_nu` calculation (when not given) <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; if `nu_max = None`, calculate `delta_nu` without applying any weight in the least-squares fit 

`tauhe : float`
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; initial guess for the acoustic depth of the He glitch (s) <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; if `tauhe = None`, `tauhe = 0.17 * acousticRadius + 18` (`acousticRadius = 1 / 2 * delta_nu)`

`dtauhe : float` 
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; guess for the error in `tauhe` (s) <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; if `dtauhe = None`, `dtauhe = 0.05 * acousticRadius` <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; search for the global minimum in the range `[tauhe - dtauhe, tauhe + dtauhe]`

`taucz : float` 
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; initial guess for the acoustic depth of the BCZ glitch (s) <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; if `taucz = None`, `taucz = 0.34 * acousticRadius + 929`

`dtaucz : float` 
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; initial guess for the error in `taucz` (s) <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; if `dtaucz = None`, `dtaucz = 0.10 * acousticRadius` <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; search for the global minimum in the range `[taucz - dtaucz, taucz + dtaucz]`

`taucz_min : float` 
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; discard realisations with fitted `taucz` below `taucz_min` (s) <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; if `taucz_min = None`, `taucz_min = 0.`

`taucz_max : float` 
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; discard realisations with fitted `taucz` above `taucz_max` (s) <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; if `taucz_max = None`, `taucz_max = acousticRadius`

`vmin : float` 
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; lower frequency limit in the amplitude averaging (muHz) <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; if `vmin = None`, use the smallest frequency as `vmin`

`vmax : float` 
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; upper frequency limit in the amplitude averaging (muHz) <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; if `vmax = None`, use the largest frequency as `vmax`
 
Note. While analysing multiple stars, the `star` group is repeated with appropriate values of `starid` and other parameters in this group.

## Running the Code

After setting up the `stars.xml` file suitably, run the following command in the `GlitchPy` directory (runtime about one hour).
  
       python3 main.py

Note. Running the code for the first time will produce some warnings due to the conversion of Fortran codes to Python modules. The warnings may be ignored (they will not show up from second run onward).
