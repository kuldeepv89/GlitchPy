# The GlitchPy Code

The GlitchPy Code infers properties of the helium ionisation zone and the base of the envelope convection zone and calculates the frequency ratios using the observed set of oscillation frequencies for the Sun-like main-sequence stars. There are two possible options to fit the glitch signatures:

(1) directly fit the individual oscillation frequencies; or

(2) fit the second differences of frequencies w.r.t. the radial order. 

The details of the methods (1) and (2) have been described in Verma et al. (2019) [https://ui.adsabs.harvard.edu/abs/2019MNRAS.483.4678V/abstract].


## Installation and Setup

The GlitchPy code can be downloaded via the git clone command, e.g.,

       git clone https://github.com/kuldeepv89/GlitchPy.git
or,

       git clone git@github.com:kuldeepv89/GlitchPy.git

The code requires standard python packages including `numpy` (version 1.21.4 or later), `scipy` (version 1.7.3 or later), `h5py` (version 3.6.0 or later) and `sklearn`/`scikit-learn` (version 0.21.3 or later). 


## Input Parameters

The input parameters in  `main.py`  are accepted through  `stars.xml`  file (brief explanation given below).

### Modes Group 

`path (str) :` Path to the oscillation frequencies. For each `star` in the `stars` list (see below), an ascii file with the $~~~~~~~~~~~~~~~~~~~~~~~~~$ name `star.txt` must exist in the folder `path` 

`stars (list) :` List of star names/IDs  

Output results go to the folder `path/star_method/` (where method is either `FQ` or `SD`, see below)

`num_of_l (int) :` Number of harmonic degrees to read from the data file starting from the radial modes (i.e. if $~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$ `num_of_l = 1`, only `l = 0` modes is read) 

`rtype (str) :` Ratio type (`r01`, `r10`, `r02`, `r010`, `r012`, `r102`)

If `rtype = None`,  only glitch properties are calculated (ratios are ignored)

### Numerical Group 

`method (str) :` Fitting method (frequencies: `FQ`; second differences: `SD`)

If fitting method is taken as `FQ`, then -

`npoly_params (int) :` Number of parameters in the smooth component (i.e. degree of polynomial + 1), in this case, $~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$ `fourth degree polynomial`, works quite well

`nderiv (int) :` Order of derivative used in the regularization, in this case, `third derivative`,  works quite well

`regu_param (float) :` Regularization parameter, in this case, a value of about `7` works quite well for above values $~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$ of `npoly_params` and `nderiv`

If fitting method is taken as `SD`, then - 

`npoly_params (int) :` Number of parameters in the smooth component (i.e. degree of polynomial + 1), in this case, $~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$ `second degree polynomial`, works quite well

`nderiv (int) :` Order of derivative used in the regularization, in this case, `first derivative`,  works quite well

`regu_param (float) :` Regularization parameter, in this case, a value of about `1000` works quite well for above $~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$ values of `npoly_params` and `nderiv`

By default, the fitting method is taken as `FQ`. However, if you want to take the fitting method as `SD`, you need to make appropriate changes in the XML file.

A set of recommended values is given below: 

       method = "SD"

       npoly_params = 3

       nderiv = 1

       regu_param = 1000

`n_rln (int) :` Number of realizations to fit for uncertainties/covariance matrix estimation

`tot_grad (float) :` Absolute tolerance on gradients during the optimization

`n_guess (int) :` Number of initial guesses to explore the parameter space for finding the global minimum

### Physical Group 

`tauhe (float)` & `dtauhe (float) :` Initial guess for the range of acoustic depth of HeIZ 
- Range : `[tauhe - dtauhe, tauhe + dtauhe]`
- if `tauhe = None`, `tauhe = 0.17 * acousticRadius + 18` 
- if `dtauhe = None`, `dtauhe = 0.05 * acousticRadius`

`taucz (float)` & `dtaucz (float) :` Initial guess for the range of acoustic depth of BCZ
- Range : `[taucz - dtaucz, taucz + dtaucz]`
- if `taucz = None`, `taucz = 0.34 * acousticRadius + 929`
- if `dtaucz = None`, `dtaucz = 0.10 * acousticRadius`

`vmin (float)` & `vmax (float) :` Frequency range for the average amplitude
- Range : `[vmin, vmax]`
- if `vmin = None`, assume `minimum` value of the fitted frequencies
- if `vmax = None`, assume `maximum` value of the fitted frequencies

## Running the Code

Go through the following steps to run the code :

* Go to the GlitchPy directory in the terminal
* Run the following command :
  
         python3 main.py

Running the code for the first time will produce some warnings due to compilation of Fortran codes to form Python modules. Ignore these warnings. These warnings will not be shown from the second time.



