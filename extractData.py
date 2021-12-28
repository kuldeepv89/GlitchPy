import numpy as np
import h5py
import sys
from loadData import loadFit
import supportGlitch as sg 



#-----------------------------------------------------------------------------------------
# Set the following path/parameters appropriately
#-----------------------------------------------------------------------------------------
# Path/folder containing fitted data 
path = "./16cyga/FQ/"

# Ignore realizations with average helium amplitude less than "AheLower" 
# --> while inferring the He glitch parameters (essentially ignore fits with 
# --> zero helium amplitude)
AheLower = 0.001

# Ignore realizations with "taucz" outside the range (tauczLower, tauczUpper)
# --> while inferring the CZ glitch parameters (use this to ignore aliased peaks)
# If tauczLower = None, set tauczLower = 0.
# If tauczUpper = None, set tauczUpper = acousticRadius
tauczLower, tauczUpper = None, None
#-----------------------------------------------------------------------------------------


# Load data
header, obsData, fitData, rtoData = loadFit(path + "fitData.hdf5")
method, regu_param, tol_grad, n_guess, tauhe, dtauhe, taucz, dtaucz = header
freq, num_of_n, delta_nu, vmin, vmax, freqDif2, icov = obsData
param, chi2, reg, ier = fitData
rtype, ratio = rtoData

# Exclude original and failed fits 
n_rln, npar = param.shape[0] - 1, param.shape[1]
param_rln, ier_rln = param[0:n_rln, :], ier[0:n_rln]  
param_rln = param_rln[ier_rln == 0, :]
nfit_rln = param_rln.shape[0]
    
# Compute reduced chi-square of the fit
if method.lower() == 'fq':
    dof = freq.shape[0] - npar
elif method.lower() == 'sd':
    dof = freqDif2.shape[0] - npar
if dof <= 0:
    dof = 1
    print ("WARNING: Degree of freedom <= 0! Setting it to 1...")
rchi2 = chi2[-1] / dof

# Compute average amplitudes
Acz_rln, Ahe_rln = np.zeros(nfit_rln), np.zeros(nfit_rln)
for j in range(nfit_rln):
    Acz_rln[j], Ahe_rln[j] = sg.averageAmplitudes(
        param_rln[j, :], 
        vmin, 
        vmax,
        delta_nu=delta_nu, 
        method=method
    )

# He glitch parameters and their uncertainties
Ahe, AheNErr, AhePErr = sg.medianAndErrors(Ahe_rln[Ahe_rln > AheLower])
The, TheNErr, ThePErr = sg.medianAndErrors(param_rln[Ahe_rln > AheLower, -2])
print (
    (9 * "%9s") 
    %("Tau (s)", "nerr", "perr", "A (muHz)", "nerr", "perr", "Chi2_r", 
        "Failed", "Alias"
    )
)
print (
    (3 * "%9.2f" + 4 * "%9.4f" + "%9d") 
    %(The, TheNErr, ThePErr, Ahe, AheNErr, AhePErr, rchi2, 
        n_rln - len(Ahe_rln[Ahe_rln > AheLower])
    )
)

# CZ glitch parameters and their uncertainties
if tauczLower is None: 
    tauczLower = 0.
if tauczUpper is None: 
    tauczUpper = 5.e5 / delta_nu
mask = np.logical_and(param_rln[:, -6] > tauczLower, param_rln[:, -6] < tauczUpper)
Acz, AczNErr, AczPErr = sg.medianAndErrors(Acz_rln[mask]) 
Tcz, TczNErr, TczPErr = sg.medianAndErrors(param_rln[mask, -6]) 
print (
    (3 * "%9.1f" + 4 * "%9.4f" + "%9d%9.1f") 
    %(Tcz, TczNErr, TczPErr, Acz, AczNErr, AczPErr, rchi2, 
        n_rln - nfit_rln, 5.e5 / delta_nu - Tcz
    )
)
