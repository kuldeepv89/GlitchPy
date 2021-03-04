import numpy as np
import h5py
import sys
from loadData import loadFit
import supportGlitch as sg 



path, star = '/Users/au572692/data/sebastien_hh1b/', 'gl_008_2_nm3_nse'
filename = path + star + '/' + 'FQ3.0_fitSummary.hdf5'

AheLower, tauczLower, tauczUpper = 0.001, 3200., 4600.
vmin, vmax = None, None


# Fit related data
header, obsData, fitData = loadFit(filename)

method, regu_param, tol_grad, tauhe, dtauhe, taucz, dtaucz = header
num_of_l, freq, num_of_n, delta_nu, freqDif2, icov = obsData
param, chi2, reg, ier = fitData


# Exclude original and failed fits 
n_rln, npar = param.shape[0] - 1, param.shape[1]
param_rln, ier_rln = param[0:n_rln, :], ier[0:n_rln]  
param_rln = param_rln[ier_rln == 0, :]
nfit_rln = param_rln.shape[0]


# Reduced chi-square
if method.lower() == 'fq':
    dof = freq.shape[0] - npar
elif method.lower() == 'sd':
    dof = freqDif2.shape[0] - npar
if dof <= 0:
    dof = 1
    #print ('Degree of freedom <= 0! Terminating the run...')
    #print ('%20s %8.2f %8.2f %8.2f %8.4f %8.4f %8.4f %8.4f %8d' %(star, -1., 
    #-1., -1., -1., -1., -1., -1., -1))
    #sys.exit(1)
rchi2 = chi2[-1] / dof


# Compute average amplitudes
if vmin is None:
    vmin = np.amin(freq[:, 2])
if vmax is None:
    vmax = np.amax(freq[:, 2])

Acz_rln, Ahe_rln = np.zeros(nfit_rln), np.zeros(nfit_rln)
for j in range(nfit_rln):
    Acz_rln[j], Ahe_rln[j] = sg.averageAmplitudes(param_rln[j, :], vmin, vmax,
        delta_nu=delta_nu, method=method)


# He parameters and their uncertainties
Ahe, AheNErr, AhePErr = sg.medianAndErrors(Ahe_rln[Ahe_rln > AheLower])
The, TheNErr, ThePErr = sg.medianAndErrors(param_rln[Ahe_rln > AheLower, -2])
print ('%20s %8.2f %8.2f %8.2f %8.4f %8.4f %8.4f %8.4f %8d' %(star, The, TheNErr, 
    ThePErr, Ahe, AheNErr, AhePErr, rchi2, n_rln - len(Ahe_rln[Ahe_rln > AheLower])))


# CZ parameters and their uncertainties
if tauczUpper is None: 
    tauczUpper = 5.e5 / delta_nu
mask = np.logical_and(param_rln[:, -6] > tauczLower, param_rln[:, -6] < tauczUpper)
Acz, AczNErr, AczPErr = sg.medianAndErrors(Acz_rln[mask]) 
Tcz, TczNErr, TczPErr = sg.medianAndErrors(param_rln[mask, -6]) 
print ('%20s %8.1f %8.1f %8.1f %8.4f %8.4f %8.4f %8.4f %8d %8.1f' %(star, Tcz, 
    TczNErr, TczPErr, Acz, AczNErr, AczPErr, rchi2, n_rln - nfit_rln,
    5.e5 / delta_nu - Tcz))
