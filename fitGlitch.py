import numpy as np
import os
import sys
import loadData as ld
import supportGlitch as sg
import plots
import h5py
import glob



#-----------------------------------------------------------------------------------------
def obsFit(inFreq, outPng, outHdf5, num_of_l=3, n_rln=1000, method='FQ', tauhe=None, 
    dtauhe=None, taucz=None, dtaucz=None, regu_param=7., tol_grad=1.e-3, n_guess=200):
    '''
    Glitch fitting for the observed frequencies and second differences
    
    Parameters
    ----------
    inFreq : str
        Input observed frequency file.
    outPng : str
        Output png file containing summary plots for the fit
    outHdf5 : str
        Output hdf5 file containing fitted parameters
    num_of_l : int
        Number of harmonic degrees (starting from l = 0)
    n_rln : int
        Number of realizations. If n_rln = 0, just fit the original frequencies/differences
    method : str
        Fitting method ('FQ' or 'SD')
    tauhe : float, optional
        Determines the range in acoustic depth (s) of He glitch for global minimum 
        search (tauhe - dtauhe, tauhe + dtauhe). 
        If tauhe = None, tauhe = 0.16 * acoustic_radius + 48 
    dtauhe : float, optional
        Determines the range in acoustic depth (s) of He glitch for global minimum 
        search (tauhe - dtauhe, tauhe + dtauhe). 
        If dtauhe = None, dtauhe = 0.05 * acoustic_radius
    taucz : float, optional
        Determines the range in acoustic depth (s) of CZ glitch for global minimum 
        search (taucz - dtaucz, taucz + dtaucz). 
        If taucz = None, taucz = 0.37 * acoustic_radius + 900
    dtaucz : float, optional
        Determines the range in acoustic depth (s) of CZ glitch for global minimum 
        search (taucz - dtaucz, taucz + dtaucz). 
        If dtaucz = None, dtaucz = 0.10 * acoustic_radius
    regu_param : float
        Regularization parameter (7 and 1000 generally work well for 'FQ' and 
        'SD', respectively)
    tol_grad : float
        tolerance on gradients (typically between 1e-2 and 1e-5 depending on quality 
        of data and 'method' used)
    n_guess : int
        Number of initial guesses in search for the global minimum

    Return
    ------
    An hdf5 file containing the fitted parameters
    A png file containing summary plots for the fit
    '''
#-----------------------------------------------------------------------------------------

    # Load observed oscillation frequencies
    freq, num_of_mode, num_of_n, delta_nu = ld.loadFreq(inFreq, num_of_l)
    print ()
    print ('Total number of modes = %d' %(num_of_mode))
    print ('Number of degrees for each l = ', num_of_n)
    print ('Large frequency separation = %.2f' %(delta_nu))


    # Compute second differences (if being fitted)
    num_of_dif2, freqDif2, icov = None, None, None
    if method.lower() == 'sd': 
        num_of_dif2, freqDif2, icov = sg.compDif2(num_of_l, freq, num_of_mode, num_of_n)
        print ()
        print ('Total number of SDs = %d' %(num_of_dif2))
    
    
    # Fit glitch signatures
    param, chi2, reg, ier = sg.fit(freq, num_of_n, delta_nu, num_of_dif2=num_of_dif2, 
        freqDif2=freqDif2, icov=icov, method=method, n_rln=n_rln, tol_grad=tol_grad, 
        regu_param=regu_param, n_guess=n_guess, tauhe=tauhe, dtauhe=dtauhe, taucz=taucz, 
        dtaucz=dtaucz)
    param_rln, ier_rln = param[0:n_rln, :], ier[0:n_rln]
    param_rln = param_rln[ier_rln == 0, :]
    nfit_rln = param_rln.shape[0]
    print ('Failed realizations = %d' %(n_rln - nfit_rln))

    # Print average amplitude, acoustic depth and phase of CZ signature
    print ()
    vmin, vmax = np.amin(freq[:, 2]), np.amax(freq[:, 2])
    Acz_rln, Ahe_rln = np.zeros(nfit_rln), np.zeros(nfit_rln)
    for j in range(nfit_rln):
        Acz_rln[j], Ahe_rln[j] = sg.averageAmplitudes(param_rln[j, :], vmin, vmax, 
            delta_nu=delta_nu, method=method)
    Acz, AczNErr, AczPErr = sg.medianAndErrors(Acz_rln)
    print ('Median Acz,  nerr,  perr = %8.4f %8.4f %8.4f' %(Acz, AczNErr, AczPErr))
    Tcz, TczNErr, TczPErr = sg.medianAndErrors(param_rln[:, -6])
    print ('Median Tcz,  nerr,  perr = %8.1f %8.1f %8.1f' %(Tcz, TczNErr, TczPErr))
    Pcz, PczNErr, PczPErr = sg.medianAndErrors(param_rln[:, -5])
    print ('Median Pcz,  nerr,  perr = %8.4f %8.4f %8.4f' %(Pcz, PczNErr, PczPErr))
    
    # Print average amplitude, acoustic width, acoustic depth and phase of He signature
    print ()
    Ahe, AheNErr, AhePErr = sg.medianAndErrors(Ahe_rln)
    print ('Median Ahe,  nerr,  perr = %8.4f %8.4f %8.4f' %(Ahe, AheNErr, AhePErr))
    Dhe, DheNErr, DhePErr = sg.medianAndErrors(param_rln[:, -3])
    print ('Median Dhe,  nerr,  perr = %8.3f %8.3f %8.3f' %(Dhe, DheNErr, DhePErr))
    The, TheNErr, ThePErr = sg.medianAndErrors(param_rln[:, -2])
    print ('Median The,  nerr,  perr = %8.2f %8.2f %8.2f' %(The, TheNErr, ThePErr))
    Phe, PheNErr, PhePErr = sg.medianAndErrors(param_rln[:, -1])
    print ('Median Phe,  nerr,  perr = %8.4f %8.4f %8.4f' %(Phe, PheNErr, PhePErr))


    # Generate a summary plot
    _ = plots.fitSummary(num_of_l, freq, num_of_n, delta_nu, param[-1, :], param_rln, 
            n_rln, method=method, freqDif2=freqDif2, tauhe=tauhe, dtauhe=dtauhe, 
            taucz=taucz, dtaucz=dtaucz, filename=outPng)
    
    
    # Store data in a HDF5 file
    if os.path.exists(outHdf5):
        os.remove(outHdf5)
    with h5py.File(outHdf5, 'w') as f:
        f.create_dataset('header/method', data=method)
        f.create_dataset('header/regu_param', data=regu_param)
        f.create_dataset('header/tol_grad', data=tol_grad)
        if tauhe is not None:
            f.create_dataset('header/tauhe', data=tauhe)
        if dtauhe is not None:
            f.create_dataset('header/dtauhe', data=dtauhe)
        if taucz is not None:
            f.create_dataset('header/taucz', data=taucz)
        if dtaucz is not None:
            f.create_dataset('header/dtaucz', data=dtaucz)
        f.create_dataset('obs/num_of_l', data=num_of_l)
        f.create_dataset('obs/freq', data=freq)
        f.create_dataset('obs/num_of_n', data=num_of_n)
        f.create_dataset('obs/delta_nu', data=delta_nu)
        if freqDif2 is not None:
            f.create_dataset('obs/freqDif2', data=freqDif2)
        if icov is not None:
            f.create_dataset('obs/icov', data=icov)
        f.create_dataset('fit/param', data=param)
        f.create_dataset('fit/chi2', data=chi2)
        f.create_dataset('fit/reg', data=reg)
        f.create_dataset('fit/ier', data=ier)

    return



#-----------------------------------------------------------------------------------------
# Main program 
#-----------------------------------------------------------------------------------------
num_of_l, n_rln, n_guess, tol_grad, method = 3, 1000, 200, 1.e-3, 'SD'

tauhe, dtauhe, taucz, dtaucz = None, None, None, None

if method.lower() == 'fq':
    regu_param = 7.e0
elif method.lower() == 'sd':
    regu_param = 1.e3
else:
    print ('Fitting method is not recognized. Terminating the run...')
    sys.exit(1)

path, stars = './', ['16cyga']


# End of parameter definition
#-----------------------------------------------------------------------------------------


for star in stars:
    print ()
    print ('Fitting star in folder %s...' %(path + star))
    inFreq = path + star + '/freq.txt'
    tolStr = str(round(abs(np.log10(tol_grad)), 1))
    outPng = path + star + '/' + method + tolStr + '_fitSummary.png'
    outHdf5 = path + star + '/' + method + tolStr + '_fitSummary.hdf5'
    _ = obsFit(inFreq, outPng, outHdf5, num_of_l=num_of_l, n_rln=n_rln, method=method, 
            tauhe=tauhe, dtauhe=dtauhe, taucz=taucz, dtaucz=dtaucz, 
            regu_param=regu_param, tol_grad=tol_grad, n_guess=n_guess)
