import numpy as np
import os
import sys
import loadData as ld
import supportGlitch as sg
import plots
import h5py
import glob



#-----------------------------------------------------------------------------------------
# Main program
#-----------------------------------------------------------------------------------------
# Number of harmonic degree
num_of_l = 3 


# Number of realizations
n_rln = 1000 


# Fitting method ("FQ" and "SD")
method = "FQ"


# Regularization parameter
# --> 7. and 1000. works well for "FQ" and "SD", respectively
regu_param = 7.


# Absolute tolerance on gradient
tol_grad = 1e-3


# Number of initial guesses to explore the parameter space 
# --> for finding the global minimum 
n_guess = 200


# Initial guess for the range of acoustic depth of HeIZ 
# --> range : [tauhe - dtauhe, tauhe + dtauhe]
# --> If tauhe = None, tauhe = 0.16 * acousticRadius + 48
# --> If dtauhe = None, dtauhe = 0.05 * acousticRadius
tauhe, dtauhe = None, None


# Initial guess for the range of acoustic depth of BCZ 
# --> range : [taucz - dtaucz, taucz + dtaucz]
# --> If taucz = None, taucz = 0.37 * acousticRadius + 900.
# --> If dtaucz = None, dtaucz = 0.10 * acousticRadius
taucz, dtaucz = None, None


# Frequency range for the average amplitude
# --> If vmin = None, assume minimum value of the fitted frequencies
# --> If vmax = None, assume maximum value of the fitted frequencies
vmin, vmax = None, None


# Ratio type ("r01", "r10", "r02", "r010", "r012", "r102")
rtype = "r012"


# Path and star names
# --> For each "star" in the list "stars" below, assume input frequency 
# file name to be stars.txt, which is present in the folder path/star/
# --> Output results goes to the folder path/star/
path, stars = './', ["16cyga"]
#-----------------------------------------------------------------------------------------



# Loop over stars
for star in stars:
    print ()
    print ("Fit star %s using method %s..." %(star, method))

    # Load observed oscillation frequencies
    filename = path + star + "/" + star + ".txt"
    if not os.path.isfile(filename):
        raise FileNotFoundError("Input frequency file not found %s!" %(filename))
    freq, num_of_mode, num_of_n, delta_nu = ld.loadFreq(filename, num_of_l)
    print ('Total number of modes = %d' %(num_of_mode))
    print ('Number of radial orders for each l = ', num_of_n)
    print ('Large frequency separation = %.2f' %(delta_nu))


    # Compute second differences (if being fitted)
    num_of_dif2, freqDif2, icov = None, None, None
    if method.lower() == 'sd': 
        num_of_dif2, freqDif2, icov = sg.compDif2(
            num_of_l, 
            freq, 
            num_of_mode, 
            num_of_n
        )
        print ('Total number of SDs = %d' %(num_of_dif2))
    
    
    # Fit glitch signatures
    param, chi2, reg, ier, ratio = sg.fit(
        freq, 
        num_of_n, 
        delta_nu, 
        num_of_dif2=num_of_dif2, 
        freqDif2=freqDif2, 
        icov=icov, 
        method=method, 
        n_rln=n_rln, 
        tol_grad=tol_grad, 
        regu_param=regu_param, 
        n_guess=n_guess, 
        tauhe=tauhe, 
        dtauhe=dtauhe, 
        taucz=taucz, 
        dtaucz=dtaucz,
        rtype=rtype
    )
    param_rln, ier_rln = param[0:n_rln, :], ier[0:n_rln]
    param_rln = param_rln[ier_rln == 0, :]
    nfit_rln = param_rln.shape[0]
    print ()
    print ("Fit completed...")
    print ('Failed realizations %d/%d' %(n_rln - nfit_rln, n_rln))

    # Print Summary
    # --> Print average amplitude, acoustic depth and phase of CZ signature
    if vmin is None:
        vmin = np.amin(freq[:, 2])
    if vmax is None:
        vmax = np.amax(freq[:, 2])
    Acz_rln, Ahe_rln = np.zeros(nfit_rln), np.zeros(nfit_rln)
    for j in range(nfit_rln):
        Acz_rln[j], Ahe_rln[j] = sg.averageAmplitudes(
            param_rln[j, :], 
            vmin, 
            vmax, 
            delta_nu=delta_nu, 
            method=method
        )
    Acz, AczNErr, AczPErr = sg.medianAndErrors(Acz_rln)
    print ("Median Acz,  nerr,  perr = " + 3 * "%9.4f" %(Acz, AczNErr, AczPErr))
    Tcz, TczNErr, TczPErr = sg.medianAndErrors(param_rln[:, -6])
    print ("Median Tcz,  nerr,  perr = " + 3 * "%9.1f" %(Tcz, TczNErr, TczPErr))
    Pcz, PczNErr, PczPErr = sg.medianAndErrors(param_rln[:, -5])
    print ("Median Pcz,  nerr,  perr = " + 3 * "%9.4f" %(Pcz, PczNErr, PczPErr))
    
    # Print average amplitude, acoustic width, acoustic depth and 
    # --> phase of He signature
    Ahe, AheNErr, AhePErr = sg.medianAndErrors(Ahe_rln)
    print ("Median Ahe,  nerr,  perr = " + 3 * "%9.4f" %(Ahe, AheNErr, AhePErr))
    Dhe, DheNErr, DhePErr = sg.medianAndErrors(param_rln[:, -3])
    print ("Median Dhe,  nerr,  perr = " + 3 * "%9.3f" %(Dhe, DheNErr, DhePErr))
    The, TheNErr, ThePErr = sg.medianAndErrors(param_rln[:, -2])
    print ("Median The,  nerr,  perr = " + 3 * "%9.2f" %(The, TheNErr, ThePErr))
    Phe, PheNErr, PhePErr = sg.medianAndErrors(param_rln[:, -1])
    print ("Median Phe,  nerr,  perr = " + 3 * "%9.4f" %(Phe, PheNErr, PhePErr))
    
    
    # Store data in HDF5 file
    outputdir = path + star + "/" + method + "/"
    if not os.path.isdir(outputdir):
        os.makedirs(outputdir)
    filename = outputdir + star + ".hdf5"  
    if os.path.isfile(filename):
        os.remove(filename)
    with h5py.File(filename, 'w') as f:
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
        f.create_dataset('obs/freq', data=freq)
        f.create_dataset('obs/num_of_n', data=num_of_n)
        f.create_dataset('obs/delta_nu', data=delta_nu)
        f.create_dataset('obs/vmin', data=vmin)
        f.create_dataset('obs/vmax', data=vmax)
        if freqDif2 is not None:
            f.create_dataset('obs/freqDif2', data=freqDif2)
        if icov is not None:
            f.create_dataset('obs/icov', data=icov)
        f.create_dataset('fit/param', data=param)
        f.create_dataset('fit/chi2', data=chi2)
        f.create_dataset('fit/reg', data=reg)
        f.create_dataset('fit/ier', data=ier)
        if rtype is not None:
            f.create_dataset('rto/rtype', data=rtype)
            f.create_dataset('rto/ratio', data=ratio)


    # Generate fit summary plots
    filename = outputdir + star + ".png"  
    plots.fitSummary(outputdir, star)
