import os
import sys
import numpy as np
import h5py
import glob
from sklearn.covariance import MinCovDet
import loadData as ld
import supportGlitch as sg
import utils_general as ug
import plots



#-----------------------------------------------------------------------------------------
# Set the following parameters/path/stars appropriately
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


# Store median values and covariance matrix
medCov = True


# Path and star names
# --> For each "star" in the list "stars" below, assume input frequency 
# file name to be stars.txt, which is present in the folder path/star/
# --> Output results go to the folder path/star/
path, stars = "./", ["16cyga"]
#-----------------------------------------------------------------------------------------



# Loop over stars
for star in stars:


    # Start the log file
    outputdir = path + star + "/" + method + "/"
    if not os.path.isdir(outputdir):
        os.makedirs(outputdir)
    filename = outputdir + "log.txt"
    stdout = sys.stdout
    sys.stdout = ug.Logger(filename)

    # Print header
    print (88 * "=")
    ug.prt_center(
        "FITTING SIGNATURES OF ACOUSTIC GLITCHES IN STELLAR OSCILLATION", 88
    )
    ug.prt_center(
        "FREQUENCIES (FQ) AS WELL AS IN SECOND DIFFERENCES (SD)", 88
    )
    print ()
    ug.prt_center("https://github.com/kuldeepv89/glitch-fitting", 88)
    print (88 * "=")

    # Print star name/ID
    print ()
    print ("Star name/ID: %s" %(star))


    # Load observed oscillation frequencies
    filename = path + star + "/" + star + ".txt"
    if not os.path.isfile(filename):
        raise FileNotFoundError("Input frequency file not found %s!" %(filename))
    freq, num_of_mode, num_of_n, delta_nu = ld.loadFreq(filename, num_of_l)
    if vmin is None:
        nu_min = np.amin(freq[:, 2])
    else:
        nu_min = vmin
    if vmax is None:
        nu_max = np.amax(freq[:, 2])
    else:
        nu_max = vmax
    print ()
    print ("The observed data:")
    print ("    - total number of modes: %d" %(num_of_mode))
    print ("    - number of n for each l:", num_of_n)
    print ("    - frequency range for averaging: (%.2f, %.2f) muHz" %(nu_min, nu_max))
    print ("    - large separation: %.2f muHz" %(delta_nu))

    # Compute second differences (if necessary)
    num_of_dif2, freqDif2, icov = None, None, None
    if method.lower() == 'sd': 
        num_of_dif2, freqDif2, icov = sg.compDif2(
            num_of_l, 
            freq, 
            num_of_mode, 
            num_of_n
        )
        print ("    - total number of SDs: %d" %(num_of_dif2))
    

    # Print fitting-method related information
    print ()
    print ("The fitting method and associated parameters:")
    print ("    - fitting method: %s" %(method))
    print ("    - regularization parameter: %.1f" %(regu_param))
    print ("    - absolute tolerance on gradients: %.1e" %(tol_grad))
    print ("    - number of attempts in global minimum search: %d" %(n_guess))

    # Print miscellaneous information    
    print ()
    print ("Miscellaneous information:")
    print ("    - tauhe, dtauhe: ({0}, {1})".format(tauhe, dtauhe))
    print ("    - taucz, dtaucz: ({0}, {1})".format(taucz, dtaucz))
    print ("    - ratio type: {0}".format(rtype))
    print ("    - store median and covariance: {0}".format(medCov))


    # Fit glitch signatures
    print ()
    print ("* Fitting data... ")
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
    print ("* Done!")
    param_rln, ier_rln = param[0:n_rln, :], ier[0:n_rln]
    param_rln = param_rln[ier_rln == 0, :]
    nfit_rln = param_rln.shape[0]
    npar = param.shape[1]
    if method.lower() == "fq":
        dof = freq.shape[0] - npar
    elif method.lower() == "sd":
        dof = freqDif2.shape[0] - npar
    if dof <= 0:
        print ()
        print ("WARNING: Degree of freedom %d <= 0! Setting it to 1..." %(dof))
        print ()
        dof = 1
    rchi2 = chi2[-1] / dof
    print ()
    print ("The fit and related summary data:")
    print ("    - total and reduced chi-squares: (%.4f, %.4f)" %(chi2[-1], rchi2))
    print ("    - failed realizations: %d/%d" %(n_rln - nfit_rln, n_rln))


    # Print fitted glitch parameters with uncertainties 
    # --> Print average amplitude, acoustic depth and phase of CZ signature
    Acz_rln, Ahe_rln = np.zeros(nfit_rln), np.zeros(nfit_rln)
    for j in range(nfit_rln):
        Acz_rln[j], Ahe_rln[j] = sg.averageAmplitudes(
            param_rln[j, :], 
            nu_min, 
            nu_max, 
            delta_nu=delta_nu, 
            method=method
        )
    Acz, AczNErr, AczPErr = sg.medianAndErrors(Acz_rln)
    print (
        "    - median Acz, nerr, perr: (%.4f, %.4f, %.4f)" %(Acz, AczNErr, AczPErr)
    )
    Tcz, TczNErr, TczPErr = sg.medianAndErrors(param_rln[:, -6])
    print (
        "    - median Tcz, nerr, perr: (%.1f, %.1f, %.1f)" %(Tcz, TczNErr, TczPErr)
    )
    Pcz, PczNErr, PczPErr = sg.medianAndErrors(param_rln[:, -5])
    print (
        "    - median Pcz, nerr, perr: (%.4f, %.4f, %.4f)" %(Pcz, PczNErr, PczPErr)
    )

    # Print average amplitude, acoustic width, acoustic depth and 
    # --> phase of He signature
    Ahe, AheNErr, AhePErr = sg.medianAndErrors(Ahe_rln)
    print (
        "    - median Ahe, nerr, perr: (%.4f, %.4f, %.4f)" %(Ahe, AheNErr, AhePErr)
    )
    Dhe, DheNErr, DhePErr = sg.medianAndErrors(param_rln[:, -3])
    print (
        "    - median Dhe, nerr, perr: (%.3f, %.3f, %.3f)" %(Dhe, DheNErr, DhePErr)
    )
    The, TheNErr, ThePErr = sg.medianAndErrors(param_rln[:, -2])
    print (
        "    - median The, nerr, perr: (%.2f, %.2f, %.2f)" %(The, TheNErr, ThePErr)
    )
    Phe, PheNErr, PhePErr = sg.medianAndErrors(param_rln[:, -1])
    print (
        "    - median Phe, nerr, perr: (%.4f, %.4f, %.4f)" %(Phe, PheNErr, PhePErr)
    )
    
    
    # Store all the data in a HDF5 file
    filename = outputdir + "fitData.hdf5"  
    if os.path.isfile(filename):
        os.remove(filename)
    with h5py.File(filename, 'w') as f:
        f.create_dataset('header/method', data=method)
        f.create_dataset('header/regu_param', data=regu_param)
        f.create_dataset('header/tol_grad', data=tol_grad)
        f.create_dataset('header/n_guess', data=n_guess)
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
        f.create_dataset('obs/vmin', data=nu_min)
        f.create_dataset('obs/vmax', data=nu_max)
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


    # Store median and covariance matrix in a HDF5 file (if necessary)
    if medCov:
        grparams = np.zeros((nfit_rln, 3))
        grparams[:, 0], grparams[:, 1], grparams[:, 2] = (
            Ahe_rln[:], param_rln[:, -3], param_rln[:, -2]
        ) 
        if rtype is not None:
            ratio_rln = ratio[0:n_rln, :]
            ratio_rln = ratio_rln[ier_rln == 0, :]
            grparams = np.hstack((ratio_rln, grparams))

        # Covariance
        j = int(round(nfit_rln / 2))
        covtmp = MinCovDet().fit(grparams[0:j, :]).covariance_
        cov = MinCovDet().fit(grparams).covariance_

        # Test convergence (change in standard deviations below a relative tolerance)
        rdif = np.amax(
            np.abs(
                np.divide(
                    np.sqrt(np.diag(covtmp)) - np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov))
                )
            )
        )
        if rdif > 0.1:
            print ()
            print (
                "WARNING: Maximum relative difference %.2e > 0.1! " 
                "Check covariance..." %(rdif)
            )
            print ()

        # Median
        norder, frq, rto = ug.specific_ratio(freq, rtype=rtype)
        ngr = grparams.shape[1]
        med = np.zeros(ngr)
        print ()
        print ("The observables with uncertainties from covariance matrix:")
        for i in range(ngr):
            med[i] = np.median(grparams[:, i])
            if i < ngr - 3:
                print (
                    "    - n, freq, median ratio, err: (%d, %.2f, %.5f, %.5f)" 
                    %(round(norder[i]), frq[i], rto[i], np.sqrt(cov[i, i]))
                )
            elif i == ngr - 3:
                print (
                    "    - median Ahe, err: (%.4f, %.4f)" %(med[i], np.sqrt(cov[i, i]))
                )
            elif i == ngr - 2:
                print (
                    "    - median Dhe, err: (%.3f, %.3f)" %(med[i], np.sqrt(cov[i, i]))
                )
            elif i == ngr - 1:
                print (
                    "    - median The, err: (%.2f, %.2f)" %(med[i], np.sqrt(cov[i, i]))
                )

        # Plot correlations
        filename = outputdir + "correlations.png"  
        plots.plot_correlations(cov, filename=filename)

        # Write to hdf5 file
        filename = outputdir + "medCov.hdf5"  
        if os.path.isfile(filename):
            os.remove(filename)
        with h5py.File(filename, "w") as ff:
            ff.create_dataset('header/method', data=method)
            ff.create_dataset('header/regu_param', data=regu_param)
            ff.create_dataset('header/tol_grad', data=tol_grad)
            ff.create_dataset('header/n_guess', data=n_guess)
            if rtype is None:
                ff.create_dataset("medglh", data=med)
                ff.create_dataset("covglh", data=cov)
            elif rtype == "r02":
                ff.create_dataset("medg02", data=med)
                ff.create_dataset("covg02", data=cov)
            elif rtype == "r01":
                ff.create_dataset("medg01", data=med)
                ff.create_dataset("covg01", data=cov)
            elif rtype == "r10":
                ff.create_dataset("medg10", data=med)
                ff.create_dataset("covg10", data=cov)
            elif rtype == "r010":
                ff.create_dataset("medg010", data=med)
                ff.create_dataset("covg010", data=cov)
            elif rtype == "r012":
                ff.create_dataset("medg012", data=med)
                ff.create_dataset("covg012", data=cov)
            elif rtype == "r102":
                ff.create_dataset("medg102", data=med)
                ff.create_dataset("covg102", data=cov)
            else:
                raise ValueError("Unrecognized ratio-type %s!" %(rtype))


    # Finally generate plots summarizing the fit
    plots.plot_fitSummary(outputdir)


    # Save log file
    sys.stdout = stdout
