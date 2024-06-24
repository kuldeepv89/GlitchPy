import os
import sys
import time
import numpy as np
import h5py
from sklearn.covariance import MinCovDet
import supportGlitch as sg
import utils_general as ug
import plots
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import xml.etree.ElementTree as ET
tree=ET.parse('stars.xml')
root=tree.getroot()

def main():
    """Main"""

    
    # Modes Group
    path = root[0][0].attrib['value']
    stars = [root[0][1].attrib['value']]
    num_of_l = int(root[0][2].attrib['value'])
    rtype = root[0][3].attrib['value']
    
    # Numerical Group
    method = root[1][0].attrib['value']
    n_rln = int(root[1][1].attrib['value'])
    npoly_params = int(root[1][2].attrib['value'])
    nderiv = int(root[1][3].attrib['value'])
    regu_param = float(root[1][4].attrib['value'])
    tol_grad = float(root[1][5].attrib['value'])
    n_guess = int(root[1][6].attrib['value'])

    # Physical Group
    if root[2][0].attrib['value'] == 'None':
        tauhe = None
    else:
        tauhe = float(root[2][0].attrib['value'])
    if root[2][1].attrib['value'] == 'None':
        dtauhe = None
    else:
        dtauhe = float(root[2][1].attrib['value'])
    if root[2][2].attrib['value'] == 'None':
        taucz = None
    else:
        taucz = float(root[2][2].attrib['value'])
    if root[2][3].attrib['value'] == 'None':
        dtaucz = None
    else:
        dtaucz = float(root[2][3].attrib['value'])
    if root[2][4].attrib['value'] == 'None':
        taucz_min = None
    else:
        taucz_min = float(root[2][4].attrib['value'])
    if root[2][5].attrib['value'] == 'None':
        taucz_max = None
    else:
        taucz_max = float(root[2][5].attrib['value'])
    if root[2][6].attrib['value'] == 'None':
        vmin = None
    else:
        vmin = float(root[2][6].attrib['value'])
    if root[2][7].attrib['value'] == 'None':
        vmax = None
    else:
        vmax = float(root[2][7].attrib['value'])


    #========================================================================
    # Code runs automatically from here...
    #========================================================================

    # Loop over stars
    for star in stars:
    
    
        # Start the log file
        outputdir = os.path.join(path, star + "_" + method+'_Fitting_observed_frequencies(6)')
        if not os.path.isdir(outputdir):
            os.makedirs(outputdir)
        stdout = sys.stdout
        sys.stdout = ug.Logger(os.path.join(outputdir, "log.txt"))
    
        # Print header
        print (88 * "=")
        ug.prt_center("FREQUENCY RATIOS AND GLITCH PROPERTIES", 88)
        print ()
        ug.prt_center("The GlitchPy code", 88)
        ug.prt_center("https://github.com/kuldeepv89/GlitchPy", 88)
        print (88 * "=")
    
        # Print local time
        t0 = time.localtime()
        print(
            "\nRun started on {0}.\n\n".format(time.strftime("%Y-%m-%d %H:%M:%S", t0))
        )

        # Print star name/ID
        print ("Analysing star: %s" %(star))
    
    
        # Load observed oscillation frequencies
        freqfile = os.path.join(path, star + '.txt')
        if not os.path.isfile(freqfile):
            raise FileNotFoundError("Input frequency file not found %s!" %(freqfile))
        freq, num_of_mode, num_of_n, delta_nu = ug.loadFreq(freqfile, num_of_l)
        if vmin is None:
            nu_min = 1334.28530945
        else:
            nu_min = vmin
        if vmax is None:
            nu_max = 2994.83963647
        else:
            nu_max = vmax
        print ("\nThe observed data:")
        print ("    - total number of modes: %d" %(num_of_mode))
        print ("    - number of n for each l:", num_of_n)
        print (
            "    - frequency range for averaging: (%.2f, %.2f) muHz" %(nu_min, nu_max)
        )
        print ("    - large separation: %.2f muHz" %(delta_nu))
        print ("    - acoustic radius: %d sec" %(5e5/delta_nu))
    
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
        print ("\nThe fitting method and associated parameters:")
        print ("    - fitting method: %s" %(method))
        print (
            "    - degree of polynomial for smooth component: %d" %(npoly_params - 1)
        )
        print ("    - order of derivative in regularization: %d" %(nderiv))
        print ("    - regularization parameter: %.1f" %(regu_param))
        print ("    - absolute tolerance on gradients: %.1e" %(tol_grad))
        print ("    - number of attempts in global minimum search: %d" %(n_guess))
    
        # Print miscellaneous information    
        print ("\nMiscellaneous information:")
        print ("    - number of realizations: %d" %(n_rln))
        print ("    - tauhe, dtauhe: ({0}, {1})".format(tauhe, dtauhe))
        print ("    - taucz, dtaucz: ({0}, {1})".format(taucz, dtaucz))
        print ("    - ratio type: {0}".format(rtype))
    
    
        # Fit the glitch signatures
        print ("\n* Fitting data... ")
        param, chi2, reg, ier, ratio, dnu = sg.fit(
            freq, 
            num_of_n, 
            delta_nu, 
            num_of_dif2=num_of_dif2, 
            freqDif2=freqDif2, 
            icov=icov, 
            method=method, 
            n_rln=n_rln,
            npoly_params=npoly_params,
            nderiv=nderiv, 
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

        # Print chi-square of the fit (to the observed data) 
        print ("\nThe summary of the fit:")
        if method.lower() == "fq":
            dof = freq.shape[0] - param.shape[1]
        elif method.lower() == "sd":
            dof = freqDif2.shape[0] - param.shape[1]
        if dof <= 0:
            print ("\nWARNING: Degree of freedom %d <= 0! Setting it to 1...\n" %(dof))
            dof = 1
        rchi2 = chi2[-1] / dof
        print ("    - total and reduced chi-squares: (%.4f, %.4f)" %(chi2[-1], rchi2))

        # Produce plots to visualize the fit
        plotdata = {}
        plotdata["method"] = method
        plotdata["npoly_params"] = npoly_params
        plotdata["tauhe"] = tauhe
        plotdata["dtauhe"] = dtauhe
        plotdata["taucz"] = taucz
        plotdata["dtaucz"] = dtaucz
        plotdata["taucz_min"]=taucz_min
        plotdata["taucz_max"]=taucz_max
        plotdata["freq"] = freq
        plotdata["num_of_n"] = num_of_n
        plotdata["delta_nu"] = delta_nu
        plotdata["vmin"] = nu_min
        plotdata["vmax"] = nu_max
        plotdata["freqDif2"] = freqDif2
        plotdata["param"] = param
        plots.fit_summary(plotdata, outputdir)
    
        # Extract successfully fitted realizations
        param_rln, ier_rln = param[0:n_rln, :], ier[0:n_rln]
        param_rln = param_rln[ier_rln == 0, :]
        nfit_rln = param_rln.shape[0]

        # Apply taucz mask if taucz_min != None and/or taucz_max != None
        if taucz_min is not None:
            mask1 = param_rln[:, -6] > taucz_min
        else:
            mask1 = np.ones(nfit_rln, dtype=bool)
        if taucz_max is not None:
            mask2 = param_rln[:, -6] < taucz_max
        else:
            mask2 = np.ones(nfit_rln, dtype=bool)
        mask = np.logical_and(mask1, mask2)
        param_rln = param_rln[mask, :]
        nfit_rln = param_rln.shape[0]

        # Extract ratios and large frequency separation
        if rtype is not None:
            ratio_rln = ratio[0:n_rln, :]
            ratio_rln = ratio_rln[ier_rln == 0, :]
            ratio_rln = ratio_rln[mask, :]
            dnu_rln = dnu[0:n_rln]
            dnu_rln = dnu_rln[ier_rln == 0]
            dnu_rln = dnu_rln[mask]

        # Compute average amplitudes of the He and CZ signatures
        Acz_rln, Ahe_rln = np.zeros(nfit_rln), np.zeros(nfit_rln)
        for j in range(nfit_rln):
            Acz_rln[j], Ahe_rln[j] = sg.averageAmplitudes(
                param_rln[j, :], 
                nu_min, 
                nu_max, 
                delta_nu=delta_nu, 
                method=method
            )

        # Extract realizations with non-zero average He amplitude
        param_rln = param_rln[Ahe_rln>1e-08, :]
        if rtype is not None:
            ratio_rln = ratio_rln[Ahe_rln>1e-08, :]
            dnu_rln = dnu_rln[Ahe_rln>1e-08]
        Acz_rln = Acz_rln[Ahe_rln>1e-08]
        Ahe_rln = Ahe_rln[Ahe_rln>1e-08]
        nfit_rln = param_rln.shape[0]
        if n_rln != nfit_rln:
            print (
                "WARNING: Fits failed (or had zero <Ahe> or inappropriate Tcz)"
                " for realizations: %d/%d" %(n_rln - nfit_rln, n_rln)
            )
    
        # Print median values and associated negative and positive errorbars of the CZ
        #     signature (average amplitude, acoustic depth and phase)
        Acz = {"unit": "muHz"}
        Acz["value"], Acz["nerr"], Acz["perr"] = ug.medianAndErrors(Acz_rln)
        print (
            "    - median Acz, nerr, perr: (%.4f, %.4f, %.4f)" 
            %(Acz["value"], Acz["nerr"], Acz["perr"])
        )

        Tcz = {"unit": "sec"}
        Tcz["value"], Tcz["nerr"], Tcz["perr"] = ug.medianAndErrors(param_rln[:, -6])
        print (
            "    - median Tcz, nerr, perr: (%.1f, %.1f, %.1f)" 
            %(Tcz["value"], Tcz["nerr"], Tcz["perr"])
        )

        Pcz = {"unit": "dimesionless"}
        Pcz["value"], Pcz["nerr"], Pcz["perr"] = ug.medianAndErrors(param_rln[:, -5])
        print (
            "    - median Pcz, nerr, perr: (%.4f, %.4f, %.4f)" 
            %(Pcz["value"], Pcz["nerr"], Pcz["perr"])
        )
    
        # Print median values and associated negative and positive errorbars of the He
        #     signature (average amplitude, acoustic width, acoustic depth and phase)
        Ahe = {"unit": "muHz"}
        Ahe["value"], Ahe["nerr"], Ahe["perr"] = ug.medianAndErrors(Ahe_rln)
        print (
            "    - median Ahe, nerr, perr: (%.4f, %.4f, %.4f)" 
            %(Ahe["value"], Ahe["nerr"], Ahe["perr"])
        )

        Dhe = {"unit": "sec"}
        Dhe["value"], Dhe["nerr"], Dhe["perr"] = ug.medianAndErrors(param_rln[:, -3])
        print (
            "    - median Dhe, nerr, perr: (%.3f, %.3f, %.3f)" 
            %(Dhe["value"], Dhe["nerr"], Dhe["perr"])
        )

        The = {"unit": "sec"}
        The["value"], The["nerr"], The["perr"] = ug.medianAndErrors(param_rln[:, -2])
        print (
            "    - median The, nerr, perr: (%.2f, %.2f, %.2f)" 
            %(The["value"], The["nerr"], The["perr"])
        )

        Phe = {"unit": "dimesionless"}
        Phe["value"], Phe["nerr"], Phe["perr"] = ug.medianAndErrors(param_rln[:, -1])
        print (
            "    - median Phe, nerr, perr: (%.4f, %.4f, %.4f)" 
            %(Phe["value"], Phe["nerr"], Phe["perr"])
        )
        
        # Combine ratios and He glitch properties into a single variable
        grparams = np.zeros((nfit_rln, 3))
        grparams[:, 0] = Ahe_rln[:]
        grparams[:, 1] = param_rln[:, -3]
        grparams[:, 2] = param_rln[:, -2]
        if rtype is not None:
            grparams = np.hstack((ratio_rln, grparams))
            #grparams = np.hstack((dnu_rln.reshape(nfit_rln, 1), grparams))
    
        # Compute the median values
        ngr = grparams.shape[1]
        gr = np.zeros(ngr)
        gr[-3], gr[-2], gr[-1] = Ahe["value"], Dhe["value"], The["value"] 
        if rtype is not None:
            norder, frq, rto = ug.specific_ratio(freq, rtype=rtype)
            for i in range(ngr-3):
                gr[i] = np.median(grparams[:, i])
    
        # Compute the covariance matrix
        j = int(round(nfit_rln / 2))
        covtmp = MinCovDet().fit(grparams[0:j, :]).covariance_
        gr_cov = MinCovDet().fit(grparams).covariance_
    
        # Test convergence (change in standard deviations below a relative 
        #    tolerance)
        rdif = np.amax(
            np.abs(
                np.divide(
                    np.sqrt(np.diag(covtmp)) - np.sqrt(np.diag(gr_cov)), 
                    np.sqrt(np.diag(gr_cov))
                )
            )
        )
        if rdif > 0.1:
            print (
                "WARNING: Maximum relative difference %.2e > 0.1! " 
                "Check the covariance..." %(rdif)
            )
    
        # Print the observables with uncertainties from covariance matrix
        print ("\nThe observables with uncertainties from covariance matrix:")
        for i in range(ngr):
            if i == ngr-3:
                print (
                    "    - median Ahe, err: (%.4f, %.4f)" 
                    %(gr[i], np.sqrt(gr_cov[i, i]))
                )
            elif i == ngr-2:
                print (
                    "    - median Dhe, err: (%.3f, %.3f)" 
                    %(gr[i], np.sqrt(gr_cov[i, i]))
                )
            elif i == ngr-1:
                print (
                    "    - median The, err: (%.2f, %.2f)" 
                    %(gr[i], np.sqrt(gr_cov[i, i]))
                )
            else:
                print (
                    "    - n, freq, median ratio, err: (%d, %.2f, %.5f, %.5f)" 
                    %(int(round(norder[i])), frq[i], gr[i], np.sqrt(gr_cov[i, i]))
                )

        # Plot the correlation matrix
        plots.correlations(gr_cov, outputdir)
    
        # Write the results to HDF5 file
        outfile = os.path.join(outputdir, "results.hdf5")  
        with h5py.File(outfile, "w") as ff:
            ff.create_dataset('header/method', data=method)
            ff.create_dataset('header/npoly_params', data=npoly_params)
            ff.create_dataset('header/nderiv', data=nderiv)
            ff.create_dataset('header/regu_param', data=regu_param)
            ff.create_dataset('header/tol_grad', data=tol_grad)
            ff.create_dataset('header/n_guess', data=n_guess)
            if tauhe is not None:
                ff.create_dataset('header/tauhe', data=tauhe)
            if dtauhe is not None:
                ff.create_dataset('header/dtauhe', data=dtauhe)
            if taucz is not None:
                ff.create_dataset('header/taucz', data=taucz)
            if dtaucz is not None:
                ff.create_dataset('header/dtaucz', data=dtaucz)
            if taucz_min is not None:
                ff.create_dataset('header/taucz_min', data=taucz_min)
            if taucz_max is not None:
                ff.create_dataset('header/taucz_max', data=taucz_max)

            ff.create_dataset('obs/freq', data=freq)
            ff.create_dataset('obs/num_of_n', data=num_of_n)
            ff.create_dataset('obs/delta_nu', data=delta_nu)
            ff.create_dataset('obs/vmin', data=nu_min)
            ff.create_dataset('obs/vmax', data=nu_max)
            if freqDif2 is not None:
                ff.create_dataset('obs/freqDif2', data=freqDif2)
            if icov is not None:
                ff.create_dataset('obs/icov', data=icov)

            ff.create_dataset('fit/param', data=param)
            ff.create_dataset('fit/chi2', data=chi2)
            ff.create_dataset('fit/reg', data=reg)
            ff.create_dataset('fit/ier', data=ier)
            if rtype is not None:
                ff.create_dataset('rto/rtype', data=rtype)
                ff.create_dataset('rto/ratio', data=ratio)
                ff.create_dataset('rto/norder', data=norder)
                ff.create_dataset('rto/frq', data=frq)
                ff.create_dataset('rto/dnu', data=dnu)
    
            ff.create_dataset('cov/params', data=gr)
            ff.create_dataset('cov/cov', data=gr_cov)

        # Print completion time 
        t1 = time.localtime()
        print(
            "\nFinished on {0}".format(time.strftime("%Y-%m-%d %H:%M:%S", t1)),
            "(runtime {0} s).".format(time.mktime(t1) - time.mktime(t0)),
        )

        # Save log file
        sys.stdout = stdout

if __name__ == "__main__":
    main()
