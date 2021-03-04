import numpy as np
import h5py



#-----------------------------------------------------------------------------------------
def loadFreq(filename, num_of_l):
    '''
    Load the observed oscillation frequencies

    Parameters
    ----------
    filename : str
        Complete path to the file containing observed modes
    num_of_l : int
        Number of harmonic degrees (starting from l = 0)

    Return
    ------
    freq : array
        Observed modes (l, n, v(muHz), err(muHz)) 
    num_of_mode : int
        Number of modes
    num_of_n : array of int
        Number of modes for each l
    delta_nu : float
        Large frequncy separation (muHz)
    '''
#-----------------------------------------------------------------------------------------

    # Read oscillation frequencies
    freq = np.genfromtxt(filename, dtype=float, comments='#')
    freq = freq[freq[:, 0]<(num_of_l - 0.5), :]
    num_of_mode = freq.shape[0]

    # Fetch number of modes for each l
    num_of_n = np.zeros(num_of_l, dtype=int)
    for i in range (num_of_l):
        num_of_n[i] = len(freq[np.rint(freq[:, 0]) == i, 0])
    
    # Estimate large separation using linear fit to the radial modes
    coefs = np.polyfit(freq[freq[:, 0]<0.5, 1], freq[freq[:, 0]<0.5, 2], 1)
    delta_nu = coefs[0]
    
    return (freq, num_of_mode, num_of_n, delta_nu)



#-----------------------------------------------------------------------------------------
def loadFit(filename):
    '''
    Load the fit related data

    Parameters
    ----------
    filename : str
        Complete path to the file containing fit related data 
    
    Return
    ------
    header : tuple
        Parameters determining the fit (method, regu_param, tol_grad, tauhe, dtauhe,
        taucz, dtaucz)
    obsData : tuple
        Observed data (num_of_l, freq, num_of_n, delta_nu, freqDif2, icov)
    fitData : tuple
        Fit related data (param, chi2, reg, ier)
    '''
#-----------------------------------------------------------------------------------------

    # Read the hdf5 file
    with h5py.File(filename, 'r') as data:
        method = data['header/method'][()]
        regu_param = data['header/regu_param'][()]
        tol_grad = data['header/tol_grad'][()]
        tauhe = data['header/tauhe'][()]
        dtauhe = data['header/dtauhe'][()]
        taucz = data['header/taucz'][()]
        dtaucz = data['header/dtaucz'][()]
        num_of_l = data['obs/num_of_l'][()]
        freq = data['obs/freq'][()]
        num_of_n = data['obs/num_of_n'][()]
        delta_nu = data['obs/delta_nu'][()]
        freqDif2 = data['obs/freqDif2'][()]
        icov = data['obs/icov'][()]
        param = data['fit/param'][()]
        chi2 = data['fit/chi2'][()]
        reg = data['fit/reg'][()]
        ier = data['fit/ier'][()]
    
    # Group the data
    header  = (method, regu_param, tol_grad, tauhe, dtauhe, taucz, dtaucz)
    obsData = (num_of_l, freq, num_of_n, delta_nu, freqDif2, icov)
    fitData = (param, chi2, reg, ier)
    
    return header, obsData, fitData
