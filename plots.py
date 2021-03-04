import sys
import numpy as np
import supportGlitch as sg
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import seaborn as sns



#-----------------------------------------------------------------------------------------
def majMinTick(xmin, xmax, nxmajor=7, nxminor=5):
    '''
    Calculate step sizes for major and minor tick levels

    Parameters
    ----------
    xmin : float
        Minimum value of x
    xmax : float
        Maximum value of x
    nxmajor : int 
        Typical number of required major ticks on the x-axis
    nxminor : int 
        Number of required minor ticks between two consecutive major ticks on the x-axis

    Return
    ------
    xmajor : float
        Step size for major ticks on the x-axis
    xminor : float
        Step size for minor ticks on the x-axis
    '''
#-----------------------------------------------------------------------------------------

    xmajor = float("{:.0e}".format((xmax - xmin) / nxmajor))
    xminor = xmajor / nxminor

    return xmajor, xminor



#-----------------------------------------------------------------------------------------
def fitSummary(num_of_l, freq, num_of_n, delta_nu, param, param_rln, n_rln, method='FQ', 
        freqDif2=None, tauhe=None, dtauhe=None, taucz=None, dtaucz=None, 
        filename='./fit.png'):
    '''
    Plot summarizing the quality of fit

    Parameters
    ----------
    num_of_l : int
        Number of harmonic degrees (starting from l = 0)
    freq : array
        Observed modes (l, n, v(muHz), err(muHz)) 
    num_of_n : array of int
        Number of modes for each l
    delta_nu : float
        An estimate of large frequncy separation (muHz)
    param : array
        Fitted parameters 
    param_rln : array
        Parameters for successfully fitted realizations
    n_rln : int
        Number of realizations. If n_rln = 0, just fit the original frequencies/differences
    method : str
        Fitting method ('FQ' or 'SD')
    freqDif2 : array
        Second differences (l, n, v(muHz), err(muHz), dif2(muHz), err(muHz))
    tauhe : float, optional
        Determines the range in acoustic depth (s) of He glitch for global minimum 
        search (tauhe - dtauhe, tauhe + dtauhe)
    dtauhe : float, optional
        Determines the range in acoustic depth (s) of He glitch for global minimum 
        search (tauhe - dtauhe, tauhe + dtauhe)
    taucz : float, optional
        Determines the range in acoustic depth (s) of CZ glitch for global minimum 
        search (taucz - dtaucz, taucz + dtaucz)
    dtaucz : float, optional
        Determines the range in acoustic depth (s) of CZ glitch for global minimum 
        search (taucz - dtaucz, taucz + dtaucz)
    filename: str
        Complete path to the output file containing the plot

    Return
    ------
    A plot summarizing the quality of fit
    '''
#-----------------------------------------------------------------------------------------

    # Initialize acoutic depths (if they are None)
    acousticRadius = 5.e5 / delta_nu
    if tauhe is None:
        tauhe = 0.15 * acousticRadius
    if dtauhe is None:
        dtauhe = 0.10 * acousticRadius
    if taucz is None:
        taucz = 0.50 * acousticRadius
    if dtaucz is None:
        dtaucz = 0.20 * acousticRadius

    # List of colors
    colorList = ['#D55E00', '#56B4E9', '#000000', 'darkgrey', '#6C9D34', '#482F76']
    markerList = ['o', '*', 'd', '^', 'h']


    fig = plt.figure()
    sns.set(rc={'text.usetex' : True})
    sns.set_style("ticks")
    fig.subplots_adjust(bottom=0.12, right=0.95, top=0.98, left=0.15, wspace=0.25, 
        hspace=0.35)
    

    # Fit to glitch signature
    #------------------------ 
    ax1 = fig.add_subplot(321)
    ax1.set_rasterization_zorder(-1)

    # Oscillation frequency fit
    if method.lower() == 'fq':

        xmin, xmax = np.amin(freq[:, 2]), np.amax(freq[:, 2])
        plt.plot((xmin - 20., xmax + 20.), (0.0, 0.0), 'k:', lw=0.5)

        obsSignal = freq[:, 2] - sg.smoothComponent(param, 
            l=freq[:, 0].astype(int), n=freq[:, 1].astype(int), num_of_l=num_of_l, 
            method=method)

        n1 = 0
        for i in range(num_of_l):
            n2 = n1 + num_of_n[i]
            ax1.errorbar(freq[n1:n2, 2], obsSignal[n1:n2], yerr=freq[n1:n2, 3], 
                fmt=markerList[i], ms=3, lw=1, ecolor=colorList[i], mec=colorList[i], 
                mfc='white', label=r'$l = $'+str(i))
            n1 = n2    

        xnu = np.linspace(xmin, xmax, 501)
        modSignal = sg.totalGlitchSignal(xnu, param)
        plt.plot(xnu, modSignal, '-', lw=1, color=colorList[-1])

    # Second difference fit
    elif method.lower() == 'sd':
        if freqDif2 is None:
            print ('freqDif2 cannot be None. Terminating the run...')
            sys.exit(1)

        xmin, xmax = np.amin(freqDif2[:, 2]), np.amax(freqDif2[:, 2])
        plt.plot((xmin - 20., xmax + 20.), (0.0, 0.0), 'k:', lw=0.5)

        obsSignal = freqDif2[:, 4] - sg.smoothComponent(param, nu=freqDif2[:, 2],
            method=method)

        n1 = 0
        for i in range(num_of_l):
            n2 = n1 + num_of_n[i] - 2
            ax1.errorbar(freqDif2[n1:n2, 2], obsSignal[n1:n2], yerr=freqDif2[n1:n2, 5], 
                fmt=markerList[i], ms=3, lw=1, ecolor=colorList[i], mec=colorList[i], 
                mfc='white', label=r'$l = $'+str(i))
            n1 = n2    

        xnu = np.linspace(xmin, xmax, 501)
        modSignal = sg.totalGlitchSignal(xnu, param)
        plt.plot(xnu, modSignal, '-', lw=1, color=colorList[-1])

    else:
        print ('Fitting method is not recognized. Terminating the run...')
        sys.exit(2)

    ax1.legend(loc='upper center', fontsize=9, handlelength=1., handletextpad=0.1, 
        ncol=num_of_l, columnspacing=0.5, frameon=False)
    ax1.set_xlabel(r'$\nu \ (\mu {\rm Hz})$', fontsize=11, labelpad=1)
    if method.lower() == 'fq':
        ax1.set_ylabel(r'$\nu_{\rm glitch} \ (\mu {\rm Hz})$', fontsize=11, labelpad=1)
    elif method.lower() == 'sd':
        ax1.set_ylabel(r'$\delta^2\nu_{\rm glitch} \ (\mu {\rm Hz})$', fontsize=11, 
            labelpad=1)
    ax1.tick_params(axis='y', labelsize=9, which='both', direction='inout',
        pad=1)
    ax1.tick_params(axis='x', labelsize=9, which='both', direction='inout',
        pad=1)

    xmajor, xminor = majMinTick(xmin-20., xmax+20., nxmajor=4, nxminor=5)
    ax1.set_xlim(left=xmin-20., right=xmax+20.)
    minLoc = MultipleLocator(xminor)
    ax1.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax1.xaxis.set_major_locator(majLoc)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    ymax = np.amax(abs(obsSignal)) + 0.1
    ymin = -ymax
    ymajor, yminor = majMinTick(ymin, ymax, nxmajor=4, nxminor=5)
    ax1.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(yminor)
    ax1.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax1.yaxis.set_major_locator(majLoc)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


    # Acoutic depth of the He signature    
    #----------------------------------
    ax2 = fig.add_subplot(322)
    ax2.set_rasterization_zorder(-1)
    
    The, TheNErr, ThePErr = sg.medianAndErrors(param_rln[:, -2])
    xmin, xmax = max(0., The - 10. * TheNErr), min(acousticRadius, The + 10. * ThePErr)

    ax2.hist(param_rln[:, -2], bins=np.linspace(xmin, xmax, 50), color=colorList[0])

    ymin, ymax = ax2.get_ylim()
    ax2.plot((tauhe - dtauhe, tauhe + dtauhe), (ymax, ymax), 'k-', lw=0.5)

    ymin, ymax = ax2.get_ylim()
    ax2.plot((param[-2], param[-2]), (ymin, ymax), 'k-', lw=1)
    
    ax2.set_xlabel(r'$\tau_{\rm He}$ (s)', fontsize=11, labelpad=1)
    ax2.set_ylabel(r'Frequency', fontsize=11, labelpad=1)
    ax2.tick_params(axis='y', labelsize=9, which='both', direction='inout',
        pad=1)
    ax2.tick_params(axis='x', labelsize=9, which='both', direction='inout',
        pad=1)

    xmajor, xminor = majMinTick(xmin, xmax, nxmajor=4, nxminor=5)
    ax2.set_xlim(left=xmin, right=xmax)
    minLoc = MultipleLocator(xminor)
    ax2.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax2.xaxis.set_major_locator(majLoc)
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    ymajor, yminor = majMinTick(ymin, ymax, nxmajor=4, nxminor=5)
    ax2.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(yminor)
    ax2.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax2.yaxis.set_major_locator(majLoc)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%d'))

    
    # Average amplitude of the He signature    
    #--------------------------------------
    ax3 = fig.add_subplot(323)
    ax3.set_rasterization_zorder(-1)
    
    nfit_rln = len(param_rln[:, 0])
    Ahe_rln = np.zeros(nfit_rln)
    vmin, vmax = np.amin(freq[:, 2]), np.amax(freq[:, 2])
    for i in range(nfit_rln):
        _, Ahe_rln[i] = sg.averageAmplitudes(param_rln[i, :], vmin, vmax, 
            delta_nu=delta_nu, method=method)
    _, Ahe_orig = sg.averageAmplitudes(param, vmin, vmax, delta_nu=delta_nu, 
        method=method)

    Ahe, AheNErr, AhePErr = sg.medianAndErrors(Ahe_rln)
    xmin, xmax = max(0., Ahe - 10. * AheNErr), Ahe + 10. * AhePErr 

    ax3.hist(Ahe_rln, bins=np.linspace(xmin, xmax, 50), color=colorList[0], 
        label=r''+str(nfit_rln)+'/'+str(n_rln))

    ymin, ymax = ax3.get_ylim()
    ax3.plot((Ahe_orig, Ahe_orig), (ymin, ymax), 'k-', lw=1)

    ax3.legend(loc='upper right', fontsize=9, frameon=False)
    ax3.set_xlabel(r'$\langle A_{\rm He} \rangle \ (\mu {\rm Hz})$', fontsize=11, 
        labelpad=1)
    ax3.set_ylabel(r'Frequency', fontsize=11, labelpad=1)
    ax3.tick_params(axis='y', labelsize=9, which='both', direction='inout',
        pad=1)
    ax3.tick_params(axis='x', labelsize=9, which='both', direction='inout',
        pad=1)

    xmajor, xminor = majMinTick(xmin, xmax, nxmajor=4, nxminor=5)
    ax3.set_xlim(left=xmin, right=xmax)
    minLoc = MultipleLocator(xminor)
    ax3.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax3.xaxis.set_major_locator(majLoc)
    ax3.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ymajor, yminor = majMinTick(ymin, ymax, nxmajor=4, nxminor=5)
    ax3.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(yminor)
    ax3.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax3.yaxis.set_major_locator(majLoc)
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    

    # Acoutic depth of the CZ signature    
    #----------------------------------
    ax4 = fig.add_subplot(324)
    ax4.set_rasterization_zorder(-1)
    
    Tcz, TczNErr, TczPErr = sg.medianAndErrors(param_rln[:, -6])
    xmin, xmax = max(0., Tcz - 10. * TczNErr), min(acousticRadius, Tcz + 10. * TczPErr)

    ax4.hist(param_rln[:, -6], bins=np.linspace(xmin, xmax, 50), color=colorList[0])

    ymin, ymax = ax4.get_ylim()
    ax4.plot((taucz - dtaucz, taucz + dtaucz), (ymax, ymax), 'k-', lw=0.5)

    ymin, ymax = ax4.get_ylim()
    ax4.plot((param[-6], param[-6]), (ymin, ymax), 'k-', lw=1)

    ax4.set_xlabel(r'$\tau_{\rm CZ}$ (s)', fontsize=11, labelpad=1)
    ax4.set_ylabel(r'Frequency', fontsize=11, labelpad=1)
    ax4.tick_params(axis='y', labelsize=9, which='both', direction='inout',
        pad=1)
    ax4.tick_params(axis='x', labelsize=9, which='both', direction='inout',
        pad=1)

    xmajor, xminor = majMinTick(xmin, xmax, nxmajor=4, nxminor=5)
    ax4.set_xlim(left=xmin, right=xmax)
    minLoc = MultipleLocator(xminor)
    ax4.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax4.xaxis.set_major_locator(majLoc)
    ax4.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    ymajor, yminor = majMinTick(ymin, ymax, nxmajor=4, nxminor=5)
    ax4.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(yminor)
    ax4.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax4.yaxis.set_major_locator(majLoc)
    ax4.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    

    fig.savefig(filename, dpi=400, bbox_inches='tight')
    plt.close(fig)

    return
