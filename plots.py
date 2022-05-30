import numpy as np
import supportGlitch as sg
import utils_general as ug
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import seaborn as sns



#-----------------------------------------------------------------------------------------
def plot_fitSummary(path):
    '''
    Plots summarizing the fit

    Parameters
    ----------
    path : str
        Complete path of the folder containing the fitted data 

    Return
    ------
    Plots summarizing the fit
    '''
#-----------------------------------------------------------------------------------------

    # Load data
    header, obsData, fitData, rtoData = ug.loadFit(path + "fitData.hdf5")
    (method, npoly_params, nderiv, regu_param, tol_grad, n_guess, tauhe, 
        dtauhe, taucz, dtaucz) = header
    freq, num_of_n, delta_nu, vmin, vmax, freqDif2, icov = obsData
    param, chi2, reg, ier = fitData

    # Number of harmonic degrees and realizations
    num_of_l = len(num_of_n)
    n_rln = len(ier) - 1

    # Glitch parameters for successfully fitted realizations
    param_rln, ier_rln = param[0:n_rln, :], ier[0:n_rln]
    param_rln = param_rln[ier_rln == 0, :]
    nfit_rln = param_rln.shape[0]

    # Average amplitudes
    Ahe_rln = np.zeros(nfit_rln)
    Acz_rln = np.zeros(nfit_rln)
    for i in range(nfit_rln):
        Acz_rln[i], Ahe_rln[i] = sg.averageAmplitudes(
            param_rln[i, :], vmin, vmax, delta_nu=delta_nu, method=method
        )
    Acz_orig, Ahe_orig = sg.averageAmplitudes(
        param[-1, :], vmin, vmax, delta_nu=delta_nu, method=method
    )

    # Initial guesses for various acoutic depths 
    acousticRadius = 5.e5 / delta_nu
    if tauhe is None:
        tauhe = 0.17 * acousticRadius + 18.
    if dtauhe is None:
        dtauhe = 0.05 * acousticRadius
    if taucz is None:
        taucz = 0.34 * acousticRadius + 929.
    if dtaucz is None:
        dtaucz = 0.10 * acousticRadius


    # List of colors
    colorList = ['#D55E00', '#56B4E9', '#000000', 'darkgrey', '#6C9D34', '#482F76']
    markerList = ['o', '*', 'd', '^', 'h']


    # Plot showing the fit to glitch signatures
    #------------------------------------------ 
    fig = plt.figure()
    sns.set(rc={'text.usetex' : True})
    sns.set_style("ticks")
    fig.subplots_adjust(bottom=0.15, right=0.95, top=0.95, left=0.15)

    ax = fig.add_subplot(111)
    ax.set_rasterization_zorder(-1)

    # Fitting oscillation frequencies
    if method.lower() == 'fq':

        xmin, xmax = np.amin(freq[:, 2]), np.amax(freq[:, 2])
        plt.axhline(y=0., ls=":", color="k", lw=1)

        nmode = freq.shape[0]
        obsSignal = np.zeros(nmode)
        for i in range(nmode):
            obsSignal[i] = freq[i, 2] - sg.smoothComponent(
                param[-1, :], 
                l=freq[i, 0].astype(int), 
                n=freq[i, 1].astype(int), 
                num_of_n=num_of_n, 
                npoly_params=npoly_params,
                method=method
            )

        n1 = 0
        for i in range(num_of_l):
            if num_of_n[i] == 0:
                continue
            n2 = n1 + num_of_n[i]
            ax.errorbar(freq[n1:n2, 2], obsSignal[n1:n2], yerr=freq[n1:n2, 3], 
                fmt=markerList[i], ms=5, lw=1, ecolor=colorList[i], mec=colorList[i], 
                mfc='white', label=r'$l = $'+str(i)
            )
            n1 = n2    

        xnu = np.linspace(xmin, xmax, 501)
        modSignal = sg.totalGlitchSignal(xnu, param[-1, :])
        plt.plot(xnu, modSignal, '-', lw=2, color=colorList[-1])

    # Fitting second differences
    elif method.lower() == 'sd':
        if freqDif2 is None:
            raise ValueError("freqDif2 cannot be None for SD!")

        xmin, xmax = np.amin(freqDif2[:, 2]), np.amax(freqDif2[:, 2])
        plt.axhline(y=0., ls=":", color="k", lw=1)

        ndif2 = freqDif2.shape[0]
        obsSignal = np.zeros(ndif2)
        for i in range(ndif2):
            obsSignal[i] = freqDif2[i, 4] - sg.smoothComponent(
                param[-1, :], 
                nu=freqDif2[i, 2],
                npoly_params=npoly_params,
                method=method
            )

        n1 = 0
        for i in range(num_of_l):
            if num_of_n[i] == 0:
                continue
            n2 = n1 + num_of_n[i] - 2
            ax.errorbar(freqDif2[n1:n2, 2], obsSignal[n1:n2], yerr=freqDif2[n1:n2, 5], 
                fmt=markerList[i], ms=5, lw=1, ecolor=colorList[i], mec=colorList[i], 
                mfc='white', label=r'$l = $'+str(i)
            )
            n1 = n2    

        xnu = np.linspace(xmin, xmax, 501)
        modSignal = sg.totalGlitchSignal(xnu, param[-1, :])
        plt.plot(xnu, modSignal, '-', lw=2, color=colorList[-1])

    else:
        raise ValueError("Unrecognized fitting method %s!" %(method))

    ax.legend(loc='upper center', fontsize=18, handlelength=1., handletextpad=0.1, 
        ncol=len(num_of_n[num_of_n>0]), columnspacing=0.5, frameon=False)
    ax.set_xlabel(r'$\nu \ (\mu {\rm Hz})$', fontsize=18, labelpad=3)
    if method.lower() == 'fq':
        ax.set_ylabel(r'$\nu_{\rm glitch} \ (\mu {\rm Hz})$', fontsize=22, labelpad=3)
    elif method.lower() == 'sd':
        ax.set_ylabel(r'$\delta^2\nu_{\rm glitch} \ (\mu {\rm Hz})$', fontsize=22, 
            labelpad=3)
    ax.tick_params(axis='y', labelsize=18, which='both', direction='inout',
        pad=3)
    ax.tick_params(axis='x', labelsize=18, which='both', direction='inout',
        pad=3)

    xmajor, xminor = majMinTick(xmin-20., xmax+20., nxmajor=6, nxminor=5)
    ax.set_xlim(left=xmin-20., right=xmax+20.)
    minLoc = MultipleLocator(xminor)
    ax.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax.xaxis.set_major_locator(majLoc)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    ymax = np.amax(abs(obsSignal)) + 0.1
    ymin = -ymax
    ymajor, yminor = majMinTick(ymin, ymax, nxmajor=7, nxminor=5)
    ax.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(yminor)
    ax.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax.yaxis.set_major_locator(majLoc)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    fig.savefig(path + "fit.png", dpi=400, bbox_inches='tight')
    plt.close(fig)


    # Plot showing distributions of the He glitch parameters
    #------------------------------------------------------- 
    fig = plt.figure()
    sns.set(rc={'text.usetex' : True})
    sns.set_style("ticks")
    fig.subplots_adjust(
        bottom=0.12, right=0.95, top=0.98, left=0.15, wspace=0.25, hspace=0.35
    )

    # Average amplitude 
    ax1 = fig.add_subplot(221)
    ax1.set_rasterization_zorder(-1)
    
    Ahe, AheNErr, AhePErr = ug.medianAndErrors(Ahe_rln)
    xmin = max(0., Ahe - 10. * AheNErr)
    xmax = Ahe + 10. * AhePErr 

    ax1.hist(Ahe_rln, bins=np.linspace(xmin, xmax, 50), color=colorList[0])
    ax1.axvline(x=Ahe_orig, ls="-", color='k', lw=1)

    ax1.set_xlabel(
        r'$\langle A_{\rm He} \rangle \ (\mu {\rm Hz})$', fontsize=11, labelpad=1
    )
    ax1.set_ylabel(r'Frequency', fontsize=11, labelpad=1)
    ax1.tick_params(axis='y', labelsize=9, which='both', direction='inout', pad=1)
    ax1.tick_params(axis='x', labelsize=9, which='both', direction='inout', pad=1)

    xmajor, xminor = majMinTick(xmin, xmax, nxmajor=4, nxminor=5)
    ax1.set_xlim(left=xmin, right=xmax)
    minLoc = MultipleLocator(xminor)
    ax1.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax1.xaxis.set_major_locator(majLoc)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ymin, ymax = ax1.get_ylim()
    ymajor, yminor = majMinTick(ymin, ymax, nxmajor=4, nxminor=5)
    ax1.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(yminor)
    ax1.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax1.yaxis.set_major_locator(majLoc)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    
    # Acoutic width 
    ax2 = fig.add_subplot(222)
    ax2.set_rasterization_zorder(-1)
    
    Dhe, DheNErr, DhePErr = ug.medianAndErrors(param_rln[:, -3])
    xmin = max(0., Dhe - 10. * DheNErr)
    xmax = min(acousticRadius, Dhe + 10. * DhePErr)

    ax2.hist(param_rln[:, -3], bins=np.linspace(xmin, xmax, 50), color=colorList[0])
    ax2.axvline(x=param[-1, -3], ls="-", color='k', lw=1)
    
    ax2.set_xlabel(r'$\Delta_{\rm He}$ ({\rm s})', fontsize=11, labelpad=1)
    ax2.set_ylabel(r'Frequency', fontsize=11, labelpad=1)
    ax2.tick_params(axis='y', labelsize=9, which='both', direction='inout', pad=1)
    ax2.tick_params(axis='x', labelsize=9, which='both', direction='inout', pad=1)

    xmajor, xminor = majMinTick(xmin, xmax, nxmajor=4, nxminor=5)
    ax2.set_xlim(left=xmin, right=xmax)
    minLoc = MultipleLocator(xminor)
    ax2.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax2.xaxis.set_major_locator(majLoc)
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    ymin, ymax = ax2.get_ylim()
    ymajor, yminor = majMinTick(ymin, ymax, nxmajor=4, nxminor=5)
    ax2.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(yminor)
    ax2.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax2.yaxis.set_major_locator(majLoc)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%d'))

    # Acoutic depth 
    ax3 = fig.add_subplot(223)
    ax3.set_rasterization_zorder(-1)
    
    The, TheNErr, ThePErr = ug.medianAndErrors(param_rln[:, -2])
    xmin = max(0., The - 10. * TheNErr)
    xmax = min(acousticRadius, The + 10. * ThePErr)

    ax3.hist(param_rln[:, -2], bins=np.linspace(xmin, xmax, 50), color=colorList[0])
    ymin, ymax = ax3.get_ylim()
    ax3.plot((tauhe - dtauhe, tauhe + dtauhe), (ymax, ymax), 'k-', lw=0.5)
    ax3.axvline(x=param[-1, -2], ls="-", color="k", lw=1)
    
    ax3.set_xlabel(r'$\tau_{\rm He}$ ({\rm s})', fontsize=11, labelpad=1)
    ax3.set_ylabel(r'Frequency', fontsize=11, labelpad=1)
    ax3.tick_params(axis='y', labelsize=9, which='both', direction='inout', pad=1)
    ax3.tick_params(axis='x', labelsize=9, which='both', direction='inout', pad=1)

    xmajor, xminor = majMinTick(xmin, xmax, nxmajor=4, nxminor=5)
    ax3.set_xlim(left=xmin, right=xmax)
    minLoc = MultipleLocator(xminor)
    ax3.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax3.xaxis.set_major_locator(majLoc)
    ax3.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    ymin, ymax = ax3.get_ylim()
    ymajor, yminor = majMinTick(ymin, ymax, nxmajor=4, nxminor=5)
    ax3.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(yminor)
    ax3.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax3.yaxis.set_major_locator(majLoc)
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%d'))

    # Phase 
    ax4 = fig.add_subplot(224)
    ax4.set_rasterization_zorder(-1)
    
    xmin, xmax = 0., 2. * np.pi 

    ax4.hist(param_rln[:, -1], bins=np.linspace(xmin, xmax, 50), color=colorList[0])
    ax4.axvline(x=param[-1, -1], ls="-", color="k", lw=1)
    
    ax4.set_xlabel(r'$\phi_{\rm He}$', fontsize=11, labelpad=1)
    ax4.set_ylabel(r'Frequency', fontsize=11, labelpad=1)
    ax4.tick_params(axis='y', labelsize=9, which='both', direction='inout', pad=1)
    ax4.tick_params(axis='x', labelsize=9, which='both', direction='inout', pad=1)

    xmajor, xminor = majMinTick(xmin, xmax, nxmajor=4, nxminor=5)
    ax4.set_xlim(left=xmin, right=xmax)
    minLoc = MultipleLocator(xminor)
    ax4.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax4.xaxis.set_major_locator(majLoc)
    ax4.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    ymin, ymax = ax4.get_ylim()
    ymajor, yminor = majMinTick(ymin, ymax, nxmajor=4, nxminor=5)
    ax4.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(yminor)
    ax4.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax4.yaxis.set_major_locator(majLoc)
    ax4.yaxis.set_major_formatter(FormatStrFormatter('%d'))

    fig.savefig(path + "heliumDist.png", dpi=400, bbox_inches='tight')
    plt.close(fig)


    # Plot showing distributions of the BCZ glitch parameters
    #-------------------------------------------------------- 
    fig = plt.figure()
    sns.set(rc={'text.usetex' : True})
    sns.set_style("ticks")
    fig.subplots_adjust(
        bottom=0.12, right=0.90, top=0.98, left=0.20, wspace=0.25, hspace=0.35
    )

    # Average amplitude 
    ax1 = fig.add_subplot(311)
    ax1.set_rasterization_zorder(-1)
    
    Acz, AczNErr, AczPErr = ug.medianAndErrors(Acz_rln)
    xmin = max(0., Acz - 10. * AczNErr)
    xmax = Acz + 10. * AczPErr 

    ax1.hist(Acz_rln, bins=np.linspace(xmin, xmax, 100), color=colorList[0], 
        label=r''+str(nfit_rln)+'/'+str(n_rln)
    )
    ax1.axvline(x=Acz_orig, ls="-", color='k', lw=1)

    ax1.legend(loc='upper right', fontsize=9, frameon=False)
    ax1.set_xlabel(
        r'$\langle A_{\rm CZ} \rangle \ (\mu {\rm Hz})$', fontsize=11, labelpad=1
    )
    ax1.set_ylabel(r'Frequency', fontsize=11, labelpad=1)
    ax1.tick_params(axis='y', labelsize=9, which='both', direction='inout', pad=1)
    ax1.tick_params(axis='x', labelsize=9, which='both', direction='inout', pad=1)

    xmajor, xminor = majMinTick(xmin, xmax, nxmajor=7, nxminor=5)
    ax1.set_xlim(left=xmin, right=xmax)
    minLoc = MultipleLocator(xminor)
    ax1.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax1.xaxis.set_major_locator(majLoc)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    ymin, ymax = ax1.get_ylim()
    ymajor, yminor = majMinTick(ymin, ymax, nxmajor=4, nxminor=5)
    ax1.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(yminor)
    ax1.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax1.yaxis.set_major_locator(majLoc)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    
    # Acoutic depth
    ax2 = fig.add_subplot(312)
    ax2.set_rasterization_zorder(-1)
    
    Tcz, TczNErr, TczPErr = ug.medianAndErrors(param_rln[:, -6])
    xmin = max(0., Tcz - 10. * TczNErr)
    xmax = min(acousticRadius, Tcz + 10. * TczPErr)

    ax2.hist(param_rln[:, -6], bins=np.linspace(xmin, xmax, 100), color=colorList[0])
    ymin, ymax = ax2.get_ylim()
    ax2.plot((taucz - dtaucz, taucz + dtaucz), (ymax, ymax), 'k-', lw=0.5)
    ax2.axvline(x=param[-1, -6], ls="-", color="k", lw=1)

    ax2.set_xlabel(r'$\tau_{\rm CZ}$ ({\rm s})', fontsize=11, labelpad=1)
    ax2.set_ylabel(r'Frequency', fontsize=11, labelpad=1)
    ax2.tick_params(axis='y', labelsize=9, which='both', direction='inout', pad=1)
    ax2.tick_params(axis='x', labelsize=9, which='both', direction='inout', pad=1)

    xmajor, xminor = majMinTick(xmin, xmax, nxmajor=6, nxminor=5)
    ax2.set_xlim(left=xmin, right=xmax)
    minLoc = MultipleLocator(xminor)
    ax2.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax2.xaxis.set_major_locator(majLoc)
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    ymin, ymax = ax2.get_ylim()
    ymajor, yminor = majMinTick(ymin, ymax, nxmajor=4, nxminor=5)
    ax2.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(yminor)
    ax2.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax2.yaxis.set_major_locator(majLoc)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    
    # Phase 
    ax3 = fig.add_subplot(313)
    ax3.set_rasterization_zorder(-1)
    
    xmin, xmax = 0., 2. * np.pi 

    ax3.hist(param_rln[:, -5], bins=np.linspace(xmin, xmax, 100), color=colorList[0])
    ax3.axvline(x=param[-1, -5], ls="-", color="k", lw=1)
    
    ax3.set_xlabel(r'$\phi_{\rm CZ}$', fontsize=11, labelpad=1)
    ax3.set_ylabel(r'Frequency', fontsize=11, labelpad=1)
    ax3.tick_params(axis='y', labelsize=9, which='both', direction='inout', pad=1)
    ax3.tick_params(axis='x', labelsize=9, which='both', direction='inout', pad=1)

    xmajor, xminor = majMinTick(xmin, xmax, nxmajor=7, nxminor=5)
    ax3.set_xlim(left=xmin, right=xmax)
    minLoc = MultipleLocator(xminor)
    ax3.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax3.xaxis.set_major_locator(majLoc)
    ax3.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    ymin, ymax = ax3.get_ylim()
    ymajor, yminor = majMinTick(ymin, ymax, nxmajor=4, nxminor=5)
    ax3.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(yminor)
    ax3.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax3.yaxis.set_major_locator(majLoc)
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%d'))

    fig.savefig(path + "convectionDist.png", dpi=400, bbox_inches='tight')
    plt.close(fig)

    return



#-----------------------------------------------------------------------------------------
def plot_correlations(cov, filename="./correlations.png"):
    """
    Plot correlation matrix 

    Parameters
    ----------
    cov : array
        Covariance matrix
    filename : str
        File name (including full path) for the plot

    Return
    ------
    A heatmap plot
    """
#-----------------------------------------------------------------------------------------

    # Compute the correlation matrix
    cor = ug.correlation_from_covariance(cov)
    n = cor.shape[0]
    

    fig = plt.figure()
    sns.set(rc={'text.usetex' : True})
    sns.set_style("ticks")
    fig.subplots_adjust(bottom=0.15, right=0.95, top=0.95, left=0.15)
    ax = fig.add_subplot(111)
    ax.set_rasterization_zorder(-1)
    
    ax = sns.heatmap(
        cor, vmin=-1, vmax=1, linewidths=0, linecolor='white', cmap='RdBu', 
        center=0, square=True, cbar_kws={"label": r"Correlation coefficient"}
    )
    ax.axhline(y=0, color='k',linewidth=2)
    ax.axhline(y=n, color='k',linewidth=2)
    ax.axvline(x=0, color='k',linewidth=2)
    ax.axvline(x=n, color='k',linewidth=2)
    
    plt.xlim(0, n)
    plt.ylim(n, 0)
    
    plt.savefig(filename, dpi=400, bbox_inches='tight')
    plt.close(fig)

    return



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
