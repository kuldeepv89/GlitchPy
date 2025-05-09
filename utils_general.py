"""
Useful functions of miscellaneous type (in particular related to ratios, print log, 
load data, etc.)
"""
import sys
import numpy as np
import h5py
from scipy.interpolate import CubicSpline
import xml.etree.ElementTree as ET
tree=ET.parse('stars.xml')
root=tree.getroot()



#-----------------------------------------------------------------------------------------
def read_xml():
    """
    Read input data
    """
#-----------------------------------------------------------------------------------------

    # Frequency group
    for j in root.iter('path'):
        path = j.attrib['value']

    for j in root.iter('num_of_l'):
        num_of_l = int(j.attrib['value'])

    for j in root.iter('type'):
        if j.attrib['value'] == 'None':
            rtype = None
            epstype = None
        if j.attrib['value'] in ['e01', 'e02', 'e012']:
            epstype = j.attrib['value']
            rtype = None
        elif j.attrib['value'] in ['r010', 'r02', 'r01', 'r10', 'r012', 'r102']:
            rtype = j.attrib['value']
            epstype = None

        else:
            raise ValueError("ERROR: Invalid ratio/epsilon type!")
            
    for j in root.iter('include_dnu'):
        if j.attrib['value'] == 'True':
            include_dnu = True
        else:
            include_dnu = False

    # Numerical parameters group
    for j in root.iter('method'):
        method = j.attrib['value']

    for j in root.iter('n_rln'):
        n_rln = int(j.attrib['value'])

    for j in root.iter('npoly_params'):
        npoly_params = int(j.attrib['value'])

    for j in root.iter('nderiv'):
        nderiv = int(j.attrib['value'])

    for j in root.iter('regu_param'):
        regu_param = float(j.attrib['value'])

    for j in root.iter('tol_grad'):
        tol_grad = float(j.attrib['value'])

    for j in root.iter('n_guess'):
        n_guess = int(j.attrib['value'])

    # Physical parameters group
    stars = []
    for star in root.iter('star'):
        stars.append(star.attrib['starid'])

    delta_nu = []
    for j in root.iter('delta_nu'):
        if j.attrib['value'] == 'None':
            delta_nu.append(None)
        else:
            delta_nu.append(float(j.attrib['value']))

    nu_max = []
    for j in root.iter('nu_max'):
        if j.attrib['value'] == 'None':
            nu_max.append(None)
        else:
            nu_max.append(float(j.attrib['value']))

    tauhe = []
    for j in root.iter('tauhe'):
        if j.attrib['value'] == 'None':
            tauhe.append(None)
        else:
            tauhe.append(float(j.attrib['value']))

    dtauhe = []
    for j in root.iter('dtauhe'):
        if j.attrib['value'] == 'None':
            dtauhe.append(None)
        else:
            dtauhe.append(float(j.attrib['value']))

    taucz = []
    for j in root.iter('taucz'):
        if j.attrib['value'] == 'None':
            taucz.append(None)
        else:
            taucz.append(float(j.attrib['value']))

    dtaucz = []
    for j in root.iter('dtaucz'):
        if j.attrib['value'] == 'None':
            dtaucz.append(None)
        else:
            dtaucz.append(float(j.attrib['value']))

    taucz_min = []
    for j in root.iter('taucz_min'):
        if j.attrib['value'] == 'None':
            taucz_min.append(None)
        else:
            taucz_min.append(float(j.attrib['value']))

    taucz_max = []
    for j in root.iter('taucz_max'):
        if j.attrib['value'] == 'None':
            taucz_max.append(None)
        else:
            taucz_max.append(float(j.attrib['value']))

    vmin = []
    for j in root.iter('vmin'):
        if j.attrib['value'] == 'None':
            vmin.append(None)
        else:
            vmin.append(float(j.attrib['value']))

    vmax = []
    for j in root.iter('vmax'):
        if j.attrib['value'] == 'None':
            vmax.append(None)
        else:
            vmax.append(float(j.attrib['value']))

    return (
            path, num_of_l, rtype, epstype, include_dnu,
            method, n_rln, npoly_params, nderiv, regu_param, tol_grad, n_guess, 
            stars, delta_nu, nu_max, tauhe, dtauhe, taucz, dtaucz, 
            taucz_min, taucz_max, vmin, vmax
    )



#-----------------------------------------------------------------------------------------
def ratios(frq):
    """
    Routine to compute the ratios (r02, r01 and r10) from oscillation
    frequencies

    Parameters
    ----------
    frq : array
        Harmonic degrees, radial orders, frequencies

    Returns
    -------
    r02 : array
        radial orders, r02 ratios,
        scratch for uncertainties (to be calculated), frequencies
    r01 : array
        radial orders, r01 ratios,
        scratch for uncertainties (to be calculated), frequencies
    r10 : array
        radial orders, r10 ratios,
        scratch for uncertainties (to be calculated), frequencies
    """
#-----------------------------------------------------------------------------------------

    names = ["l", "n", "freq", "err"]
    fmts = [int, int, float, float]
    freq = np.zeros(frq.shape[0], dtype={"names": names, "formats": fmts})
    freq[:]["l"] = np.rint(frq[:, 0]).astype(int)
    freq[:]["n"] = np.rint(frq[:, 1]).astype(int)
    freq[:]["freq"] = frq[:, 2]
    freq[:]["err"] = frq[:, 3]

    # Isolate l = 0 modes
    f0 = freq[freq[:]["l"] == 0]
    if len(f0) == 0:
        raise ValueError("ERROR: Radial modes not found!")
    else:
        if len(f0) != f0[-1]["n"] - f0[0]["n"] + 1:
            # Missing radial order (not implemented)!
            r02, r01, r10 = None, None, None
            return r02, r01, r10

    # Isolate l = 1 modes
    f1 = freq[freq[:]["l"] == 1]
    if (len(f1) == 0):
        raise ValueError("ERROR: Dipole modes not found!")
    else:
        if len(f1) != f1[-1]["n"] - f1[0]["n"] + 1:
            # Missing radial order (not implemented)!
            r02, r01, r10 = None, None, None
            return r02, r01, r10

    # Five-point frequency ratio (R01)
    # ---------------------------------
    # Find lowest indices for l = 0, 1, and 2
    if f0[0]["n"] >= f1[0]["n"]:
        i00 = 0
        i01 = f0[0]["n"] - f1[0]["n"]
    else:
        i00 = f1[0]["n"] - f0[0]["n"]
        i01 = 0

    # Number of r01s
    if f0[-1]["n"] - 1 >= f1[-1]["n"]:
        nr01 = f1[-1]["n"] - f1[i01]["n"]
    else:
        nr01 = f0[-1]["n"] - f0[i00]["n"] - 1

    # R01
    r01 = np.zeros((nr01, 4))
    for i in range(nr01):
        r01[i, 0] = f0[i00 + i + 1]["n"]
        r01[i, 3] = f0[i00 + i + 1]["freq"]
        r01[i, 1] = (
            f0[i00 + i]["freq"]
            + 6.0 * f0[i00 + i + 1]["freq"]
            + f0[i00 + i + 2]["freq"]
        )
        r01[i, 1] -= 4.0 * (f1[i01 + i + 1]["freq"] + f1[i01 + i]["freq"])
        r01[i, 1] /= 8.0 * (f1[i01 + i + 1]["freq"] - f1[i01 + i]["freq"])

    # Five-point frequency ratio (R10)
    # ---------------------------------
    # Find lowest indices for l = 0, 1, and 2
    if f0[0]["n"] - 1 >= f1[0]["n"]:
        i00 = 0
        i01 = f0[0]["n"] - f1[0]["n"] - 1
    else:
        i00 = f1[0]["n"] - f0[0]["n"] + 1
        i01 = 0

    # Number of r10s
    if f0[-1]["n"] >= f1[-1]["n"]:
        nr10 = f1[-1]["n"] - f1[i01]["n"] - 1
    else:
        nr10 = f0[-1]["n"] - f0[i00]["n"]

    # R10
    r10 = np.zeros((nr10, 4))
    for i in range(nr10):
        r10[i, 0] = f1[i01 + i + 1]["n"]
        r10[i, 3] = f1[i01 + i + 1]["freq"]
        r10[i, 1] = (
            f1[i01 + i]["freq"]
            + 6.0 * f1[i01 + i + 1]["freq"]
            + f1[i01 + i + 2]["freq"]
        )
        r10[i, 1] -= 4.0 * (f0[i00 + i + 1]["freq"] + f0[i00 + i]["freq"])
        r10[i, 1] /= -8.0 * (f0[i00 + i + 1]["freq"] - f0[i00 + i]["freq"])

    # Isolate l = 2 modes (if available) and compute r02
    f2 = freq[freq[:]["l"] == 2]
    if (len(f2) != 0) and (len(f2) == f2[-1]["n"] - f2[0]["n"] + 1):

        # Two-point frequency ratio (R02)
        # ---------------------------------
        n0 = (f0[0]["n"] - 1, f1[0]["n"], f2[0]["n"])
        l0 = n0.index(max(n0))

        # Find lowest indices for l = 0, 1, and 2
        if l0 == 0:
            i00 = 0
            i01 = f0[0]["n"] - f1[0]["n"] - 1
            i02 = f0[0]["n"] - f2[0]["n"] - 1
        elif l0 == 1:
            i00 = f1[0]["n"] - f0[0]["n"] + 1
            i01 = 0
            i02 = f1[0]["n"] - f2[0]["n"]
        elif l0 == 2:
            i00 = f2[0]["n"] - f0[0]["n"] + 1
            i01 = f2[0]["n"] - f1[0]["n"]
            i02 = 0

        # Number of r02s
        nn = (f0[-1]["n"], f1[-1]["n"], f2[-1]["n"] + 1)
        ln = nn.index(min(nn))
        if ln == 0:
            nr02 = f0[-1]["n"] - f0[i00]["n"] + 1
        elif ln == 1:
            nr02 = f1[-1]["n"] - f1[i01]["n"]
        elif ln == 2:
            nr02 = f2[-1]["n"] - f2[i02]["n"] + 1

        # R02
        r02 = np.zeros((nr02, 4))
        for i in range(nr02):
            r02[i, 0] = f0[i00 + i]["n"]
            r02[i, 3] = f0[i00 + i]["freq"]
            r02[i, 1] = f0[i00 + i]["freq"] - f2[i02 + i]["freq"]
            r02[i, 1] /= f1[i01 + i + 1]["freq"] - f1[i01 + i]["freq"]

    else:
        # Quadrupole modes unavailable or missing radial order!
        r02 = None

    return r02, r01, r10


#-----------------------------------------------------------------------------------------
def compute_epsilondiff(
    osckey,
    osc,
    avgdnu,
    sequence="e012",
    nsorting=True,
    extrapolation=False,
    nrealisations=20000,
    debug=False,
):
    """
    Compute epsilon differences and covariances.

    From Roxburgh 2016:
    * Eq. 1: Epsilon(n,l)
    * Eq. 4: EpsilonDifference(l=0,l=(1,2))

    Epsilon differences are independent of surface phase shift/outer
    layers when the epsilons are evaluated at the same frequency. It
    therefore relies on splining from epsilons at the observed frequencies
    of the given degree and order to the frequency of the compared/subtracted
    epsilon. See function "compute_epsilondiffseqs" for further clarification.

    For MonteCarlo sampling of the covariances, it is replicated from the
    covariance determination of frequency ratios in BASTA, (sec 4.1.3 of
    Aguirre Børsen-Koch et al. 2022). A number of realisations of the
    epsilon differences are drawn from random Gaussian distributions of the
    individual frequencies within their uncertainty.

    Parameters
    ----------
    osckey : array
        Array containing the angular degrees and radial orders of the modes.
    osc : array
        Array containing the modes (and inertias).
    avgdnu : float
        Average value of the large frequency separation.
    sequence : str, optional
        Similar to ratios, what sequence of epsilon differences to be computed.
        Can be e01, e02 or e012 for a combination of the two first.
    nsorting : bool, optional
        If True (default), the sequences are sorted by n-value of the frequencies. If
        False, the entire 01 sequence is followed by the 02 sequence.
    extrapolation : bool, optional
        If False (default), modes outside the range of the l=0 modes are discarded to
        avoid extrapolation.
    nrealisations : int or bool, optional
        If int: number of realisations used for MC-sampling the covariances
        If bool: Whether to use MC (True) or analytic (False) deternubation
        of covariances. If True, nrealisations of 20,000 is used.
    debug : bool, optional
        Print additional output and make plots for debugging (incl. a plot of the
        correlation matrix).

    Returns
    -------
    epsdiff : array
        Array containing the modes in the observed data.
    epsdiff_cov : array
        Covariances matrix.
    """
#-----------------------------------------------------------------------------------------
    # Remove modes outside of l=0 range
    if not extrapolation:
        indall = osckey[0, :] > -1
        ind0 = osckey[0, :] == 0
        ind12 = osckey[0, :] > 0
        # print(osc[0, ind12], max(osc[0, ind0]))
        mask = np.logical_and(
            osc[0, ind12] < max(osc[0, ind0]), osc[0, ind12] > min(osc[0, ind0])
        )
        indall[ind12] = mask
        if debug and any(mask):
            print(
                "The following modes have been skipped from epsilon differences to avoid extrapolation:"
            )
            for f, (l, n) in zip(osc[0, ~indall], osckey[:, ~indall].T):
                print(" - (l,n,f) = ({0}, {1:02d}, {2:.3f})".format(l, n, f))
        # print(osc[1])
        # print(osckey[1])
        osc = osc[:, indall]
        osckey = osckey[:, indall]
        # print(osc[1])
        # print(osckey[1])
    epsdiff = compute_epsilondiffseqs(
        osckey, osc, avgdnu, sequence=sequence, nsorting=nsorting
    )

    return epsdiff 




#-----------------------------------------------------------------------------------------
def compute_epsilondiffseqs(
    osckey,
    osc,
    avgdnu,
    sequence,
    nsorting=True,
):
    """
    Computed epsilon differences, based on Roxburgh 2016 (eq. 1 and 4)

    Epsilons E of frequency v with order n and degree l is determined as:
    E(n,l) = E(v(n,l)) = v(n,l)/dnu - n - l/2

    From this, an epsilon is determined for each original frequncy. These
    are not independent on the surface layers, but their differences
    between different degrees are, if evaluated at the same frequency.
    Therefore, the epsilon differences dE of e.g. E(n,l=0) and E(n,l=2),
    dE(02) is determined from interpolating/splining the l=0 epsilon sequence
    SE0 and evaluating it at v(n,l=2), and subtracting the corresponding
    E(n,l=2). Therefore, the epsilon difference can be summarised as
    dE(0l) = SE0(v(n,l)) - E(n,l)

    Parameters
    ----------
    osckey : array
        Array containing the angular degrees and radial orders of the modes
    osc : array
        Array containing the modes (and inertias)
    avgdnu : float
        Average large frequency separation
    sequence : str
        Similar to ratios, what sequence of epsilon differences to be computed.
        Can be 01, 02 or 012 for a combination of the two first.
    nsorting : bool
        If True (default), the sequences are sorted by n-value of the frequencies.
        If False, the entire 01 sequence is followed by the 02 sequence.

    Returns
    -------
    deps : array
        Array containing epsilon differences. First index correpsonds to:
        0 - Epsilon differences
        1 - Indentifying frequencies
        2 - Identifying degree l
        3 - Radial degree n of identifying l={1,2} mode
    """
#-----------------------------------------------------------------------------------------

    # Select the sequence(s) to use
    if sequence == "e012":
        l_used = [1, 2]
    elif sequence == "e02":
        l_used = [2]
    elif sequence == "e01":
        l_used = [1]
    else:
        raise KeyError("Undefined epsilon difference sequence requested!")

    # Epsilon is computed analytically from the frequency information
    epsilon = np.zeros(osc.shape[1])

    for i, freq in enumerate(osc[0, :]):
        ll, nn = osckey[:, i]
        epsilon[i] = freq / avgdnu - nn - ll / 2
    # Setup base l=0 interpolater object
    nu0 = osc[0, osckey[0, :] == 0]
    eps0 = epsilon[osckey[0, :] == 0]
    eps0_intpol = CubicSpline(nu0, eps0)
    # Compute the epsilon differences of the selected sequence(s)
    nmodes = sum([sum(osckey[0] == ll) for ll in l_used])
    # print(nmodes) 
    deps = np.zeros((4, nmodes))
    Niter = 0
    for ll in l_used:
        # Extract freq and epsilon for l=ll modes
        nul = osc[0, osckey[0] == ll]
        epsl = epsilon[osckey[0] == ll]

        # Evaluate epsilon(l=0) at nu(l=ll)
        eps0_at_nul = eps0_intpol(nul)

        # Difference
        diff_eps0l = eps0_at_nul - epsl

        # Store 0: difference, 1: freq, 2: l, 3: n
        deps[0, Niter : Niter + len(diff_eps0l)] = diff_eps0l
        deps[1, Niter : Niter + len(diff_eps0l)] = nul
        deps[2, Niter : Niter + len(diff_eps0l)] = ll
        deps[3, Niter : Niter + len(diff_eps0l)] = osckey[1][osckey[0] == ll]

        Niter += len(diff_eps0l)
    # print(nul)
    # Sort according to n if flagged (ensure l=1 before l=2 with 0.1)
    if nsorting:
        mask = np.argsort(deps[3, :] + deps[2, :] * 0.1)
        deps = deps[:, mask]
    # print(deps)
    return deps


#-----------------------------------------------------------------------------------------
def combined_ratios(r02, r01, r10):
    """
    Routine to combine r02, r01 and r10 ratios to produce ordered ratios r010,
    r012 and r102

    Parameters
    ----------
    r02 : array
        radial orders, r02 ratios,
        scratch for uncertainties (to be calculated), frequencies
    r01 : array
        radial orders, r01 ratios,
        scratch for uncertainties (to be calculated), frequencies
    r10 : array
        radial orders, r10 ratios,
        scratch for uncertainties (to be calculated), frequencies

    Returns
    -------
    r010 : array
        radial orders, r010 ratios,
        scratch for uncertainties (to be calculated), frequencies
    r012 : array
        radial orders, r012 ratios,
        scratch for uncertainties (to be calculated), frequencies
    r102 : array
        radial orders, r102 ratios,
        scratch for uncertainties (to be calculated), frequencies
    """
#-----------------------------------------------------------------------------------------

    # R010 (R01 followed by R10)
    n01 = r01.shape[0]
    n10 = r10.shape[0]
    n010 = n01 + n10
    r010 = np.zeros((n010, 4))
    r010[0:n01, :] = r01[:, :]
    r010[n01 : n01 + n10, 0] = r10[:, 0] + 0.1
    r010[n01 : n01 + n10, 1:4] = r10[:, 1:4]
    r010 = r010[r010[:, 0].argsort()]
    r010[:, 0] = np.round(r010[:, 0])

    if r02 is not None:
        n02 = r02.shape[0]
        n012 = n01 + n02
        n102 = n10 + n02

        # R012 (R01 followed by R02)
        r012 = np.zeros((n012, 4))
        r012[0:n01, :] = r01[:, :]
        r012[n01 : n01 + n02, 0] = r02[:, 0] + 0.1
        r012[n01 : n01 + n02, 1:4] = r02[:, 1:4]
        r012 = r012[r012[:, 0].argsort()]
        r012[:, 0] = np.round(r012[:, 0])

        # R102 (R10 followed by R02)
        r102 = np.zeros((n102, 4))
        r102[0:n10, :] = r10[:, :]
        r102[n10 : n10 + n02, 0] = r02[:, 0] + 0.1
        r102[n10 : n10 + n02, 1:4] = r02[:, 1:4]
        r102 = r102[r102[:, 0].argsort()]
        r102[:, 0] = np.round(r102[:, 0])
    else:
        r012, r102 = (None, None)

    return r010, r012, r102



#-----------------------------------------------------------------------------------------
def specific_ratio(frq, rtype="r012"):
    """
    Routine to compute specific type of ratios from oscillation
    frequencies

    Parameters
    ----------
    frq : array
        Harmonic degrees, radial orders, frequencies
    rtype : str 
        Ratio type (one of ["r01", "r10", "r02", "r010", "r012", "r102"])

    Returns
    -------
    norder : array
        Radial order values
    frequency : array
        Frequency values (in muHz)
    ratio : array
        Ratio values
    """
#-----------------------------------------------------------------------------------------
    
    if rtype not in ["r02", "r01", "r10", "r010", "r012", "r102"]:
        raise ValueError("ERROR: Unrecognized ratio-type %s!" %(rtype))

    # Compute ratios
    r02, r01, r10 = ratios(frq)

    # Compute combined ratios (if necessary)
    if rtype in ["r010", "r012", "r102"]:
        r010, r012, r102 = combined_ratios(r02, r01, r10)

    # Return the ratio type of "rtype"
    if rtype == "r01":
        norder = r01[:, 0]
        frequency = r01[:, 3]
        ratio = r01[:, 1]
    elif rtype == "r10":
        norder = r10[:, 0]
        frequency = r10[:, 3]
        ratio = r10[:, 1]
    elif rtype == "r010":
        norder = r010[:, 0]
        frequency = r010[:, 3]
        ratio = r010[:, 1]
    elif rtype == "r02":
        if r02 is not None:
            norder = r02[:, 0]
            frequency = r02[:, 3]
            ratio = r02[:, 1]
        else:
            norder, frequency, ratio = (None, None, None)
    elif rtype == "r012":
        if r02 is not None:
            norder = r012[:, 0]
            frequency = r012[:, 3]
            ratio = r012[:, 1]
        else:
            norder, frequency, ratio = (None, None, None)
    elif rtype == "r102":
        if r02 is not None:
            norder = r102[:, 0]
            frequency = r102[:, 3]
            ratio = r102[:, 1]
        else:
            norder, frequency, ratio = (None, None, None)
    return (norder, frequency, ratio)


#-----------------------------------------------------------------------------------------
def specific_eps(frq, dnu, epstype="e012"):
    """
    Routine to compute specific type of epsilon differences from oscillation
    frequencies

    Parameters
    ----------
    frq : array
        Harmonic degrees, radial orders, frequencies
    dnu : float
        Average large frequency separation
    epstype : str 
        epsilon diff type (one of ["e01", "e02", "e012"])

    Returns
    -------
    norder : array
        Radial order values
    loder : array
        Spherical order values (eg. '1' for 'e01')
    frequency : array
        Frequency values (in muHz)
    ratio : array
        Ratio values
    """
#-----------------------------------------------------------------------------------------
    
    if epstype not in ["e02", "e01", "e012"]:
        raise ValueError("ERROR: Unrecognized epsilon-type %s!" %(epstype))
    # Read frequencies from file
    frecu = frq[:,2]
    errors = frq[:,3]
    norder = frq[:,1]
    ldegree = frq[:,0]
    # Build osc and osckey in a sorted manner
    f = np.asarray([])
    n = np.asarray([])
    e = np.asarray([])
    l = np.asarray([])
    for li in [0, 1, 2]:
        given_l = ldegree == li
        incrn = np.argsort(norder[given_l], kind="mergesort")
        l = np.concatenate([l, ldegree[given_l][incrn]])
        n = np.concatenate([n, norder[given_l][incrn]])
        f = np.concatenate([f, frecu[given_l][incrn]])
        e = np.concatenate([e, errors[given_l][incrn]])
    assert len(f) == len(n) == len(e) == len(l)
    osckey = np.asarray([l, n], dtype=int)
    osc = np.array([f, e])
    epsdiff = compute_epsilondiff(osckey, osc, dnu, sequence = epstype)
    eps = epsdiff[0]
    frequency = epsdiff[1]
    lorder = epsdiff[2]
    norder = epsdiff[3]

    return (norder, lorder, frequency, eps)


#-----------------------------------------------------------------------------------------
def dnu0(frq, nu_max=None, weight="none"):
#-----------------------------------------------------------------------------------------
    """
    Routine to compute the large frequency separation using radial modes
 
    Parameters
    ----------
    frq : array
        Harmonic degrees, radial orders, frequencies, errorbar
    nu_max : float
        Frequency of maximum power 
        Used only if weight = "white" (see below)
    weight : str
        Weight in the linear least-squares fitting
        If weight = "white", apply weight following White et al. (2011)
        If weight = "sigma" (default), weight with the uncertainties on the frequencies 
        If weight = "none", no weight 

    Returns
    -------
    dnu : float
        Large frequency separation
    """

    yfitdnu = frq[np.rint(frq[:, 0]).astype(int) == 0, 2]
    xfitdnu = frq[np.rint(frq[:, 0]).astype(int) == 0, 1]

    if weight.lower() == "white":
        FWHM_sigma = 2.0 * np.sqrt(2.0 * np.log(2.0))
        wfitdnu = np.exp(
            -1.0
            * np.power(yfitdnu - nu_max, 2)
            / (2 * np.power(0.25 * nu_max / FWHM_sigma, 2.0))
        )
        fitcoef = np.polyfit(xfitdnu, yfitdnu, 1, w=np.sqrt(wfitdnu))
    elif weight.lower() == "sigma":
        wfitdnu = frq[np.rint(frq[:, 0]).astype(int) == 0, 3]
        fitcoef = np.polyfit(xfitdnu, yfitdnu, 1, w=1./wfitdnu)
    elif weight.lower() == "none":
        fitcoef = np.polyfit(xfitdnu, yfitdnu, 1)
    else:
        raise ValueError("Unrecognized weight %s!" %(weight))

    dnu = fitcoef[0]

    return dnu


#-----------------------------------------------------------------------------------------
class Logger(object):
    """
    Class used to redefine stdout to terminal and an output file.

    Parameters
    ----------
    outfilename : str
        Absolute path to an output file
    """
#-----------------------------------------------------------------------------------------

    # Credit: http://stackoverflow.com/a/14906787
    def __init__(self, outfilename):
        self.terminal = sys.stdout
        self.log = open(outfilename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass



#-----------------------------------------------------------------------------------------
def prt_center(text, llen):
    """
    Prints a centered line

    Parameters
    ----------
    text : str
        The text string to print

    llen : int
        Length of the line

    Returns
    -------
    None
    """
#-----------------------------------------------------------------------------------------

    print("{0}{1}{0}".format(int((llen - len(text)) / 2) * " ", text))



#-----------------------------------------------------------------------------------------
def medianAndErrors(param_rln):
    '''
    Compute the median and (negative and positive) uncertainties from the realizations

    Parameters
    ----------
    param_rln : array
        Parameter values for different realizations

    Return
    ------
    med : float
        Median value
    nerr : float
        Negative error 
    perr : float
        Positive error
    '''
#-----------------------------------------------------------------------------------------

    per16 = np.percentile(param_rln, 16)
    per50 = np.percentile(param_rln, 50)
    per84 = np.percentile(param_rln, 84)
    med, nerr, perr = per50, per50 - per16, per84 - per50

    return med, nerr, perr



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
    
    return (freq, num_of_mode, num_of_n)



#-----------------------------------------------------------------------------------------
def correlation_from_covariance(cov):
    """
    Compute correlation matrix from covariance matrix

    Parameters
    ----------
    cov : array
        Covariance matrix

    Return
    ------
    cor : array
        Correlation matrix
    """
#-----------------------------------------------------------------------------------------

    v = np.sqrt(np.diag(cov))
    outer_v = np.outer(v, v)
    cor = cov / outer_v
    cor[cov == 0] = 0

    return cor



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
