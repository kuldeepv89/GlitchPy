"""
Functions for frequency ratios calculations
"""
import numpy as np


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

    names = ["l", "n", "freq", "err"]
    fmts = [int, int, float, float]
    freq = np.zeros(frq.shape[0], dtype={"names": names, "formats": fmts})
    freq[:]["l"] = np.rint(frq[:, 0]).astype(int)
    freq[:]["n"] = np.rint(frq[:, 1]).astype(int)
    freq[:]["freq"] = frq[:, 2]
    freq[:]["err"] = frq[:, 3]

    # Isolate l = 0 modes
    f0 = freq[freq[:]["l"] == 0]
    if (len(f0) == 0) or (len(f0) != f0[-1]["n"] - f0[0]["n"] + 1):
        # Missing modes detected (not implemented)!
        r02, r01, r10 = None, None, None
        return r02, r01, r10

    # Isolate l = 1 modes
    f1 = freq[freq[:]["l"] == 1]
    if (len(f1) == 0) or (len(f1) != f1[-1]["n"] - f1[0]["n"] + 1):
        # Missing modes detected (not implemented)!
        r02, r01, r10 = None, None, None
        return r02, r01, r10

    # Isolate l = 2 modes
    f2 = freq[freq[:]["l"] == 2]
    if (len(f2) == 0) or (len(f2) != f2[-1]["n"] - f2[0]["n"] + 1):
        # Missing modes detected (not implemented)!
        r02, r01, r10 = None, None, None
        return r02, r01, r10

    # Two-point frequency ratio
    # ---------------------------
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

    return r02, r01, r10


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

    # Number of ratios
    n02 = r02.shape[0]
    n01 = r01.shape[0]
    n10 = r10.shape[0]
    n010 = n01 + n10
    n012 = n01 + n02
    n102 = n10 + n02

    # R010 (R01 followed by R10)
    r010 = np.zeros((n010, 4))
    r010[0:n01, :] = r01[:, :]
    r010[n01 : n01 + n10, 0] = r10[:, 0] + 0.1
    r010[n01 : n01 + n10, 1:4] = r10[:, 1:4]
    r010 = r010[r010[:, 0].argsort()]
    r010[:, 0] = np.round(r010[:, 0])

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

    return r010, r012, r102


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
    ratio : array
        Ratio values
    """
    
    # Compute ratios
    r02, r01, r10 = ratios(frq)
    if r02 is None:
        raise ValueError("Error: Missing radial orders!")

    # Compute combined ratios (if necessary)
    if rtype in ["r010", "r012", "r102"]:
        r010, r012, r102 = combined_ratios(r02, r01, r10)

    # Return the ratio type of "rtype"
    if rtype == "r01":
        ratio = np.zeros(r01.shape[0])
        ratio = r01[:, 1]
    elif rtype == "r10":
        ratio = np.zeros(r10.shape[0])
        ratio = r10[:, 1]
    elif rtype == "r02":
        ratio = np.zeros(r02.shape[0])
        ratio = r02[:, 1]
    elif rtype == "r010":
        ratio = np.zeros(r010.shape[0])
        ratio = r010[:, 1]
    elif rtype == "r012":
        ratio = np.zeros(r012.shape[0])
        ratio = r012[:, 1]
    elif rtype == "r102":
        ratio = np.zeros(r102.shape[0])
        ratio = r102[:, 1]
    else:
        raise ValueError("Unrecognized ratio-type %s!" %(rtype))
    
    return ratio
