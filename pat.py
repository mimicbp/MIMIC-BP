#!/usr/bin/env python
# coding: utf-8

#
# MIMIC-BP - (C) 2024
# https://doi.org/10.7910/DVN/DBM1NF
#
# Open Data Commons Open Database License (ODbL) v1.0
# https://opendatacommons.org/licenses/odbl/1-0/
#

"""
Module defining functions dedicated to feature extraction from combined ECG and PPG signals

"""

import numpy as np
from scipy.signal import butter, filtfilt

verbose = False

order = 4  # 4 (original value; Butterworth approximation)
fc1ecg = 0.8  # 0.8 (original value)
fc2ecg = 35  # 35 (original value)
fc1ppg = 0.4  # 0.8 (original value)
fc2ppg = 8  # 35 (original value)
NFeat = 10  # number of features plus 2


def fund_freq_assert(e, p, fs):
    assert isinstance(
        e, (list, tuple, np.ndarray)
    ), f'Input "e" must be of type "list", "tuple" or "numpy.ndarray", got {type(e)}'
    assert isinstance(
        p, (list, tuple, np.ndarray)
    ), f'Input "p" must be of type "list", "tuple" or "numpy.ndarray", got {type(p)}'
    assert isinstance(
        fs, int
    ), f'Input "fs" must be of type "int", got {type(fs)}'
    assert fs > 0, f'Input "fs" must be > 0, got fs = {fs}'


def fundamental_freq_corr(
    e: "list, tuple, np.ndarray", p: "list, tuple, np.ndarray", fs: int
) -> "float or np.nan":
    """
    Fundamental frequency estimation based on the correlation between ECG and PPG signals

    Parameters
    ----------
    e : list, tuple, np.ndarray
        The ECG signal

    p : list, tuple, np.ndarray
        The PPG signal

    fs : int
        The signal sampling frequency


    Returns
    -------
    f0 : one of np.nan, float
        The estimated fundamental frequency of the signals.
        Returns np.nan if unable to compute f0


    Raises
    ------
    AssertionError
        If one or more inputs are not given correctly

    """
    fund_freq_assert(e=e, p=p, fs=fs)

    corr = np.convolve(e, p[::-1])
    N = 4 * int(2 ** np.ceil(np.log2(len(corr))))
    Corr = np.abs(np.fft.rfft(corr, N))
    f = np.array(range(N // 2 + 1)) / N * fs  # half since using rfft
    find = f < 5
    Nc = np.argmax(Corr[find])
    f0 = Nc / N * fs
    if f0 < 0.2:  # too low heart rate
        # Critical problem computing fundamental frequency
        return np.nan
    return f0


def fund_freq_returns(E, find, Np, e, p, fs, fe, fp, verbose):
    if np.abs(fe - fp) > 0.3:
        if verbose:
            print("Need to refine search for f0")
            print(f"fe = {fe:.2f}, fp = {fp:.2f}")
        if np.abs(fe / 2 - fp) < 0.3:
            f0 = (fe / 2 + fp) / 2
            return True, f0
        elif np.abs(fe / 3 - fp) < 0.3:
            f0 = (fe / 3 + fp) / 2
            return True, f0
        elif np.abs(fe / 4 - fp) < 0.3:
            f0 = (fe / 4 + fp) / 2
            return True, f0
        else:  # check if ECG is above threshold at max(PPG)
            maxE = np.max(E[find])
            if E[Np] > maxE * 0.85:
                return True, fp
            else:  # correlation between ECG & PPG
                return True, fundamental_freq_corr(e, p, fs)
    return False, None


def fundamental_freq(
    e: "list, tuple or np.ndarray", p: "list, tuple or np.ndarray", fs: int
) -> "float or np.nan":
    """
    Fundamental frequency estimation based on the DFT of ecg and ppg signals

    Parameters
    ----------
    e : list, tuple, np.ndarray
        The ECG signal

    p : list, tuple, np.ndarray
        The PPG signal

    fs : int
        The signal sampling frequency


    Returns
    -------
    f0 : one of np.nan, float
        The estimated fundamental frequency of the signals.
        Returns np.nan if unable to compute f0


    Raises
    ------
    AssertionError
        If one or more inputs are not given correctly

    """
    fund_freq_assert(e=e, p=p, fs=fs)

    N = 8 * int(2 ** np.ceil(np.log2(len(e))))
    f = np.array(range(N // 2 + 1)) / N * fs  # half since using rfft
    find = f < 5
    hw = np.hamming(len(e))
    E = np.abs(np.fft.rfft(e * hw, N))  # magnitude ECG
    P = np.abs(np.fft.rfft(p * hw, N))  # magnitude PPG
    Ne = np.argmax(E[find])
    Np = np.argmax(P[find])
    if Ne == 0 or Np == 0:
        # Critical problem computing fundamental frequency
        return np.nan
    fe = Ne / N * fs
    fp = Np / N * fs
    check, f0_or_p = fund_freq_returns(
        E=E, find=find, Np=Np, e=e, p=p, fs=fs, fe=fe, fp=fp, verbose=verbose
    )
    if check:
        return f0_or_p

    f0 = (fe + fp) / 2
    return f0


def feat_est_assert(ecg, ppg, fs, t0):
    assert isinstance(
        ecg, (list, tuple, np.ndarray)
    ), f'Input "ecg" must be of type "list", "tuple" or "numpy.ndarray", got {type(ecg)}'
    assert isinstance(
        ppg, (list, tuple, np.ndarray)
    ), f'Input "ppg" must be of type "list", "tuple" or "numpy.ndarray", got {type(ppg)}'
    assert isinstance(
        fs, int
    ), f'Input "fs" must be of type "int", got {type(fs)}'
    assert fs > 0, f'Input "fs" must be > 0, got fs = {fs}'
    assert isinstance(
        t0, (int, float)
    ), f'Input "t0" must be of type "int" or "float", got {type(t0)}'
    assert t0 >= 0, f'Input "t0" must be >= 0, got t0 = {t0}'
    L = len(ecg)
    assert L == len(ppg), "ECG and PPG must have same length"
    T = L / fs
    assert T > 9, "ECG and PPG too short for analysis"
    assert T < 31, "ECG and PPG too long for analysis"
    assert fs > 90, "Sampling frequency too low"


def check_superpositions(pat, feat):
    for i in range(len(pat) - 1):  # check sobreposition
        if feat[i, 0] + feat[i, 1] >= feat[i + 1, 0]:
            feat[i, -1] = 0
            feat[i + 1, -1] = 0
    return feat


def check_epmax(epmax, dsp, ppg):
    if epmax - dsp < 0:
        pmin = epmax - np.argmin(ppg[0:epmax])
    else:
        pmin = dsp - np.argmin(ppg[epmax - dsp : epmax])
    return pmin


def feat_alterations(ppg, ecg, x1, x2, NFeat, sp, sp2, spp, dsp, feat, fs):
    while x2 <= len(ecg):
        f = np.zeros((1, NFeat))
        em = np.argmax(ecg[x1:x2])
        em += x1
        if em + sp <= len(ppg):
            # search for max ppg
            pmax = np.argmax(ppg[em : em + sp])
            if pmax < 3:  # too few samples
                x1 = em + sp2
                x2 = x1 + spp
                continue
            epmax = em + pmax
            # search for min ppg before epmax
            pmin = check_epmax(epmax=epmax, dsp=dsp, ppg=ppg)
            if pmin < 2:  # too few samples
                x1 = em + sp2
                x2 = x1 + spp
                continue
            # current period estimation
            pper = pmin + np.argmin(ppg[epmax : min(epmax + dsp, len(ppg))])
            dpdt = np.diff(ppg[epmax - pmin : epmax])
            maxd = np.argmax(dpdt) + 1
            maxa = max(dpdt)
            incl = (ppg[epmax] - ppg[epmax - pmin]) / pmin
            p1 = ppg[epmax - pmin]  # left side min ppg
            p2 = ppg[min(epmax - pmin + pper, len(ppg))]  # right side min ppg
            n = np.arange(pper + 1)
            ppgbase = p1 + n * (p2 - p1) / pper  # ppg values
            ampl = ppg[epmax] - ppgbase[pmin]
            intg = np.cumsum(
                ppg[epmax - pmin : epmax - pmin + pper + 1] - ppgbase
            )
            f[0, 0] = em / fs
            f[0, 1] = pmax / fs
            f[0, 2] = pmin / fs
            f[0, 3] = pper / fs
            f[0, 4] = maxd / fs
            f[0, 5] = ampl
            f[0, 6] = maxa * fs
            f[0, 7] = incl * fs
            f[0, 8] = intg[-1] / fs
            feat = np.vstack((feat, f))
        x1 = em + sp2
        x2 = x1 + spp
    return feat


def feat_interval_check(feat, bstd, pstd, pat, pmean, pmed):
    if bstd < 0.12 and pstd < 0.1 and len(pat) > 2:
        feat[abs(feat[:, 1] - pmean) <= 3 * pstd, -1] = 1
        feat[feat[:, 1] <= pmed / 2, -1] = 0
        feat[feat[:, 1] >= 2 * pmed, -1] = 0
        feat[feat[:, 2] <= 0, -1] = 0
        feat[feat[:, 4] <= 0, -1] = 0
        feat[feat[:, 4] >= feat[:, 2], -1] = 0
        feat[feat[:, 5] <= 0, -1] = 0
        feat = check_superpositions(pat=pat, feat=feat)
    return feat


def feat_estimation(
    ecg: "list, tuple or np.ndarray",
    ppg: "list, tuple or np.ndarray",
    fs: int,
    t0: "float >= 0" = 0,
) -> np.ndarray:
    """
    Features estimation
    Expects 10 to 30 s of ECG and PPG signals sampled at fs Hz (fs >= 100Hz).
    Returns a matrix of features as described below.
    Last column of this matrix is a 0 or 1 flag indicating the confidence
    in the estimated feature values: value 1 indicates higher confidence
    and value 0 indicates lower confidence.
    Assumes signals already preprocessed by a bandpass Butterworth filter.
    If t0 is given, the R wave time instants are shifited by that value.
    Matrix columns:
    column 0: time instant of the ECG peak wave R
    column 1: time duration of corresponding PAT
    column 2: time between maximum of PPG and previous minimum
    column 3: period of PPG (equal to the period of ECG)
    column 4: time between minimum of PPG and max of d(PPG)/dt
    column 5: max(PPG) - min(PPG)
    column 6: max(d(PPG)/dt)
    column 7: (column 5)/(column 2)
    column 8: area under (PPG - min(PPG))
    column 9: reserved to be used as confidence flag
    Notes:
    1. PAT subjected to normalization by subjects height (approx: k * PWV)
    2. (column 8)/((column 3)x(column 5)) is inversely correlated to
       ventricular stroke volume
    3. See figure of the "BP Project" MChat on 17Feb2022, 10h26 AM


    Parameters
    ----------
    ecg : list, tuple, np.ndarray
        The ECG signal

    ppg : list, tuple, np.ndarray
        The PPG signal

    fs : int
        The signal sampling frequency

    t0 : float >= 0  = 0
        Value corresponding to R-wave shifts. Default = 0.


    Returns
    -------
    feat : np.ndarray
        The estimated features of the signal, as described above


    Raises
    ------
    AssertionError
        If the ecg and ppg variables are not of the same lengh, if the signals are too long or too short for analysis or if the sampling frequency is too low

    """
    feat_est_assert(ecg, ppg, fs, t0)

    feat = np.empty((0, NFeat))

    f0 = fundamental_freq(ecg, ppg, fs)
    if np.isnan(f0):
        return f0
    sp = int(fs / f0)  # approx. number of samples per period
    spp = (11 * sp) // 10  # 10% more samples than sp
    sp2 = sp // 2  # 50% less samples than sp
    dsp = (9 * sp) // 10  # lookup range 10% smaller than sp
    x1, x2 = 0, spp
    feat = feat_alterations(
        ppg=ppg,
        x1=x1,
        x2=x2,
        ecg=ecg,
        NFeat=NFeat,
        sp=sp,
        sp2=sp2,
        spp=spp,
        dsp=dsp,
        feat=feat,
        fs=fs,
    )
    if feat.size == 0:
        return np.zeros((1, NFeat))
    feat[:, 0] += t0
    if feat.shape[0] == 1:
        feat[0, -1] = 0
        return feat
    # confidence treatment
    pat = feat[:, 1]
    pmed = np.median(pat)
    pmean = np.mean(pat)
    pstd = np.std(pat, ddof=1)
    bat = feat[:, 3]
    bstd = np.std(bat, ddof=1)
    feat = feat_interval_check(
        feat=feat, bstd=bstd, pstd=pstd, pat=pat, pmean=pmean, pmed=pmed
    )
    # avoiding meaningless negative values
    feat[feat[:, 5] <= 0, 5] = 0
    feat[feat[:, 6] <= 0, 6] = 0
    feat[feat[:, 7] <= 0, 7] = 0
    feat[feat[:, 8] <= 0, 8] = 0

    return feat


def butter_filter(
    waveform: "list, tuple or np.ndarray",
    fc1: "float > 0",
    fc2: "float > 0",
    fs: "int > 0",
    order: "int > 0",
) -> np.ndarray:
    """
    Filter waveform through bandpass Butterworth filter

    Parameters
    ----------
    waveform : list, tuple, np.ndarray
        ...

    fc1 : float > 0
        Lowcut limit for the band-pass filter

    fc2 : float > fc1
        Highcut limit for the band-pass filter

    fs : int > 0
        The signal sampling frequency

    order : int > 0
        The filter order


    Returns
    -------
    filtfilt : np.ndarray
        The filtered signal
    """
    assert isinstance(
        fc1, (int, float)
    ), f'Input "fc1" must be of type "int" or "float", got {type(fc1)}'
    assert fc1 > 0, f'Input "fc1" must be > 0, got fc1 = {fc1}'
    assert isinstance(
        fc2, (int, float)
    ), f'Input "fc2" must be of type "int" or "float", got {type(fc2)}'
    assert (
        fc2 > fc1
    ), f'Input "fc2" must be > fc1, got fc2 = {fc2} and fc1 = {fc1}'
    assert isinstance(
        fs, int
    ), f'Input "fs" must be of type "int", got {type(fs)}'
    assert fs > 0, f'Input "fs" must be > 0, got fs = {fs}'
    assert isinstance(
        order, int
    ), f'Input "order" must be of type "int", got {type(order)}'
    assert order > 0, f'Input "order" must be > 0, got order = {order}'
    nyq = fs / 2
    b, a = butter(order, [fc1 / nyq, fc2 / nyq], btype="band")
    return filtfilt(b, a, waveform)


def feat_R_assert(ecg, ppg, fc1e, fc2e, fc1p, fc2p, fs, order, T, t0, scan):
    assert isinstance(
        ecg, (list, tuple, np.ndarray)
    ), f'Input "ecg" must be of type "list", "tuple" or "numpy.ndarray", got {type(ecg)}'
    assert isinstance(
        ppg, (list, tuple, np.ndarray)
    ), f'Input "ppg" must be of type "list", "tuple" or "numpy.ndarray", got {type(ppg)}'

    assert isinstance(
        fc1e, (int, float)
    ), f'Input "fc1e" must be of type "int" or "float", got {type(fc1e)}'
    assert fc1e > 0, f'Input "fc1e" must be > 0, got fc1e = {fc1e}'
    assert isinstance(
        fc2e, (int, float)
    ), f'Input "fc2e" must be of type "int" or "float", got {type(fc2e)}'
    assert (
        fc2e > fc1e
    ), f'Input "fc2e" must be > fc1e, got fc2e = {fc2e} and fc1e = {fc1e}'

    assert isinstance(
        fc1p, (int, float)
    ), f'Input "fc1p" must be of type "int" or "float", got {type(fc1p)}'
    assert fc1p > 0, f'Input "fc1p" must be > 0, got fc1p = {fc1p}'
    assert isinstance(
        fc2p, (int, float)
    ), f'Input "fc2p" must be of type "int" or "float", got {type(fc2p)}'
    assert (
        fc2p > fc1p
    ), f'Input "fc2p" must be > fc1p, got fc2p = {fc2p} and fc1p = {fc1p}'

    assert isinstance(
        fs, int
    ), f'Input "fs" must be of type "int", got {type(fs)}'
    assert fs > 0, f'Input "fs" must be > 0, got fs = {fs}'
    assert isinstance(
        order, int
    ), f'Input "order" must be of type "int", got {type(order)}'
    assert order > 0, f'Input "order" must be > 0, got order = {order}'

    assert isinstance(
        T, (int, float)
    ), f'Input "T" must be of type "int" or "float", got {type(T)}'
    assert T >= 0, f'Input "T" must be >= 0, got T = {T}'
    assert isinstance(
        t0, (int, float)
    ), f'Input "t0" must be of type "int" or "float", got {type(t0)}'
    assert t0 >= 0, f'Input "t0" must be >= 0, got t0 = {t0}'

    assert isinstance(
        scan, bool
    ), f'Input "scan" should be of type "bool", got {type(scan)}'


def FEAT_R(
    ecg: "list, tuple, np.ndarray",
    ppg: "list, tuple, np.ndarray",
    fc1e: "float > 0",
    fc2e: "float > 0",
    fc1p: "float > 0",
    fc2p: "float > 0",
    fs: "int > 0",
    order: "int > 0",
    T: "float >= 0" = 10,
    t0: "float >= 0" = 0,
    scan: bool = False,
) -> np.ndarray:
    """
    Receives samples of ecg and ppg signals sampled at fs Hz.
    A band-pass Butterworth filter will be applited to both
    signals. The corresponding cut-off frequencies are fc1e and fc2e
    for the ecg signal and fc1p and fc2p for the ppg signal.
    The argument order receives the desired filter order.
    A sequence of arrays is returned, where each array is of the form
    (ti, feati, flagi) where ti is the time instant of the ecg R wave
    in seconds, feati represents a sequence of the extracted features
    and flagi indicates the confidence in the estimates (1: higher
    confidence, 0: lower confidence).
    Argument T defines the duration of each analysis window.
    If t0 is given, the R wave time instants are shifited by that value.
    scan is not yet implemented (the input signals are analysed with
    overlapping windows in order to increase probability of results
    with higher confidence)

    Parameters
    ----------
    ecg : list, tuple, np.ndarray
        The ECG signal

    ppg : list, tuple, np.ndarray
        The PPG signal

    fc1e : float > 0
        The lowcut limit for the ECG bandpass filter

    fc2e : float > fc1e
        The highcut limit for the ECG bandpass filter

    fc1p : float > 0
        The lowcut limit for the PPG bandpass filter

    fc2p : float > fc1p
        The highcut limit for the PPG bandpass filter

    fs : int > 0
        The signals' sampling frequency

    order : int > 0
        The Butterworth filter's order

    T : float >= 0 = 10
        The analysis window lengh in seconds. Default = 10.

    t0 : float >= 0 = 0
        Value to shift the detected R waves. Default = 0.

    scan : bool = False
        Describe describe


    Returns
    -------
    feat : np.ndarray
        The signal features, as described in function feat_estimation()


    Raises
    ------
    AssertionError
        If the ECG and PPG signals have different lenghs


    Example:
    >>> import pandas as pd
    >>> fname = 'pat_test.csv'  # W4VDN0_*_116_75_94.csv
    >>> invppg = True
    >>> fs = 100
    >>> df = pd.read_csv(fname)
    >>> ecg = np.array(df['ECG'])
    >>> if invppg:
    ...     ppg = -np.array(df['PPG'])
    ... else:
    ...     ppg = np.array(df['PPG'])
    >>> features = FEAT_R(ecg, ppg, fc1ecg, fc2ecg, fc1ppg, fc2ppg,
    ...                   fs, order)
    >>> print(np.around(features[8:11, :], decimals=2))
    [[1.336000e+01 3.800000e-01 1.700000e-01 6.400000e-01 7.000000e-02
      3.801340e+03 3.898344e+04 2.114457e+04 1.620100e+03 1.000000e+00]
     [1.401000e+01 4.000000e-01 2.000000e-01 6.600000e-01 8.000000e-02
      4.906610e+03 4.940348e+04 2.461290e+04 2.097480e+03 1.000000e+00]
     [1.466000e+01 3.900000e-01 1.800000e-01 6.600000e-01 8.000000e-02
      4.934920e+03 4.845974e+04 2.645452e+04 2.243350e+03 1.000000e+00]]
    """
    feat_R_assert(
        ecg=ecg,
        ppg=ppg,
        fc1e=fc1e,
        fc2e=fc2e,
        fc1p=fc1p,
        fc2p=fc2p,
        fs=fs,
        order=order,
        T=T,
        t0=t0,
        scan=scan,
    )
    scan = False
    L = len(ecg)
    assert L == len(ppg), "ECG and PPG must have same length"
    if scan:
        D = 1  # D seconds of window shift
    else:
        D = T
    # prevent NaN input values affecting the whole processing
    ecg[np.isnan(ecg)] = 0
    ppg[np.isnan(ppg)] = 0
    # bandpass filtering
    ecg_ = butter_filter(ecg, fc1e, fc2e, fs, order)
    ppg_ = butter_filter(ppg, fc1p, fc2p, fs, order)
    # sample index
    n = np.array(range(0, L))
    # time index
    t = n / fs
    nT = int(T * fs)  # samples in T seconds
    nD = int(D * fs)  # samples in D seconds
    ni, nj = 0, nT
    feat = np.empty((0, NFeat))
    while nj <= L:
        ti = ni / fs + t0
        fe = feat_estimation(ecg_[ni:nj], ppg_[ni:nj], fs, ti)
        if not np.any(np.isnan(fe)):
            feat = np.vstack((feat, fe))
        ni, nj = ni + nD, nj + nD
    if scan:  # cleaning eventual duplicates
        pass
    return feat
