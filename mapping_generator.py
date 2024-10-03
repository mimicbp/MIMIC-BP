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
Expected input:
    Directory to the database common path, e.g.:
        dbpath = '/data-local/Blood-Pressure/files'
    List of file names to be processed, e.g.:
        fname = 'good-files-all.csv'
    Analysis windows of T seconds at every T seconds, 10<=T<=30, e.g.:
        T = 30 (-T 30 or --window_time 30)
    Flag to invert (reflective) or not invert (transmissive) the PPG signal:
        invppg

Expected output:
    Files with the following extensions (having fname as basename):
    ext = '_BP.csv'  # extension for main output file
    ext_not_ok = '_BP_not_ok.csv'  # extension for log of problem files


Example of file good-files-all.csv:
    > head -3 good-files-all.csv
    fname
    p00/p009258/3013677_0026
    p00/p009258/3013677_0014
    > wc good-files-all.csv
       55899   55899 1397456 good-files-all.csv

Usage example 1: (MIMIC: transmissive - do not invert PPG)
> python mapping_generator.py --dbpath /data-local/Blood-Pressure/files
                              --fname good-files-all.csv
                              -T 30
Then, check output files:
good-files-all_BP.csv
good-files-all_BP_not_ok.csv

Usage example 2: (reflective - invert PPG)
> python mapping_generator.py --dbpath /data-local/Blood-Pressure/reflective
                              --fname good-files-reflective.csv
                              -T 30
                              --invppg
Then, check output files:
good-files-reflective_BP.csv
good-files-reflective_BP_not_ok.csv

"""
import numpy as np
import pandas as pd

import pat
import wfdb

import argparse


# ### Auxiliary functions


def fprint_array(file_handle, array: np.ndarray, decimals: int = 0) -> None:
    """
    Print array into file

    Parameters
    ----------
    file_handle :
        The file where the array will be printed. This function does not open
        or close the file.

    array : np.ndarray
        The array to be saved

    decimals : int = 0
        The number of decimal places to consider when saving the array
    """
    L = len(array)
    file_handle.write(",")  # separator
    for i, n in enumerate(array):
        if i == L - 1:
            if decimals:
                file_handle.write(f"{n:.1f}")
            else:
                file_handle.write(f"{n:.0f}")
        else:
            if decimals:
                file_handle.write(f"{n:.1f} ")
            else:
                file_handle.write(f"{n:.0f} ")


def save_record(
    file_handle,
    fname: str,
    r_instants: np.ndarray,
    ppg_start: np.ndarray,
    periods: np.ndarray,
    sbp: np.ndarray,
    dbp: np.ndarray,
    conf: np.ndarray,
) -> None:
    """
    Save data into openned file with handle file_handle

    Parameters
    ----------
    file_handle :
        The file where the array will be printed. This function does not open
        or close the file.

    fname : str
        A string containing the MIMIC file's filename being processed

    r_instants, ppg_start, periods, sbp, dbp, conf : np.ndarray
        The variables that will be saved, obtained from function pat.FEAT_R()
        (some variables need processing)
    """
    # header: file_handle.write('fname,r_wave,period,sbp,dbp,conf\n')
    file_handle.write(fname)
    fprint_array(file_handle, r_instants, 0)
    fprint_array(file_handle, ppg_start, 0)
    fprint_array(file_handle, periods, 0)
    fprint_array(file_handle, sbp, 1)
    fprint_array(file_handle, dbp, 1)
    fprint_array(file_handle, conf, 0)
    file_handle.write("\n")


def get_abp(abp: np.ndarray, nstart: np.ndarray, per: np.ndarray) -> tuple:
    """
    Get beat-to-beat SBP and DBP

    Parameters
    ----------
    nstart : np.ndarray
        Array containing the first sample index of each period

    per : np.ndarray
        Corresponding period lengh in number of samples


    Returns
    -------
    (np.array(sbp), np.array(dbp)) : tuple
        A tuple of arrays containing the SBP and DBP values in the period.
        Returning a tuple makes it easy to retrieve the results via tuple
        unpacking

    NOTE
    ----
    "nstart" and "per" are in terms of sample indexes, not seconds

    """
    sbp = []
    dbp = []
    for r, p in zip(nstart, per):
        sbp_ = np.max(abp[r : r + p + 1])
        sbp.append(sbp_)
        dbp_ = np.min(abp[r : r + p + 1])
        dbp.append(dbp_)
    return np.array(sbp), np.array(dbp)


def fn_fnprob(icsv, fname, ext_not_ok, ext):
    if icsv > 0:
        fn = fname[:icsv] + ext  # main output file
        fnprob = fname[:icsv] + ext_not_ok  # keep names of files with problems
    else:
        fn = fname + ext  # main output file
        fnprob = (
            fname + ext_not_ok
        )  # file  # keep names of files with problems
    return fn, fnprob


def define_ind(W, n0, N, ecg):
    if W > 0:
        ind = np.array(range(n0, min(n0 + N * W, len(ecg))))
    else:
        ind = np.array(range(n0, len(ecg)))
    return ind


def try_main(record, ecg, ppg, abp, record_fn, fprob, fs):
    try:
        assert record.fs == fs, "Unexpected sampling frequency"
        assert len(ecg) == len(ppg), "Lengths differ: ecg, ppg"
        assert len(ecg) == len(abp), "Lengths differ: ecg, abp"
    except Exception as inst:
        print(f"Problem processing <{record_fn}>")
        print(f"{type(inst)}: {inst}")
        fprob.write(f"{record_fn}\n")
        return True
    return False


def main(args) -> None:
    """
    Processing a sequence of records
    """
    # reading sequence of records into fnames
    invppg = -1 if args.invppg else 1
    dbpath = args.dbpath
    fname = args.fname
    T = args.window_time

    print()
    print(f"Input parameters:")
    print(f'{"dbpath":>16}: {dbpath}')
    print(f'{"fname":>16}: {fname}')
    print(f'{"T":>16}: {T}')
    print(f'{"invppg":>16}: {invppg} (1: MIMIC; -1: reflective)')
    print()
    feat = pd.read_csv(fname)
    fnames = feat["fname"]

    fs = 125  # MIMIC sampling frequency, Hz
    # invppg = 1  # 1: do not invert PPG (MIMIC) or -1: invert PPG (reflective)

    # openning output file and problem_file
    icsv = fname.find(".csv")
    ext = "_BP.csv"
    ext_not_ok = "_BP_not_ok.csv"

    fn, fnprob = fn_fnprob(icsv, fname, ext_not_ok, ext)

    fp = open(fn, "w")
    fp.write("fname,r_wave,ppg_start,period,sbp,dbp,conf\n")
    fprob = open(fnprob, "w")
    fprob.write("fname\n")

    # general analysis parameters
    T += 0.9  # analysis window duration in seconds
    N = T * fs  # samples per analysis window
    # t0 to be included as command line argument
    t0 = 0 / fs  # initial overall analysis time
    n0 = int(t0 * fs)  # initial sample index
    # W to be included as command line argument
    # e.g. if W = 20 process 20 analysis windows
    # e.g. if W = 0  process the whole signal
    W = 0  # W analysis windows (if W==0: process whole signal)
    count = 0
    for record_fn in fnames:
        record_fn_full = dbpath + "/" + record_fn
        record = wfdb.rdrecord(record_fn_full)

        # getting signals of interest
        df = pd.DataFrame(record.p_signal, columns=record.sig_name)
        ecg = np.array(df["II"])
        ppg = np.array(df["PLETH"]) * invppg
        abp = np.array(df["ABP"])

        # visualization {True, False}

        # feature extraction
        except_ = try_main(
            record=record,
            ecg=ecg,
            ppg=ppg,
            abp=abp,
            record_fn=record_fn,
            fprob=fprob,
            fs=fs,
        )
        if except_:
            continue

        ind = define_ind(W=W, n0=n0, N=N, ecg=ecg)

        features = pat.FEAT_R(
            ecg[ind],
            ppg[ind],
            pat.fc1ecg,
            pat.fc2ecg,
            pat.fc1ppg,
            pat.fc2ppg,
            fs,
            pat.order,
            T,
            t0,
        )
        if len(features) == 0:
            print(f"Problem processing <{record_fn}>")
            fprob.write(f"{record_fn}\n")
            continue

        # getting SBP and DBP
        r_instant = np.array(
            [int(n) for n in np.floor(features[:, 0] * fs)]
        )  # sample index
        ppg_start = r_instant + np.array(
            [int(n) for n in np.floor((features[:, 1] - features[:, 2]) * fs)]
        )  # sample index
        ppg_start[0] = 0 if ppg_start[0] < 0 else ppg_start[0]
        # duration in samples
        period = np.array([int(n) for n in np.floor(features[:, 3] * fs)])
        conf = np.array([int(n) for n in np.floor(features[:, -1])])
        abp[np.isnan(abp)] = -1  # converting abp nan to -1
        sbp, dbp = get_abp(abp[ind], ppg_start, period)

        # writing into .csv file
        save_record(
            fp, record_fn, r_instant, ppg_start, period, sbp, dbp, conf
        )
        count = count + 1
        print(count, end=" ", flush=True)

    fp.close()
    fprob.close()
    print(f"\n({count}) files processed from a total of ({len(fnames)})")
    print(f"Please, also check <{fnprob}>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--invppg", action="store_true")
    parser.add_argument("-d", "--dbpath", type=str, required=True)
    parser.add_argument("-f", "--fname", type=str, required=True)
    parser.add_argument("-T", "--window_time", type=int, required=True)
    args = parser.parse_args()
    if args.window_time < 10 or args.window_time > 30:
        parser.error(
            "\nSpecify proper 'T' (window_time) in the range [10, 30]\n"
        )

    main(args)
