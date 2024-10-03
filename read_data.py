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
Example usage:
> python read_data.py -d mimic-bp -p p093833 -i 29 -g
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def showBP(dbPath, patient, idx):
    "Prints SBP and DBP of segment idx from patient"

    if not os.path.isdir(dbPath):
        raise Exception(f'Check if dbPath "{dbPath}" is correct')

    labels_fn = patient + "_labels.npy"
    labels = np.load(os.path.join(dbPath, labels_fn))
    assert labels.shape == (30, 2), 'Problem reading "{labels_fn}"'

    # systolic blood pressure, sbp, and diastolic blood pressure, dbp
    sbp, dbp = labels[idx, :]
    print(f"Patient {patient}, segment {idx}")
    print(f"SBP = {sbp} mmHg\nDBP = {dbp} mmHg")


def plotWaves(dbPath, patient, idx):
    "Plots first 5 seconds of waveforms"
    print(f"\nFirst 5 seconds of waveforms\n")
    waves = ["abp", "ecg", "ppg", "resp"]
    for wav in waves:
        wave_fn = patient + "_" + wav + ".npy"
        wave = np.load(os.path.join(dbPath, wave_fn))
        assert wave.shape == (30, 3750), 'Problem reading "{wave_fn}"'

        fs = 125  # sampling frequency
        N = len(wave[idx])  # number of samples in a segment
        t = np.arange(N) / fs  # time index
        tidx = t < 5  # plot first five seconds
        plt.plot(t[tidx], wave[idx][tidx])
        plt.xlabel("time (s)")
        if "abp" == wav:
            plt.ylabel("ABP (mmHg)")
        else:
            plt.ylabel(f"{wav.upper()}")
        plt.title(f"Patient {patient}, segment {idx}")
        plt.grid()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="read_data",
        description="How to read files from MIMIC-BP",
        epilog="Last update: 20Jun2023",
    )
    parser.add_argument(
        "-d", "--dbPath", help="path to .npy files", required=True
    )
    parser.add_argument(
        "-p", "--patient", help="patient ID (eg, p093833)", required=True
    )
    parser.add_argument(
        "-i", "--idx", type=int, help="segment index", required=True
    )
    parser.add_argument(
        "-g", "--graph", action="store_true", help="flag for graphs"
    )
    args = parser.parse_args()

    showBP(args.dbPath, args.patient, args.idx)

    if args.graph:
        plotWaves(args.dbPath, args.patient, args.idx)
