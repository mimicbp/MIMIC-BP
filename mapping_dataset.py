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
mapping_dataset.py creates dataset input (train, valid, test) from
files created by mapping_reader.py, entered via option -f in this script
The option -d argument points to the database directory
The number of segments per subject is determined by option -s
Sampling rate of the database is specified by option -r
Option -i causes the inversion of the PPG signal and should be applied
when the source of PPG is inverted (reflective PPG)
The output format can be controlled by the option -m (--model)

> python mapping_dataset.py -h
usage: mapping_dataset.py [-h] -d DBPATH -f FILE -s NSEGS -m {mimic} [-r RATE] [-i] [-e SEED]

optional arguments:
  -h, --help            show this help message and exit
  -d DBPATH, --dbpath DBPATH
                        database path
  -f FILE, --file FILE  input file name
  -s NSEGS, --nsegs NSEGS
                        number of segments per patient
  -m {mimic}, --model {mimic}
                        dataset format
  -r RATE, --rate RATE  sampling frequency (rate), Hz
  -i, --invppg          invert ppg (Reflective PPG)
  -e SEED, --seed SEED  seed

Usage examples:

> python mapping_dataset.py \
         -d /data-local/Blood-Pressure/files/ \
         -f mapping_T30_DeltaBP25_SBPmin60_DBPmax120_MaxSeg60000.txt \
         -s 30

Debugger:
> python -m pdb mapping_dataset.py \
         -d /data-local/Blood-Pressure/files/ \
         -f mapping_T30_DeltaBP25_SBPmin60_DBPmax120_MaxSeg60000.txt \
         -s 30

IPython:
%run mapping_dataset.py \
         -d /data-local/Blood-Pressure/files/ \
         -f mapping_T30_DeltaBP25_SBPmin60_DBPmax120_MaxSeg60000.txt \
         -s 30
    or:
!python mapping_dataset.py \
         -d /data-local/Blood-Pressure/files/ \
         -f mapping_T30_DeltaBP25_SBPmin60_DBPmax120_MaxSeg60000.txt \
         -s 30

IPython & debugger:
%run -m pdb mapping_dataset.py \
         -d /data-local/Blood-Pressure/files/ \
         -f mapping_T30_DeltaBP25_SBPmin60_DBPmax120_MaxSeg60000.txt \
         -s 30
"""
import re
import os
import wfdb
import numpy as np
import pandas as pd

import argparse


def return_patients_to_delete(P, nsegs):
    patients_to_delete = []
    for patient in P:
        if len(P[patient]) < nsegs:
            patients_to_delete.append(patient)
    for patient in patients_to_delete:
        del P[patient]
    return P


def read_file(filename, D, P):
    with open(filename) as file:
        i = 0
        line = file.readline()
        while line:
            if i == 0:
                m = re.findall(r"\d+", line)
                T = int(m[0])
                N = int(m[1])
                print(f"T = {T}, N = {N}")
            i += 1
            m = re.search(" - (.+):", line)
            if m:
                record_fn = m.group(1)
                if record_fn in D:
                    print(f"Duplicated record <{record_fn}>")
                else:
                    _, patient, _ = record_fn.split("/")
                    D[record_fn] = []
                    if patient not in P:
                        P[patient] = []
            m = re.search("(^\d+),(.+),(.+)$", line)
            if m:
                n0 = int(m.group(1))
                SBP = float(m.group(2))
                DBP = float(m.group(3))

                P[patient].append((record_fn, n0, SBP, DBP))

            line = file.readline()
    return D, P, N


def check_nsegs(P, NP, nsegs):
    for patient in P:
        if len(P[patient]) == nsegs:
            NP += 1
        else:
            raise Exception(f"WARNING: patient should have {nsegs} segments")
    return NP


def raise_exept_main(NP, P):
    if NP < 50:
        raise Exception(f"WARNING: too few patients: {NP}")
    if NP != len(P):
        raise Exception(
            f"WARNING: inconsistency on number of patients:{NP} vs {len(P)}"
        )


def main():
    dbpath = args.dbpath
    filename = args.file
    np.random.seed(seed=args.seed)
    D = {}
    P = {}  # patients
    D, P, N = read_file(filename=filename, D=D, P=P)
    P = return_patients_to_delete(P, nsegs=args.nsegs)

    # sort and leave only args.nsegs per patient
    PP = {}
    for patient in P:
        Np = len(P[patient])
        rand_idx = np.random.choice(Np, args.nsegs, replace=False)
        PP[patient] = []
        for idx in sorted(rand_idx):
            PP[patient].append(P[patient][idx])
    P = PP
    NP = 0

    NP = check_nsegs(P=P, NP=NP, nsegs=args.nsegs)
    raise_exept_main(NP, P)

    print(f'{"Total number of patients":>25} = {NP:4d}')
    print(f'{"Segments per patient":>25} = {args.nsegs:4d}')

    # all patients
    Plist = sorted(P.keys())

    # creating dataset
    if args.model == "mimic":
        create_mimicbp_dataset(plist=Plist, N=N, D=P, dbpath=dbpath)
    else:
        raise Exception(f"WARNING: unknown dataset {args.model}")


def create_mimicbp_dataset(plist, N, D, dbpath, dataset_name="mimic-bp"):
    os.makedirs(dataset_name, exist_ok=True)
    signals_mapper = dict(ABP="abp", PLETH="ppg", II="ecg", RESP="resp")

    for patient in plist:
        patient_records = []
        patient_bp_labels = []

        for segment in D[patient]:
            record_fn = segment[0]
            n0 = segment[1]
            abp_labels = segment[2:4]  # SBP, DBP

            record_fn_full = "/".join([dbpath, record_fn])
            record = wfdb.rdrecord(record_fn_full)

            record = (
                pd.DataFrame(record.p_signal, columns=record.sig_name)
                .iloc[n0 : n0 + N, :]
                .filter(signals_mapper)
                .replace(np.nan, 0)
                .rename(signals_mapper, axis=1)
            )
            record["ppg"] *= (-1) ** args.invppg
            record["resp"] = (
                0 if "resp" not in record.columns else record["resp"]
            )

            patient_records.append(record.copy())
            patient_bp_labels.append(abp_labels)

        patient_records = np.array(patient_records).reshape(
            args.nsegs, N, len(signals_mapper)
        )  # shape = (args.nsegs, N)
        patient_bp_labels = np.array(
            patient_bp_labels
        )  # shape = (args.nsegs, 2)

        for signal_idx, signal_name in enumerate(signals_mapper.values()):
            signal = patient_records[:, :, signal_idx]

            with open(
                os.path.join(dataset_name, f"{patient}_{signal_name}.npy"),
                "wb",
            ) as f:
                np.save(f, signal)

        with open(
            os.path.join(dataset_name, f"{patient}_labels.npy"), "wb"
        ) as f:
            np.save(f, patient_bp_labels)

        print(
            f"Files <{patient}_{{ecg,ppg,resp,abp,abp_labels}}.npy> saved!",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dbpath", help="database path", required=True)
    parser.add_argument(
        "-f", "--file", type=str, help="input file name", required=True
    )
    parser.add_argument(
        "-s",
        "--nsegs",
        type=int,
        help="number of segments per patient",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="dataset format",
        choices=["mimic"],
        required=True,
    )
    parser.add_argument(
        "-r",
        "--rate",
        type=int,
        help="sampling frequency (rate), Hz",
        default=125,
    )
    parser.add_argument(
        "-i", "--invppg", help="invert ppg", action="store_true"
    )
    parser.add_argument("-e", "--seed", type=int, help="seed", default=None)
    args = parser.parse_args()
    main()
