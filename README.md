# MIMIC-BP

This repository is related to the MIMIC-BP dataset, hosted at Harvard Dataverse:

https://doi.org/10.7910/DVN/DBM1NF

At [Harvard Dataverse](https://doi.org/10.7910/DVN/DBM1NF), you will find relevant additional information.

Here, we have the scripts used to build MIMIC-BP from the original [MIMIC-III Waveform Database Matched Subset](https://physionet.org/content/mimic3wdb-matched/1.0/) (2.4 TB).

You only need these scripts if you want to rebuild the MIMIC-BP dataset from the original MIMIC-III dataset.

Otherwise, if you just want to use the MIMIC-BP dataset in your research, you only need to download and use the files already available at [Harvard Dataverse](https://doi.org/10.7910/DVN/DBM1NF).

# MIMIC-BP dataset generation pipeline

Instructions to generate the MIMIC-BP or similar subsets

## Development environment

The required environment, `mimic-bp`, can be created by

```bash
conda create -n mimic-bp python=3.8.13 numpy=1.19.5 scipy=1.8.0 wfdb=3.4.1 pandas=1.4.2 matplotlib=3.5.2
```

## Dataset generation

Below is just one example of a sequence of bash commands that
- from a list of suitable MIMIC records (`good-files-all.csv`)
- `mapping_generator.py` maps these records on a beat-to-beat format (`good-files-all_BP.csv`)
- then `mapping_reader.py` creates a new list of records that contain suitable segments
- finally, `mapping_dataset.py` saves the MIMIC-BP data files in `mimic-bp` folder

```bash
#!/bin/bash

MIMIC=/data-local/Blood-Pressure/files  # MIMIC root directory
SDUR=30  # segment duration
SPP=30  # segments per patient
TAG=mimic_T${SDUR}_${SPP}spp  # name of directory for dataset and result

# maps MIMIC database beat-to-beat and produces good-files-all_BP.csv
python mapping_generator.py --dbpath ${MIMIC} \
   --fname good-files-all.csv \
   -T 30

# uses good-files-all_BP.csv to produce list of segments of ${SDUR} seconds
# output file: mapping_T${SDUR}_DeltaBP25_SBPmin60_DBPmax120_MaxSeg60000.txt
python mapping_reader.py -d ${MIMIC} \
   -f good-files-all_BP.csv \
   -T ${SDUR}

# # uses output from above command to generate the dataset with
# # ${SDUR}-second segments and ${SPP} segments per patient
python mapping_dataset.py -d ${MIMIC} \
   -f mapping_T${SDUR}_DeltaBP25_SBPmin60_DBPmax120_MaxSeg60000.txt \
   -m mimic \
   -s ${SPP}

# After the above command, MIMIC-BP dataset will be under "mimic-bp" folder
```
