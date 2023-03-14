# Wearable devices identify Parkinson’s disease up to 7 years before clinical diagnosis

This repository contains all the code necessary to replicate the analysis presented in the pre-print: 
Wearable devices can identify Parkinson’s disease up to 7 years before clinical diagnosis, Schalkamp et al., 2022 (https://doi.org/10.1101/2022.11.28.22282809)

## Requirements

All major packages are listed in envionment/requirements.txt but the conda environments are also provided in env.yml. Most analyses are carried out under env.yml, but the accelerometer data processing requires the accelerometer.yml environment and deriving the statistical tests relies on pythonstats.yml. Data extraction from UK Biobank relies on a customised version of the ukbb_parser package (https://github.com/nadavbra/ukbb_parser) and processing of the accelerometry data uses the biobankAccelerometerAnalysis package (https://github.com/OxWearables/biobankAccelerometerAnalysis).

## Folder Structure

The resources folder contains information on UKBB specific clinical codes and field codes as well as utilities to handle the data.
The analyses folder contains all relevent scripts and notebooks from extracting and preprocessing the data to running the models.
The make_figures contains notebooks to remake all figures included in the paper as well as extracting the relevant supplemental table information.
