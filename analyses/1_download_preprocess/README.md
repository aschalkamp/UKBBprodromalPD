# Extract data from UK Biobank

The first notebook (1_ExtractData.ipynb) extracts all information from the tabular format data of UKBB. This includes accelerometer statistics, blood measures, lifestyle factors, demographic information, and the diagnosis codes.

To download and preprocess the raw accelerometry data (bulk data), we used this parallelized script: 2_Download_cwa_process.sh.
Which makes use of preprocess_data.sh

To create the disease groups (3_CurateDatasets.ipynb) we identify prodromal and diagnosed cases and match age-, and sex-matched unaffected controls to them.

To extract more accelerometry features, we run 4_run_feature_extraction_parallel.sh, which makes use of feature_extraction_parallel.py.

We monitor all this and merge all data together in 5_Accelerometer_ExtractFeatures.ipynb.

We merge this info with information on prodromal symptoms in 6_ProdromalSymptoms.ipynb.

To inspect the cohort we use 7_PredictionCohort.ipynb.