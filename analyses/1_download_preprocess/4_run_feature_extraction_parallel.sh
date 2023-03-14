#!/bin/bash
#SBATCH --mail-type=NONE # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=SchalkampA@cardiff.ac.uk # Your email address
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16gb
#SBATCH --account=scw1329
#SBATCH --time=00-08:00:00 # Time limit hh:mm:ss
#SBATCH -e /scratch/c.c21013066/log/acc/%x-%A_%a.err # Standard error
#SBATCH -o /scratch/c.c21013066/log/acc/%x-%A_%a.out # Standard output
#SBATCH --job-name=accfeatures# Descriptive job name
#SBATCH --partition=c_compute_dri1
##### END OF JOB DEFINITION  #####
set -e

module load anaconda
source activate py38R

#files=( $(find /scratch/scw1329/annkathrin/data/ukbiobank/accelerometer -maxdepth 1 -type d -name "cwa_HC_*") )
#file=${files[${SLURM_ARRAY_TASK_ID}]}
#echo $file
python /scratch/c.c21013066/Paper/ProdromalUKBB/analyses/1_download_preprocess/feature_extraction_parallel.py /scratch/scw1329/annkathrin/data/ukbiobank/accelerometer/cwa_miss_2
