#!/bin/bash
#SBATCH --mail-type=NONE # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=SchalkampA@cardiff.ac.uk # Your email address
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64gb
#SBATCH --account=scw1329
#SBATCH --time=00-23:00:00 # Time limit hh:mm:ss
#SBATCH -e /scratch/c.c21013066/log/acc/%x-%A_%a.err # Standard error
#SBATCH -o /scratch/c.c21013066/log/acc/%x-%A_%a.out # Standard output
#SBATCH --job-name=stacked_noOsteo# Descriptive job name
#SBATCH --partition=c_compute_dri1
##### END OF JOB DEFINITION  #####
set -e

module load anaconda
source activate py38R

#python /scratch/c.c21013066/Paper/ProdromalUKBB/analyses/3_logistic_regression/acc_predict_HCmodels.py
#python /scratch/c.c21013066/Paper/ProdromalUKBB/analyses/3_logistic_regression/acc_predict_popmodels.py
python /scratch/c.c21013066/Paper/ProdromalUKBB/analyses/3_logistic_regression/acc_predict_stacked.py