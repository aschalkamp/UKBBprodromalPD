#!/bin/bash
#SBATCH --mail-type=NONE # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=SchalkampA@cardiff.ac.uk # Your email address
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --account=scw1329
#SBATCH --time=00-01:30:00 # Time limit hh:mm:ss
#SBATCH -e /scratch/c.c21013066/log/acc/%x-%A_%a.err # Standard error
#SBATCH -o /scratch/c.c21013066/log/acc/%x-%A_%a.out # Standard output
#SBATCH --job-name=rsf_auprc# Descriptive job name
#SBATCH --partition=c_compute_dri1
##### END OF JOB DEFINITION  #####
set -e

module load anaconda
#source activate py38R
source activate sksurvauprc

#python /scratch/c.c21013066/Paper/ProdromalUKBB/analyses/4_survival_model/get_risk.py
python /scratch/c.c21013066/Paper/ProdromalUKBB/analyses/4_survival_model/get_auprc.py
#python /scratch/c.c21013066/Paper/ProdromalUKBB/analyses/4_survival_model/get_NCaseControl.py