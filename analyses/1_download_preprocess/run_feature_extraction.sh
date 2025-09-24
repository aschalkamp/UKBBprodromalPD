#!/bin/bash
#SBATCH --mail-type=NONE # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=SchalkampA@cardiff.ac.uk # Your email address
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32gb
#SBATCH --account=scw1329
#SBATCH --time=01-23:00:00 # Time limit hh:mm:ss
#SBATCH -e /scratch/c.c21013066/log/acc/%x-%A_%a.err # Standard error
#SBATCH -o /scratch/c.c21013066/log/acc/%x-%A_%a.out # Standard output
#SBATCH --job-name=accfeatures# Descriptive job name
#SBATCH --partition=c_compute_dri1
##### END OF JOB DEFINITION  #####
set -e

module load anaconda
source activate py38R

python /scratch/c.c21013066/Paper/ProdromalUKBB/analyses/1_download_preprocess/accelerometry_features.py