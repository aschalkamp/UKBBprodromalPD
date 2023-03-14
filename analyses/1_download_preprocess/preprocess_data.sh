#!/bin/bash
#SBATCH --mail-type=NONE # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=SchalkampA@cardiff.ac.uk # Your email address
#SBATCH --nodes=1
#SBATCH --ntasks=1 # Run a single serial task
#SBATCH --cpus-per-task=1
#SBATCH --mem=32gb
#SBATCH --account=scw1329
#SBATCH --time=0-10:00:00 # Time limit hh:mm:ss
#SBATCH -e /scratch/c.c21013066/log/acc/ukbiobankQC_%A_%a.err # Standard error
#SBATCH -o /scratch/c.c21013066/log/acc/ukbiobankQC_%A_%a.out # Standard output
#SBATCH --job-name=acc # Descriptive job name
#SBATCH --partition=c_compute_dri1           # Use a serial partition 24 cores/7days
##### END OF JOB DEFINITION  #####
set -e
module load anaconda
source activate accelerometer

i=$1

filenames="/scratch/scw1329/annkathrin/data/ukbiobank/accelerometer/cwa_miss_${i}/*_90001_0_0.cwa"
for entry in $filenames
do
    echo $entry
    b=$(basename $entry .cwa)
    echo $b
    file="/scratch/scw1329/annkathrin/data/ukbiobank/accelerometer/cwa_miss_${i}/${b}-timeSeries.csv.gz"
    if test -f "$file"; 
    then
        echo "$file exists."
    else
        accProcess $entry --outputFolder /scratch/scw1329/annkathrin/data/ukbiobank/accelerometer/cwa_miss_${i} || echo "failed to preprocess"
        #accPlot $file || echo "failed to create plot"
    fi
done
# add to zip and delete
#zip /scratch/c.c21013066/data/ukbiobank/phenotypes/accelerometer/cwa_depr_${i}.zip /scratch/c.c21013066/data/ukbiobank/phenotypes/accelerometer/cwa_depr_${i}/*.cwa
#rm /scratch/c.c21013066/data/ukbiobank/phenotypes/accelerometer/cwa_depr_${i}/*.cwa
#mv /scratch/c.c21013066/data/ukbiobank/phenotypes/accelerometer/cwa_depr_${i}/*-timeSeries.csv.gz /scratch/c.c21013066/data/ukbiobank/phenotypes/accelerometer