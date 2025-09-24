# specify which field to download from UKBB
field='90001'
data=50687 # this is the basket number from UKBB which holds the necessary data
#generate bulk file
./ukbconv ukb${data}.enc_ukb bulk -s${field}
# fetch the data and download
# can specify how many, max is 50000 so we do it in batches of 1000
start=1
stop=1 # max number of files to download
incr=10
cd /scratch/c.c21013066/data/ukbiobank # where to store the cwa files

start=`date +%s`
./ukbfetch -e1000057 -d90001_0_0
end=`date +%s`

runtime=$((end-start))
echo $runtime
runtime=$( echo "$end - $start" | bc -l )