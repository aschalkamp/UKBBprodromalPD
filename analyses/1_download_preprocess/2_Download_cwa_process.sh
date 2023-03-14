# specify which field to download from UKBB
field='90001'
data=50687 # this is the basket number from UKBB which holds the necessary data
#generate bulk file
data=50687_90001_0_0_HCmiss
#./ukbconv ukb${data}.enc_ukb bulk -s${field}
# fetch the data and download
# can specify how many, max is 50000 so we do it in batches of 1000
start=1
stop=55 # max number of files to download
incr=100
cd /scratch/scw1329/annkathrin/data/ukbiobank # where to store the cwa files

FLAG=0
for ((i=$start; i <= $stop; i += $incr))
do
        name=$(($i + 1))
        end=$(($i+(($incr))))
        echo $end
        if (($end < $stop))
        then
                echo "until end"
                echo "$i - $end"
                ./ukbfetch -bukb${data}.bulk -s${i} -m${incr}
        else
                echo "until stop"
                echo "$i - $stop"
                remains=$((($stop-(($i))+1)))
                ./ukbfetch -bukb${data}.bulk -s${i} -m${remains}
        fi
        # move to folder
        mkdir -p /scratch/scw1329/annkathrin/data/ukbiobank/accelerometer/cwa_miss_${name}
        mv /scratch/scw1329/annkathrin/data/ukbiobank/*.cwa /scratch/scw1329/annkathrin/data/ukbiobank/accelerometer/cwa_miss_${name}
        
        # preprocess data using the UKBB provided software https://github.com/OxWearables/biobankAccelerometerAnalysis
        prev_id=$( sbatch /scratch/c.c21013066/Paper/ProdromalUKBB/analyses/1_download_preprocess/preprocess_data.sh $name| cut -d ' ' -f4)
done