cd /scratch/scw1329/annkathrin/data/ukbiobank

for ((i=2771; i <= 6571; i += 100))
#for i in 11302 12102 15602 18602 25902 26702 34402;
do
        zip /gluster/dri02/rdscw/shared/public/UKBIOBANK/phenotypes/accelerometer/cwa_HC_${i}.zip /scratch/scw1329/annkathrin/data/ukbiobank/accelerometer/cwa_HC_${i}/*.cwa
        
        rm /scratch/scw1329/annkathrin/data/ukbiobank/accelerometer/cwa_HC_${i}/*.cwa
done