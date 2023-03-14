import numpy as np
import pandas as pd
import glob
import re
import os
from scipy import stats
import sys
sys.path.insert(1,'/scratch/c.c21013066/software/biobankAccelerometerAnalysis/accelerometer')
import utils

data_path='/scratch/c.c21013066/data/ukbiobank/sample/withGP/'
save_path1='/scratch/c.c21013066/data/ukbiobank/phenotypes/accelerometer/'
save_path='/scratch/scw1329/annkathrin/data/ukbiobank/accelerometer/'

# run through given folder and extract features and save
folder = sys.argv[1]
print(folder)

filenames = pd.read_csv(f'{save_path1}/subject_file_lookup.csv')
filenames = filenames[filenames['path']==folder]
#eids = pd.read_csv('/scratch/scw1329/annkathrin/data/ukbiobank/to_process4.csv')
#eids = eids['eid']
#intersect = np.intersect1d(eids,filenames['eid'])
#filenames = filenames[filenames['eid'].isin(intersect)]

# index = int(sys.argv[1])
# length = filenames.shape[0]//25
# start = index*length
# print(index,start,start+length)
# if index<24:
#     filenames = filenames.iloc[start:start+length,:]
# else:
#     filenames = filenames.iloc[start:,:]
    
subjects_avail = filenames['eid']
#subjects = glob.glob(f"{folder}/*timeSeries.csv.gz")

classes = ['sleep','light','sedentary','MVPA','imputed']
cols = np.hstack(['covered_days','complete_days_starting_10h','complete_days_starting_0h','complete_days_starting_7h', [f'mean_{cl}_hours_perday' for cl in classes],
                  [f'std_{cl}_hours_perday' for cl in classes],
                  [f'mean_{cl}_hours_per24h' for cl in classes],
                  [f'std_{cl}_hours_per24h' for cl in classes],
                  [f'mean_movement_during_{cl}' for cl in classes],
                  [f'std_movement_during_{cl}' for cl in classes],
                  [f'mean_max_{cl}_hours_consecutive_perday' for cl in classes],
                  [f'mean_max_{cl}_hours_consecutive_per24h' for cl in classes],
                  [f'max_{cl}_hours_consecutive' for cl in classes],
                  [f'mean_N_{cl}_intervals_per24h' for cl in classes],
                  [f'mean_N_{cl}_intervals_perday' for cl in classes],
                  [f'mean_N_{cl}_intervals_22-10' for cl in classes],
                  [f'mean_N_{cl}_intervals_10-22' for cl in classes],
                  [f'mean_N_{cl}_intervals_07-23' for cl in classes],
                  [f'mean_N_{cl}_intervals_23-07' for cl in classes]])
sum_sleep_raw = pd.DataFrame(index=subjects_avail,columns=cols)
thr = 2878 # last day stops for all 30sec early, so allow for 1 min to be missing each hour

for eid,file in zip(subjects_avail,filenames['file']):
    # check where eid is in foldersystem
    data_raw = pd.read_csv(file)
    data_raw['time'] = data_raw['time'].apply(utils.date_parser)
    data_raw = data_raw.set_index('time')
    # check how much time coverage
    sum_sleep_raw.loc[eid,f'covered_days'] = (data_raw.index[-1] - data_raw.index[0]) / np.timedelta64(1,'D')
    sum_sleep_raw.loc[eid,f'complete_days_starting_10h'] = (data_raw.groupby(pd.Grouper(freq='24h', offset='10h', label='left')).size() >= thr).sum() # remove incomplete ones
    sum_sleep_raw.loc[eid,f'complete_days_starting_0h'] = (data_raw.groupby(pd.Grouper(freq='24h', label='left')).size() >= thr).sum() # remove first and last day and all incomplete ones
    sum_sleep_raw.loc[eid,f'complete_days_starting_7h'] = (data_raw.groupby(pd.Grouper(freq='24h', offset='7h',label='left')).size() >= thr).sum() # remove first and last day and all incomplete ones
    data_full = data_raw.groupby(pd.Grouper(freq='24h', label='left')).filter(lambda x: len(x) >= thr )
    data_full_10h = data_raw.groupby(pd.Grouper(freq='24h', offset='10h',label='left')).filter(lambda x: len(x) >=thr )
    data_full_7h = data_raw.groupby(pd.Grouper(freq='24h', offset='7h',label='left')).filter(lambda x: len(x) >= thr )
    for cl in classes:
        #sum_sleep_raw.loc[eid,f'total_{cl}_hours'] = data[cl].sum() # invalid as biased by how long people wore it
        # data recorded in 30sec intervals where then label is given
        # to get hours of sleep per day, we have to sum 30sec labels per day and divide by 60*2 # remove first and last day
        sum_sleep_raw.loc[eid,f'mean_{cl}_hours_perday'] = (data_full.groupby([data_full.index.date])[cl].sum()/120).mean()
        sum_sleep_raw.loc[eid,f'std_{cl}_hours_perday'] = (data_full.groupby([data_full.index.date])[cl].sum()/120).std()
        # instead use 24h intervals from first 10h to last 10h
        sum_sleep_raw.loc[eid,f'mean_{cl}_hours_per24h'] = (data_full_10h.groupby(pd.Grouper(freq='24h', offset='10h', label='left'))[cl].sum()/120).mean()
        sum_sleep_raw.loc[eid,f'std_{cl}_hours_per24h'] = (data_full_10h.groupby(pd.Grouper(freq='24h', offset='10h', label='left'))[cl].sum()/120).std()
        
        sum_sleep_raw.loc[eid,f'mean_movement_during_{cl}'] = data_raw.loc[data_raw[cl]>0,'acc'].mean()
        sum_sleep_raw.loc[eid,f'std_movement_during_{cl}'] = data_raw.loc[data_raw[cl]>0,'acc'].std()

        # how often wake up during sleep
        # identify sleep window and count 
        data_raw[f'consec_{cl}'] = data_raw[cl] * (data_raw.groupby((data_raw[cl] != data_raw[cl].shift()).cumsum()).cumcount() + 1)
        data_full[f'consec_{cl}'] = data_full[cl] * (data_full.groupby((data_full[cl] != data_full[cl].shift()).cumsum()).cumcount() + 1)
        data_full_10h[f'consec_{cl}'] = data_full_10h[cl] * (data_full_10h.groupby((data_full_10h[cl] != data_full_10h[cl].shift()).cumsum()).cumcount() + 1)
        data_full_7h[f'consec_{cl}'] = data_full_7h[cl] * (data_full_7h.groupby((data_full_7h[cl] != data_full_7h[cl].shift()).cumsum()).cumcount() + 1)
        sum_sleep_raw.loc[eid,f'mean_max_{cl}_hours_consecutive_perday'] = (data_full.groupby(pd.Grouper(freq='24h',label='left'))[f'consec_{cl}'].max()/120).mean()
        sum_sleep_raw.loc[eid,f'mean_max_{cl}_hours_consecutive_per24h'] = (data_full_10h.groupby(pd.Grouper(freq='24h', offset='10h', label='left'))[f'consec_{cl}'].max()/120).mean()
        sum_sleep_raw.loc[eid,f'max_{cl}_hours_consecutive'] = data_raw[f'consec_{cl}'].max()/120
        # how often asleep during 24h?
        data_raw[f'starts_{cl}'] = data_raw[f'consec_{cl}'] == 1
        data_full[f'starts_{cl}'] = data_full[f'consec_{cl}'] == 1
        data_full_10h[f'starts_{cl}'] = data_full_10h[f'consec_{cl}'] == 1
        data_full_7h[f'starts_{cl}'] = data_full_7h[f'consec_{cl}'] == 1
        sum_sleep_raw.loc[eid,f'mean_N_{cl}_intervals_per24h'] = (data_full_10h.groupby(pd.Grouper(freq='24h', offset='10h', label='left'))[f'starts_{cl}'].sum()).mean()
        sum_sleep_raw.loc[eid,f'mean_N_{cl}_intervals_perday'] = (data_full.groupby(pd.Grouper(freq='24h', label='left'))[f'starts_{cl}'].sum()).mean()
        # how often nap during day?
        sum_sleep_raw.loc[eid,f'mean_N_{cl}_intervals_22-10'] = (data_full_10h.groupby(pd.Grouper(freq='12h', offset='10h', label='left'))[f'starts_{cl}'].sum())[1::2].mean()
        # how often awake during night?
        sum_sleep_raw.loc[eid,f'mean_N_{cl}_intervals_10-22'] = (data_full_10h.groupby(pd.Grouper(freq='12h', offset='10h', label='left'))[f'starts_{cl}'].sum())[::2].mean()
        # alternative definition of day/night
        # as recording starts at 10am and ends at 10am, need to cutoff incomplete ones
        sum_sleep_raw.loc[eid,f'mean_N_{cl}_intervals_23-07'] = (data_full_7h.groupby(pd.Grouper(freq='8h', offset='7h', label='left'))[f'starts_{cl}'].sum())[2::3].mean()
        first_8h = data_full_7h.groupby(pd.Grouper(freq='8h', offset='7h', label='left'))[f'starts_{cl}'].sum()[::3]
        second_8h = data_full_7h.groupby(pd.Grouper(freq='8h', offset='7h', label='left'))[f'starts_{cl}'].sum()[1::3]
        sum_sleep_raw.loc[eid,f'mean_N_{cl}_intervals_07-23'] = (first_8h.values + second_8h.values).mean()

        # acceleration shortly before waking up
        # select 2min (4 instances) before last sleep label and calculate acc mean
print(sum_sleep_raw.describe())
sum_sleep_raw.to_csv(f'/scratch/c.c21013066/data/ukbiobank/phenotypes/accelerometer/allsubject25_summary_from_raw.csv')
#sum_sleep_raw.to_csv(f'/scratch/c.c21013066/data/ukbiobank/phenotypes/accelerometer/allsubject{index}_summary_from_raw.csv')
#sum_sleep_raw.to_csv(f'{folder}/summary_fromraw.csv')
#sum_sleep_raw.to_csv(f'/scratch/scw1329/annkathrin/data/ukbiobank/accelerometer/to_process3/summary_fromraw.csv')