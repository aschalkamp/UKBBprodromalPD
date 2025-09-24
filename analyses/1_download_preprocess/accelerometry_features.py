import numpy as np
import pandas as pd
import glob
import re
import os
from scipy import stats
import sys
sys.path.insert(1,'/scratch/c.c21013066/software/biobankAccelerometerAnalysis/accelerometer')
import utils

name='ParkinsonDisease'
savename = 'Healthy'
diag_name='ParkinsonDisease'
drop_healthy='neurology'
data_path='/scratch/c.c21013066/data/ukbiobank/sample/withGP/'
save_path1='/scratch/c.c21013066/data/ukbiobank/phenotypes/accelerometer/'
save_path='/scratch/scw1329/annkathrin/data/ukbiobank/accelerometer/'

merged_clean = pd.read_csv(f'{data_path}{name}_controlNoneurology_acc_QC.csv',dtype={'eid':int}).set_index('eid')
matched_eid = pd.read_csv(f'{data_path}{name}_controlNo{drop_healthy}_match_accage_acc_QC.txt',header=None,names=['eid'])
merged_clean['diagnosis'] = merged_clean[name].replace([0,1],['Healthy',diag_name])
merged_clean.loc[merged_clean[f'{name}_age']<=0,f'{name}_age'] = merged_clean.filter(regex='_age')[merged_clean.filter(regex='_age')>0].min(axis=1)
merged_clean['acc_time_since_diagnosis'] = merged_clean[f'accelerometry_age'] - merged_clean[f'{name}_age']
merged_clean['acc_time_to_diagnosis'] =  merged_clean[f'{name}_age'] - merged_clean[f'accelerometry_age']
merged_clean['acc_incident'] = merged_clean[f'{name}_age'] > merged_clean[f'accelerometry_age']
merged_clean.loc[merged_clean['acc_time_since_diagnosis'].isna(),'acc_incident'] = np.nan
merged_clean['diagnosis_prod'] = merged_clean['diagnosis'].copy(deep=True)
merged_clean.loc[np.logical_and(merged_clean['acc_incident']==1,merged_clean['diagnosis']==diag_name),'diagnosis_prod'] = 'Prodromal'
merged_clean['diagnosis_prod_conservative'] = merged_clean['diagnosis_prod'].copy(deep=True)
merged_clean.loc[np.logical_and(merged_clean['acc_time_to_diagnosis']<2,merged_clean['diagnosis_prod']=='Prodromal'),'diagnosis_prod_conservative'] = diag_name
matched_sample = merged_clean[merged_clean['diagnosis']=='Healthy']#merged_clean.loc[matched_eid['eid']]

filenames = pd.read_csv('/scratch/c.c21013066/data/ukbiobank/phenotypes/accelerometer/subject_file_lookup.csv')
subjects_avail = filenames['eid']
intersect = np.intersect1d(matched_sample.index,subjects_avail)
matched_avail = matched_sample.loc[intersect]

sum_sleep_raw = matched_avail.copy(deep=True)
processed = pd.read_csv(f'{save_path1}allsubjects_summary_fromraw.csv').set_index('eid')
sum_sleep_raw = pd.merge(sum_sleep_raw,processed,on='eid',how='left',suffixes=['','_drop'])
sum_sleep_raw = sum_sleep_raw.drop(columns=sum_sleep_raw.filter(regex='_drop'))
sum_sleep_raw.loc[np.intersect1d(processed.index,sum_sleep_raw.index),:] = processed.loc[np.intersect1d(processed.index,sum_sleep_raw.index),:]

todo = np.setdiff1d(sum_sleep_raw.index,processed.index)
print('to process: ', len(todo))

for eid in todo:
    # check where eid is in foldersystem
    file = filenames.loc[filenames['eid']==eid,'file'].iloc[0]
    print(file)
    data_raw = pd.read_csv(file)
    data_raw['time'] = data_raw['time'].apply(utils.date_parser)
    data_raw = data_raw.set_index('time')
    for cl in ['sleep','light','sedentary','MVPA','imputed']:
        sum_sleep_raw.loc[eid,f'total_{cl}_hours'] = data_raw[cl].sum()
        # data recorded in 30sec intervals where then label is given
        # to get hours of sleep per day, we have to sum 30sec labels per day and divide by 60*2 # remove first and last day
        sum_sleep_raw.loc[eid,f'mean_{cl}_hours_perday'] = (data_raw.groupby([data_raw.index.date])[cl].sum()/120)[1:-1].mean()
        sum_sleep_raw.loc[eid,f'std_{cl}_hours_perday'] = (data_raw.groupby([data_raw.index.date])[cl].sum()/120)[1:-1].std()
        # instead use 24h intervals from first 10h to last 10h
        sum_sleep_raw.loc[eid,f'mean_{cl}_hours_per24h'] = (data_raw.groupby(pd.Grouper(freq='24h', offset='10h', label='left'))[cl].sum()/120).mean()
        sum_sleep_raw.loc[eid,f'std_{cl}_hours_per24h'] = (data_raw.groupby(pd.Grouper(freq='24h', offset='10h', label='left'))[cl].sum()/120).std()
        sum_sleep_raw.loc[eid,f'mean_movement_during_{cl}'] = data_raw.loc[data_raw[cl]>0,'acc'].mean()
        sum_sleep_raw.loc[eid,f'std_movement_during_{cl}'] = data_raw.loc[data_raw[cl]>0,'acc'].std()

        # how often wake up during sleep
        # identify sleep window and count 
        data_raw[f'consec_{cl}'] = data_raw[cl] * (data_raw.groupby((data_raw[cl] != data_raw[cl].shift()).cumsum()).cumcount() + 1)
        sum_sleep_raw.loc[eid,f'mean_max_{cl}_hours_consecutive_per24h'] = (data_raw.groupby(pd.Grouper(freq='24h', offset='10h', label='left'))[f'consec_{cl}'].max()/120).mean()
        sum_sleep_raw.loc[eid,f'max_{cl}_hours_consecutive'] = data_raw[f'consec_{cl}'].max()/120
        # how often asleep during 24h?
        data_raw[f'starts_{cl}'] = data_raw[f'consec_{cl}'] == 1
        sum_sleep_raw.loc[eid,f'mean_N_{cl}_intervals_per24h'] = (data_raw.groupby(pd.Grouper(freq='24h', offset='10h', label='left'))[f'starts_{cl}'].sum()).mean()
        # how often nap during day?
        sum_sleep_raw.loc[eid,f'mean_N_{cl}_intervals_22-10'] = (data_raw.groupby(pd.Grouper(freq='12h', offset='10h', label='left'))[f'starts_{cl}'].sum())[1::2].mean()
        # how often awake during night?
        sum_sleep_raw.loc[eid,f'mean_N_{cl}_intervals_10-22'] = (data_raw.groupby(pd.Grouper(freq='12h', offset='10h', label='left'))[f'starts_{cl}'].sum())[::2].mean()
        # acceleration shortly before waking up
        # select 2min (4 instances) before last sleep label and calculate acc mean

sum_sleep_raw.to_csv(f'{save_path1}{savename}_summary_fromraw.csv')
