import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(1,'/scratch/c.c21013066/software/ukbb_parser/ukbb_parser')
sys.path.insert(1,'/scratch/c.c21013066/software/ukbb_parser/ukbb_parser/shared_utils')
import ukbb_parser as ukbb_parser
import ukbb_phenotype_dataset as ukbb_phenotype_dataset
from shared_utils.util import summarize
sys.path.insert(1,'../')
import phenotypesnew as pheno_info
import datetime
#sys.path.insert(1,'/scratch/c.c21013066/UKBIOBANK_DataPreparation/phenotypes')
import _preprocess
from functools import reduce

ukbb_paths = ukbb_parser._load_ukbb_paths()

def get_environment(nrows=None):
    eid, demographics, covariates = ukbb_phenotype_dataset.create_phenotype_dataset(pheno_info.DEMOGRAPHICS,nrows=nrows,parse_dataset_covariates_kwargs={'use_genotyping_metadata':False},no_kinship=False, only_caucasians=False)
    demographics['eid'] = eid
    demographics.set_index('eid',inplace=True)
    demographics = pd.merge(demographics,_preprocess.recode_ethnicity(demographics[['ethnicity']],1001),on='eid')
    demographics = _preprocess.get_birthdate(demographics)

    eid, baseline, covariates = ukbb_phenotype_dataset.create_phenotype_dataset(pheno_info.ASSESSMENTS,nrows=nrows,parse_dataset_covariates_kwargs={'use_genotyping_metadata':False},no_kinship=False, only_caucasians=False)
    baseline['eid'] = eid
    baseline.set_index('eid',inplace=True)
    baseline['visit'] = 0
    baseline['date_visit'] = pd.to_datetime(baseline['date_visit'],format='%Y-%m-%d',errors='coerce')

    eid, physical, covariates = ukbb_phenotype_dataset.create_phenotype_dataset(pheno_info.PHYSICAL,nrows=nrows,parse_dataset_covariates_kwargs={'use_genotyping_metadata':False},no_kinship=False, only_caucasians=False)
    physical['eid'] = eid
    physical.set_index('eid',inplace=True)

    eid, familyhistory, covariates = ukbb_phenotype_dataset.create_phenotype_dataset(pheno_info.FAMILY,nrows=nrows,parse_dataset_covariates_kwargs={'use_genotyping_metadata':False},no_kinship=False, only_caucasians=False)
    familyhistory['eid'] = eid
    familyhistory.set_index('eid',inplace=True)
    familyhistory = _preprocess.recode_family(familyhistory)

    eid, lifestyle, covariates = ukbb_phenotype_dataset.create_phenotype_dataset(pheno_info.LIFESTYLE,nrows=nrows,parse_dataset_covariates_kwargs={'use_genotyping_metadata':False},no_kinship=False, only_caucasians=False)
    lifestyle['eid'] = eid
    lifestyle.set_index('eid',inplace=True)
    lifestyle = pd.merge(lifestyle,_preprocess.recode(lifestyle[['AlcoholFrequency']],100402),on='eid')
    lifestyle["AlcoholFrequency_LessThanWeekly"] = lifestyle[['AlcoholFrequency_Never','AlcoholFrequency_Onetothreetimesamonth',
                                                      'AlcoholFrequency_Specialoccasionsonly']].max(axis=1)
    lifestyle = pd.merge(lifestyle,_preprocess.recode(lifestyle[['DaytimeSleepiness']],100346),on='eid')
    lifestyle = pd.merge(lifestyle,_preprocess.recode_pesticides(lifestyle[['Pesticides']],493),on='eid')
    lifestyle = pd.merge(lifestyle,_preprocess.recode(lifestyle.filter(regex='Status'),90),on='eid')

    eid, genes, covariates = ukbb_phenotype_dataset.create_phenotype_dataset(pheno_info.GENES,nrows=nrows,parse_dataset_covariates_kwargs={'use_genotyping_metadata':False},no_kinship=False, only_caucasians=False)
    genes['eid'] = eid
    genes.set_index('eid',inplace=True)

    eid, icd10diagnoses, covariates = ukbb_phenotype_dataset.create_phenotype_dataset(pheno_info.DIAGNOSESICD10,nrows=nrows,parse_dataset_covariates_kwargs={'use_genotyping_metadata':False},no_kinship=False, only_caucasians=False,code='19')
    icd10diagnoses['eid'] = eid
    
    return [demographics,baseline,icd10diagnoses,lifestyle,familyhistory,physical,genes]
    
def get_risks(nrows=None):
    eid, demographics, covariates = ukbb_phenotype_dataset.create_phenotype_dataset(pheno_info.DEMOGRAPHICS,nrows=nrows,parse_dataset_covariates_kwargs={'use_genotyping_metadata':False},no_kinship=False, only_caucasians=False)
    demographics['eid'] = eid
    demographics.set_index('eid',inplace=True)
    demographics = pd.merge(demographics,_preprocess.recode_ethnicity(demographics[['ethnicity']],1001),on='eid')
    demographics = _preprocess.get_birthdate(demographics)

    eid, baseline, covariates = ukbb_phenotype_dataset.create_phenotype_dataset(pheno_info.ASSESSMENTS,nrows=nrows,parse_dataset_covariates_kwargs={'use_genotyping_metadata':False},no_kinship=False, only_caucasians=False)
    baseline['eid'] = eid
    baseline.set_index('eid',inplace=True)
    baseline['visit'] = 0
    baseline['date_visit'] = pd.to_datetime(baseline['date_visit'],format='%Y-%m-%d',errors='coerce')

    eid, icd10diagnoses, covariates = ukbb_phenotype_dataset.create_phenotype_dataset(pheno_info.DIAGNOSESICD10,nrows=nrows,parse_dataset_covariates_kwargs={'use_genotyping_metadata':False},no_kinship=False, only_caucasians=False,code='19')
    icd10diagnoses['eid'] = eid
    
    eid, physical, covariates = ukbb_phenotype_dataset.create_phenotype_dataset(pheno_info.PHYSICAL,nrows=nrows,parse_dataset_covariates_kwargs={'use_genotyping_metadata':False},no_kinship=False, only_caucasians=False)
    physical['eid'] = eid
    physical.set_index('eid',inplace=True)

    eid, familyhistory, covariates = ukbb_phenotype_dataset.create_phenotype_dataset(pheno_info.FAMILY,nrows=nrows,parse_dataset_covariates_kwargs={'use_genotyping_metadata':False},no_kinship=False, only_caucasians=False)
    familyhistory['eid'] = eid
    familyhistory.set_index('eid',inplace=True)
    familyhistory = _preprocess.recode_family(familyhistory)

    eid, lifestyle, covariates = ukbb_phenotype_dataset.create_phenotype_dataset(pheno_info.LIFESTYLE,nrows=nrows,parse_dataset_covariates_kwargs={'use_genotyping_metadata':False},no_kinship=False, only_caucasians=False)
    lifestyle['eid'] = eid
    lifestyle.set_index('eid',inplace=True)
    lifestyle = pd.merge(lifestyle,_preprocess.recode(lifestyle[['AlcoholFrequency']],100402),on='eid')
    lifestyle["AlcoholFrequency_LessThanWeekly"] = lifestyle[['AlcoholFrequency_Never','AlcoholFrequency_Onetothreetimesamonth',
                                                      'AlcoholFrequency_Specialoccasionsonly']].max(axis=1)
    lifestyle = pd.merge(lifestyle,_preprocess.recode(lifestyle[['DaytimeSleepiness']],100346),on='eid')
    lifestyle = pd.merge(lifestyle,_preprocess.recode_pesticides(lifestyle[['Pesticides']],493),on='eid')
    lifestyle = pd.merge(lifestyle,_preprocess.recode(lifestyle.filter(regex='Status'),90),on='eid')

    eid, chem, covariates = ukbb_phenotype_dataset.create_phenotype_dataset(pheno_info.BloodChemistry,nrows=nrows,parse_dataset_covariates_kwargs={'use_genotyping_metadata':False},no_kinship=False, only_caucasians=False)
    chem['eid'] = eid
    chem.set_index('eid',inplace=True)
    
    return [demographics,baseline,icd10diagnoses,lifestyle,familyhistory,chem,physical]

def get_blood(nrows=None):
    eid, demographics, covariates = ukbb_phenotype_dataset.create_phenotype_dataset(pheno_info.DEMOGRAPHICS,nrows=nrows,parse_dataset_covariates_kwargs={'use_genotyping_metadata':False},no_kinship=False, only_caucasians=False)
    demographics['eid'] = eid
    demographics.set_index('eid',inplace=True)
    demographics = pd.merge(demographics,_preprocess.recode_ethnicity(demographics[['ethnicity']],1001),on='eid')
    demographics = _preprocess.get_birthdate(demographics)

    eid, baseline, covariates = ukbb_phenotype_dataset.create_phenotype_dataset(pheno_info.ASSESSMENTS,nrows=nrows,parse_dataset_covariates_kwargs={'use_genotyping_metadata':False},no_kinship=False, only_caucasians=False)
    baseline['eid'] = eid
    baseline.set_index('eid',inplace=True)
    baseline['visit'] = 0
    baseline['date_visit'] = pd.to_datetime(baseline['date_visit'],format='%Y-%m-%d',errors='coerce')

    eid, icd10diagnoses, covariates = ukbb_phenotype_dataset.create_phenotype_dataset(pheno_info.DIAGNOSESICD10,nrows=nrows,parse_dataset_covariates_kwargs={'use_genotyping_metadata':False},no_kinship=False, only_caucasians=False,code='19')
    icd10diagnoses['eid'] = eid

    eid, chem, covariates = ukbb_phenotype_dataset.create_phenotype_dataset(pheno_info.BloodChemistry,nrows=nrows,parse_dataset_covariates_kwargs={'use_genotyping_metadata':False},no_kinship=False, only_caucasians=False)
    chem['eid'] = eid
    chem.set_index('eid',inplace=True)

    return [demographics,baseline,icd10diagnoses,chem]

def get_accelerometer(nrows=None):
    eid, demographics, covariates = ukbb_phenotype_dataset.create_phenotype_dataset(pheno_info.DEMOGRAPHICS,nrows=nrows,parse_dataset_covariates_kwargs={'use_genotyping_metadata':False},no_kinship=False, only_caucasians=False)
    demographics['eid'] = eid
    demographics.set_index('eid',inplace=True)
    demographics = pd.merge(demographics,_preprocess.recode_ethnicity(demographics[['ethnicity']],1001),on='eid')
    demographics = _preprocess.get_birthdate(demographics)

    eid, baseline, covariates = ukbb_phenotype_dataset.create_phenotype_dataset(pheno_info.ASSESSMENTS,nrows=nrows,parse_dataset_covariates_kwargs={'use_genotyping_metadata':False},no_kinship=False, only_caucasians=False)
    baseline['eid'] = eid
    baseline.set_index('eid',inplace=True)
    baseline['visit'] = 0
    baseline['date_visit'] = pd.to_datetime(baseline['date_visit'],format='%Y-%m-%d',errors='coerce')

    eid, icd10diagnoses, covariates = ukbb_phenotype_dataset.create_phenotype_dataset(pheno_info.DIAGNOSESICD10,nrows=nrows,parse_dataset_covariates_kwargs={'use_genotyping_metadata':False},no_kinship=False, only_caucasians=False,code='19')
    icd10diagnoses['eid'] = eid
    
    eid, physical, covariates = ukbb_phenotype_dataset.create_phenotype_dataset(pheno_info.PHYSICAL,nrows=nrows,parse_dataset_covariates_kwargs={'use_genotyping_metadata':False},no_kinship=False, only_caucasians=False)
    physical['eid'] = eid
    physical.set_index('eid',inplace=True)

    eid, accelerometer, covariates = ukbb_phenotype_dataset.create_phenotype_dataset(pheno_info.ACCELEROMETER,nrows,parse_dataset_covariates_kwargs={'use_genotyping_metadata':False},no_kinship=False, only_caucasians=False)
    accelerometer['eid'] = eid
    accelerometer.set_index('eid',inplace=True)
    
    return [demographics,baseline,icd10diagnoses,accelerometer,physical]

def merge_data(dfs):
    # combine datasets
    merged = reduce(lambda left,right: pd.merge(left,right,on='eid',how='outer'), dfs)
    merged = _preprocess.get_visit_age(merged).set_index('eid')
    return merged

def get_disorder(merged,name='AllCauseDementia',incident=True):
    # extract the subgroup you want to work with
    print(ukbb_paths.SAMPLES_DIR)
    df = pd.read_csv(os.path.join(ukbb_paths.SAMPLES_DIR, f'{name}.csv'))
    df = _preprocess.date_to_datetime(df)
    subgroup = pd.merge(df,merged.reset_index(),on='eid',how='left',suffixes=['_drop','']).set_index('eid')
    subgroup = subgroup.drop(columns=subgroup.filter(regex='_drop').columns)
    subgroup[name] = 1
    if incident:
        subgroup = subgroup[subgroup[f'{name}_incident']==1]
    merged.loc[subgroup.index,name] = 1
    return merged,subgroup

def get_healthy(merged,name='AllCauseDementia',drop_healthy=[]):
    if not drop_healthy:
        drop_healthy = name
    healthy = pd.read_csv(os.path.join(ukbb_paths.SAMPLES_DIR, f'healthy_not_{drop_healthy}.csv'))
    healthy = _preprocess.date_to_datetime(healthy)
    healthy = pd.merge(healthy,merged.reset_index(),on='eid',how='left',suffixes=['_drop','']).set_index('eid')
    healthy = healthy.drop(columns=healthy.filter(regex='_drop').columns)
    healthy[name] = 0
    return healthy
    
def get_healthy_disorder(merged,name,covs=['visit_age','male','TownsendDeprivationIndex'],
                     predictors=[],incident=True,exclude=[]):
    if not exclude:
        exclude = name
    # extract the subgroup you want to work with
    merged, subgroup = get_disorder(merged,name,incident=incident)
    healthy = get_healthy(merged,name=name,drop_healthy=exclude)
    # want to only consider healthy for match that do have information
    healthy = healthy.dropna(subset=covs,how='any',axis='rows')
    healthy = healthy.dropna(subset=predictors,how='all',axis='rows')
    print('people in HC and Case: ',np.intersect1d(subgroup.index,healthy.index).shape)
    healthy = healthy.loc[np.setdiff1d(healthy.index,subgroup.index)]
    print('people in HC and Case: ',np.intersect1d(subgroup.index,healthy.index).shape)
    merged_ = pd.concat([healthy,subgroup])
    # define time to diagnosis and diagnosis age for those who are not diseased (as latest as possible)
    last_update = pd.datetime(2021,3,1)
    merged_['time_to_diagnosis'] = merged_[f'{name}_age'] - merged_['visit_age']
    merged_.loc[merged_[name]==0,'time_to_diagnosis'] = (last_update - merged_.loc[merged_[name]==0,'date_visit']) /  np.timedelta64(1,'Y')
    merged_.loc[merged_[name]==0,f'{name}_age'] = (last_update - merged_.loc[merged_[name]==0,'date_birth']) /  np.timedelta64(1,'Y')
    return merged_

def get_matched(merged_clean,name,exclude=[],file=[],save=[]):
    try:
        matched_eid = pd.read_csv(os.path.join(ukbb_paths.SAMPLES_DIR, file),header=None,names=['eid'],index_col=0)
        matched_sample = merged_clean.loc[matched_eid['eid'],:]
        matched_sample.loc[matched_sample[name].isna(),name]=0
    except:
        print('matched file does not exist, so creating one')
        # subsample from healthy control
        if len(exclude)==0:
            exclude = [name]
        matched_eid,matched_sample = _preprocess.match(merged_clean,target=name,exclude=exclude)
        matched_sample.loc[matched_sample[name].isna(),name]=0
    if save:
        np.savetxt(os.path.join(ukbb_paths.SAMPLES_DIR, file), 
           matched_eid.reset_index().values.flatten().T, fmt='%d')
    return matched_sample

def get_matched_acc(merged_clean,name,exclude=[],matched_cols=[],file=[],save=[]):
    try:
        matched_eid = pd.read_csv(os.path.join(ukbb_paths.SAMPLES_DIR, file),header=None,names=['eid'],index_col=0)
        matched_sample = merged_clean.loc[matched_eid['eid'],:]
        matched_sample.loc[matched_sample[name].isna(),name]=0
    except:
        print('matched file does not exist, so creating one')
        # subsample from healthy control
        if len(exclude)==0:
            exclude = [name]
        if len(matched_cols)==0:
            matched_cols = ['visit_age','male']
        matched_eid,matched_sample = _preprocess.match_acc(merged_clean,target=name,exclude=exclude,match_cols=matched_cols)
        matched_sample.loc[matched_sample[name].isna(),name]=0
    if save:
        np.savetxt(os.path.join(ukbb_paths.SAMPLES_DIR, file), 
           matched_eid.reset_index().values.flatten().T, fmt='%d')
    return matched_sample