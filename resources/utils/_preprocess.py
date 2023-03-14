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

ukbb_paths = ukbb_parser._load_ukbb_paths()

# various functions to make life easier

def get_codes_ATC(atc_code=['L04A','M01A']):
    atc = pd.read_excel(os.path.join(ukbb_paths.CODINGS_DIR, 'ukbb_ATC_to_coding4.xls'),sheet_name='Supplementary Data 1',skiprows=1,usecols=np.arange(4),
                       dtype={'Coding a':str})
    match = atc['Medication ATC code'].str.contains(atc_code[0])
    if len(atc_code) > 1:
        for code in atc_code[1:]:
            match = match | atc['Medication ATC code'].str.contains(code)
    return atc.loc[match]

def extract_medication(df,atc_code=['L04A','M01A'],name='immunosuppressants'):
    selected_med = get_codes_ATC(atc_code)
    codes = np.unique(selected_med['Coding a'])
    return df.isin(codes).astype(int).max(axis=1).rename(name)

def combine_assessments(dfs,assessments):
    """Takes list of dataframes and list of corresponding assessments info and combines to one long format dataframe"""
    for i,df in enumerate(dfs):
        df = merge_with_assessment(df,assessments[i])
    return dfs

def date_to_datetime(df,columns=[]):
    if columns:
        dates = columns
    else:
        dates = df.filter(regex='^date').columns
    df[dates] = df[dates].apply(pd.to_datetime)
    return df

def date_to_datetime_end(df,columns=[]):
    if columns:
        dates = columns
    else:
        dates = df.filter(regex='date$').columns
    df[dates] = df[dates].apply(pd.to_datetime)
    return df

def make_categorical(df,scale,levels):
    for sc in scale:
        df[sc] = pd.Categorical(df[sc],categories=levels)
    return df

def merge_with_assessment(df,assessment):
    """Merge extracted info about assessment (date,site) with info about phenotype"""
    return pd.merge(df,assessment,right_index=True,left_index=True)

def get_birthdate(df):
    df['date_birth'] = pd.to_datetime(df['year_birth'], format='%Y',errors='coerce')
    df.loc[~df['month_birth'].isna(),'date_birth'] = pd.to_datetime('15' + 
            df.loc[~df['month_birth'].isna(),'month_birth'].astype(int).astype(str) + 
            df.loc[~df['month_birth'].isna(),'year_birth'].astype(int).astype(str), format='%d%m%Y',errors='coerce')
    return df

def get_visit_age(df):
    df['visit_age'] = (df['date_visit'] - df['date_birth']) / np.timedelta64(1,'Y')
    return df

def get_diagnosis_age(df,diags=['ParkinsonDisease'],source=['hospital','death']):
    if len(source) > 0:
        for diag in diags:
            mini = df[[f"{diag}_{s}_date" for s in source]].min(axis=1)
            df[f'{diag}_age'] = (mini - df['date_birth']) / np.timedelta64(1,'Y')
    else:
        for diag in diags:
            df[f'{diag}_age'] = (df[f"{diag}_date"] - df['date_birth']) / np.timedelta64(1,'Y')
    return df

def extract_prevalent(df,diag='ParkinsonDisease'):
    return df[df[f'{diag}_age']<=df['visit_age']]

def extract_incident(df,diag='ParkinsonDisease'):
    return df[df[f'{diag}_age']>df['visit_age']]

def extract_disorder(merged,icd10,icd9,selfes,names,save='/scratch/c.c21013066/data/ukbiobank/sample',covariates=['male']):
    '''given info from each source (self, icd10, icd9) combine them into one comprehensive file and store it. Will hold info about source (hospital, self, death), age of diagnosis, prevalent or incident'''
    for c10,c9,sel,name in zip(icd10,icd9,selfes,names):
        merged_ = merged.copy(deep=True)
        print(c10,c9,sel,name)
        if c9 == 'fillicd9':
            merged[c9] = np.nan
            merged[[f"{c9}_{i}" for i in ['age','hospital','hospital_date']]] = np.nan
        if sel == 'fillself':
            merged[sel] = np.nan
            merged[f"{sel}_age"] = np.nan
        merged_ = merged.loc[merged[[c10,c9,sel]].max(axis=1)==1,
                             np.hstack([[f'{c10}_age', f'{c9}_age', 'visit_age', f'{c10}_death_date', f'{c10}_hospital_date', f'{c9}_hospital_date', 'date_visit',
                                f'{sel}', f'{sel}_age', f'{c10}', f'{c9}', f'{c10}_death', f'{c10}_hospital', f'{c9}_hospital'],
                                       covariates])]
        not_disorder = merged.loc[~merged.index.isin(merged_.index),
                                       np.hstack(['visit_age', 'date_visit',
                                sel,c10,c9,covariates])]
        merged_[[f"{name}_{i}" for i in ['incident','prevalent','death','selfreported','hospital',
                                       'death_only','selfreported_only']]] = np.nan
        merged_.loc[merged_[f'{c10}_death']==1,f'{name}_death'] = 1
        merged_.loc[merged_[f'{c10}_death']==0,f'{name}_death'] = 0
        merged_.loc[merged_[f'{sel}']==1,f'{name}_selfreported'] = 1
        merged_.loc[merged[f'{sel}']==0,f'{name}_selfreported'] = 0
        merged_.loc[merged_[[f'{c10}_hospital',f'{c9}_hospital']].max(axis=1)==1,f'{name}_hospital'] = 1
        merged_.loc[merged_[[f'{c10}_hospital',f'{c9}_hospital']].max(axis=1)==0,f'{name}_hospital'] = 0
        merged_.loc[merged_[[f'{name}_hospital',f'{name}_selfreported']].max(axis=1)==1,f'{name}_death_only'] = 0
        merged_.loc[np.logical_and(np.logical_or(merged_[[f'{name}_hospital',f'{name}_selfreported']].max(axis=1)==0,
                                  merged_[[f'{name}_hospital',f'{name}_selfreported']].isna().all(axis=1)),
                                   merged_[f'{name}_death']== 1),f'{name}_death_only'] = 1
        merged_.loc[merged_[[f'{name}_death',f'{name}_hospital']].max(axis=1)==1,f'{name}_selfreported_only'] = 0
        merged_.loc[np.logical_and(np.logical_or(merged_[[f'{name}_hospital',f'{name}_death']].max(axis=1)==0,
                                  merged_[[f'{name}_hospital',f'{name}_death']].isna().all(axis=1)),
                                   merged_[f'{name}_selfreported']== 1),f'{name}_selfreported_only'] = 1
        merged_[f'{name}_age'] = merged_[[f'{c10}_age',f'{c9}_age',f'{sel}_age']].min(axis=1)
        merged_.loc[extract_incident(merged_,name).index,f'{name}_incident'] = 1
        merged_.loc[extract_prevalent(merged_,name).index,f'{name}_prevalent'] = 1
        if save:
            merged_.to_csv(f'{save}/{name}.csv')
            not_disorder.to_csv(f'{save}/healthy_not_{name}.csv')
            
def extract_disorder_withGP(merged,icd10,icd9,selfes,gps,names,save='/scratch/c.c21013066/data/ukbiobank/sample',covariates=['male']):
    '''given info from each source (self, icd10, icd9) combine them into one comprehensive file and store it. Will hold info about source (hospital, self, death), age of diagnosis, prevalent or incident'''
    for c10,c9,sel,gp,name in zip(icd10,icd9,selfes,gps,names):
        merged_ = merged.copy(deep=True)
        print(c10,c9,sel,gp,name)
        if c9 == 'fillicd9':
            merged[c9] = np.nan
            merged[[f"{c9}_{i}" for i in ['age','hospital','hospital_date']]] = np.nan
        if sel == 'fillself':
            merged[sel] = np.nan
            merged[f"{sel}_age"] = np.nan
        if gp == 'fillgp':
            merged[gp] = np.nan
            merged[f"{gp}_date"] = np.nan
            merged[f"{gp}_age"] = np.nan
        merged_ = merged.loc[merged[[c10,c9,sel,gp]].max(axis=1)==1,
                             np.hstack([[f'{c10}_age', f'{c9}_age', 'visit_age', f'{c10}_death_date', f'{c10}_hospital_date', f'{c9}_hospital_date', 'date_visit',
                                f'{sel}', f'{sel}_age', f'{c10}', f'{c9}', f'{c10}_death', f'{c10}_hospital', f'{c9}_hospital',f'{gp}_date',f'{gp}_age',gp],
                                       covariates])]
        not_disorder = merged.loc[~merged.index.isin(merged_.index),
                                       np.hstack(['visit_age', 'date_visit',
                                sel,c10,c9,gp,covariates])]
        merged_[[f"{name}_{i}" for i in ['incident','prevalent','death','selfreported','hospital','gp','primarycare']]] = np.nan
        merged_.loc[merged_[f'{c10}_death']==1,f'{name}_death'] = 1
        merged_.loc[merged_[f'{c10}_death']==0,f'{name}_death'] = 0
        merged_.loc[merged_[f'{sel}']==1,f'{name}_selfreported'] = 1
        merged_.loc[merged[f'{sel}']==0,f'{name}_selfreported'] = 0
        merged_.loc[merged_[f'{gp}']==1,f'{name}_primarycare'] = 1
        merged_.loc[merged[f'{gp}']==0,f'{name}_primarycare'] = 0
        merged_.loc[merged_[[f'{c10}_hospital',f'{c9}_hospital']].max(axis=1)==1,f'{name}_hospital'] = 1
        merged_.loc[merged_[[f'{c10}_hospital',f'{c9}_hospital']].max(axis=1)==0,f'{name}_hospital'] = 0
        #merged_.loc[merged_[[f'{name}_hospital',f'{name}_selfreported']].max(axis=1)==1,f'{name}_death_only'] = 0
        #merged_.loc[np.logical_and(np.logical_or(merged_[[f'{name}_hospital',f'{name}_selfreported']].max(axis=1)==0,
        #                          merged_[[f'{name}_hospital',f'{name}_selfreported']].isna().all(axis=1)),
        #                           merged_[f'{name}_death']== 1),f'{name}_death_only'] = 1
        #merged_.loc[merged_[[f'{name}_death',f'{name}_hospital']].max(axis=1)==1,f'{name}_selfreported_only'] = 0
        #merged_.loc[np.logical_and(np.logical_or(merged_[[f'{name}_hospital',f'{name}_death']].max(axis=1)==0,
        #                          merged_[[f'{name}_hospital',f'{name}_death']].isna().all(axis=1)),
        #                           merged_[f'{name}_selfreported']== 1),f'{name}_selfreported_only'] = 1
        merged_[f'{name}_age'] = merged[[f'{c10}_age',f'{c9}_age',f'{sel}_age',f'{gp}_age']][merged_[[f'{c10}_age',f'{c9}_age',f'{sel}_age',f'{gp}_age']]>=0].min(axis=1)
        merged_.loc[extract_incident(merged_,name).index,f'{name}_incident'] = 1
        merged_.loc[extract_prevalent(merged_,name).index,f'{name}_prevalent'] = 1
        if save:
            merged_.to_csv(f'{save}/{name}.csv')
            not_disorder.to_csv(f'{save}/healthy_not_{name}.csv')
    
            
def extract_healthy(merged,icd10,icd9,selfes,name,save='/scratch/c.c21013066/data/ukbiobank/sample',covariates=['male']):
    '''given info from each source (self, icd10, icd9) combine them into one comprehensive file and store it. Then grab all not in that set and store their eid as healthy'''
    for c10,c9,sel in zip(icd10,icd9,selfes):
        merged_ = merged.copy(deep=True)
        print(c10,c9,sel,name)
        if c9 == 'fillicd9':
            merged[c9] = np.nan
        if sel == 'fillself':
            merged[sel] = np.nan
    merged_ = merged.loc[merged[np.hstack([icd10,icd9,selfes])].max(axis=1)==1,
                             np.hstack(['visit_age', 'date_visit',
                                selfes,icd10, icd9,covariates])]
    healthy = merged.loc[~merged.index.isin(merged_.index),np.hstack(['visit_age', 'date_visit',
                                selfes,icd10,icd9,covariates])]
    if save:
        #merged_.to_csv(f'{save}/{name}.csv')
        #np.savetxt(f'{save}/{name}.txt', merged_.index, fmt='%d')
        healthy.to_csv(f'{save}/healthy_not_{name}.csv')
        np.savetxt(f'{save}/healthy_not_{name}.txt', healthy.index, fmt='%d')
    return healthy

def add_missing_dummy_columns(d,columns,name):
    cols = [f'{name}_{c}' for c in set(columns)]
    missing_cols = set(cols) - set(d.columns)
    for c in missing_cols:
        d[c] = 0
    return d

def match(df,target='ParkinsonDisease',exclude=['AllCauseParkinsonism'],match_cols=['visit_age_rounded','male']):
    '''match by age and gender to find healthy (do not have exclude disease or target disorder) for each patient'''
    df['visit_age_rounded'] = df['visit_age'].round(0)
    case = df[df[target]==1]
    print(case.shape)
    control = df[np.logical_and(df[target]!=1,df[exclude].sum(axis=1) == 0)]
    print(control.shape)
    no_matches = []
    eids = pd.DataFrame(index=case.index,columns=['control_match'])
    for key,row in case.iterrows():
        # find match
        try:
            match = control[(control['visit_age_rounded']==row['visit_age_rounded']) & (control['male']==row['male'])].sample(n=1)
            # append match and remove it from control pool for sampling without retaking
            eids.loc[key,'control_match'] = match.index.values[0]
            control = control[~control.index.isin(eids['control_match'])]
        except:
            print('no match found for ',key)
            eids = eids.drop(index=[key])
            no_matches.append([key])
    matched = df.loc[np.hstack([eids['control_match'],eids.index])]
    if len(no_matches)>0:
        no_matches = pd.Series(no_matches).rename('eid')
        no_matches.to_csv(os.path.join(ukbb_paths.SAMPLES_DIR,f'no_match/{target}_risk.csv'))
    return eids, matched

def match_acc(df,target='ParkinsonDisease',exclude=['AllCauseParkinsonism'],match_cols=['accelerometry_age_rounded','male']):
    '''match by age and gender to find healthy (do not have exclude disease or target disorder) for each patient'''
    df[match_cols[0]] = df[match_cols[0].replace('_rounded','')].round(0)
    case = df[df[target]==1]
    print(case.shape)
    control = df[np.logical_and(df[target]!=1,df[exclude].sum(axis=1) == 0)]
    print(control.shape)
    eids = pd.DataFrame(index=case.index,columns=['control_match'])
    no_matches = []
    for key,row in case.iterrows():
        # find match
        try:
            match = control[(control[match_cols[0]]==row[match_cols[0]]) & (control[match_cols[1]]==row[match_cols[1]])].sample(n=1)
            # append match and remove it from control pool for sampling without retaking
            eids.loc[key,'control_match'] = match.index.values[0]
            control = control[~control.index.isin(eids['control_match'])]
        except:
            print('no match found for ',key)
            eids = eids.drop(index=[key])
            no_matches.append([key])
    matched = df.loc[np.hstack([eids['control_match'],eids.index])]
    if len(no_matches)>0:
        no_matches = pd.Series(no_matches).rename('eid')
        no_matches.to_csv(os.path.join(ukbb_paths.SAMPLES_DIR,f'no_match/{target}_acc.csv'))
    return eids, matched

def subsample(df,target="Hypertension"):
    '''subsample but not matched'''
    targetsize = df[target].sum().astype(int)
    patient = df[df[target]==1]
    control = df[df[target]==0].sample(n=targetsize)
    return patient.append(control).sample(frac=1)

def get_date_diagnosis(icd10,date,codes=['G20'],collapse='Parkinson'):
    '''get from icd10 codes and corresponding dates when diagnosis was first reported for each subject
        icd10: dataframe n_subjects,223 holds for each subject each diagnosis ever given in coding19
        date: corresponds to icd10 and holds respective date
        codes: ICD10 codes of disorder to extract
        collapse: if name given, return minimal date across codes, else report for each code the date
        returns diagnosis date: dataframe n_subjects holding date when first diagnosed'''
    date.columns = icd10.columns
    date[date.columns] = date[date.columns].apply(pd.to_datetime,format='%Y-%m-%d')
    # create mask showing which subject had diagnosis at which array index and lay over similarly shaped date dataframe
    if collapse:
        diagnosis_date = pd.DataFrame(date[icd10.isin(codes)].min(axis=1),columns=[collapse])
    else:
        diagnosis_date = pd.DataFrame(columns=codes)
        for code in codes:
            diagnosis_date[code] = date[icd10==code].min(axis=1)
    return diagnosis_date

def get_diagnosis(icd10,codes=['G20'],collapse='Parkinson'):
    '''get from icd10 codes for each subject
        icd10: dataframe n_subjects,223 holds for each subject each diagnosis ever given in coding19
        codes: ICD10 codes of disorder to extract
        collapse: if name given, return minimal date across codes, else report for each code the date
        returns diagnosis: dataframe n_subjects holding whether diagnosed'''
    if collapse:
        diagnosis = pd.DataFrame(icd10.isin(codes).max(axis=1).astype(int),columns=[collapse])
    else:
        diagnosis = pd.DataFrame(columns=codes)
        for code in codes:
            diagnosis[code] = (icd10==code).max(axis=1).astype(int)
    return diagnosis

def get_icd10diagnosis_source_date(codes_list,collapse_list,nrows=None):
    eid,icd10,covariates = ukbb_parser.create_dataset(pheno_info.ICD10_FIELDS,nrows=nrows,parse_dataset_covariates_kwargs={'use_genotyping_metadata':False},
                                            no_kinship=False, only_caucasians=False,code='19')
    icd10['eid'] = eid
    icd10.set_index('eid',inplace=True)
    icd10.columns = icd10.columns.str.replace('41270', 'Diagnosis')
    icd10.columns = icd10.columns.str.replace('40006', 'Cancer')
    icd10.columns = icd10.columns.str.replace('40001', 'PrimaryDeath')
    icd10.columns = icd10.columns.str.replace('40002', 'SecondaryDeath')
    icd10.columns = icd10.columns.str.replace('41280', 'Diagnosis_date')
    icd10.columns = icd10.columns.str.replace('40005', 'Cancer_date')
    icd10.columns = icd10.columns.str.replace('40000', 'Death_date')
    
    result = pd.DataFrame(columns=pd.MultiIndex.from_product([collapse_list,['death','hospital','hospital_date','death_date']]),index=icd10.index)
    for codes,collapse in zip(codes_list,collapse_list):
        result[collapse,'hospital_date'] = get_date_diagnosis(icd10.filter(regex='Diagnosis-'),icd10.filter(regex='Diagnosis_date-'),codes=codes,collapse=collapse)
        result[collapse,'hospital'] = get_diagnosis(icd10.filter(regex='Diagnosis-'),codes=codes,collapse=collapse)
        death1 = get_diagnosis(icd10.filter(regex='PrimaryDeath-'),codes=codes,collapse=collapse)
        death2 = get_diagnosis(icd10.filter(regex='SecondaryDeath-'),codes=codes,collapse=collapse)
        result[collapse,'death'] = pd.merge(death1,death2,on='eid').max(axis=1).astype(int)
        deathdate1 = get_date_diagnosis(icd10.filter(regex='PrimaryDeath-'),icd10.filter(regex='Death_date-'),codes=codes,collapse=collapse)
        seconddeath0 = pd.Series(icd10.filter(regex='SecondaryDeath-0').isin(codes).max(axis=1).astype(int),name='SecondaryDeath-0')
        seconddeath0.loc[seconddeath0==1] = codes[0]
        seconddeath1 = pd.Series(icd10.filter(regex='SecondaryDeath-1').isin(codes).max(axis=1).astype(int),name='SecondaryDeath-1')
        seconddeath1.loc[seconddeath1==1] = codes[0]
        seconddeath = pd.merge(seconddeath0,seconddeath1,on='eid')
        deathdate2 = get_date_diagnosis(pd.merge(seconddeath0,seconddeath1,on='eid'),icd10.filter(regex='Death_date-'),codes=codes,collapse=collapse)
        result[collapse,'death_date'] = pd.merge(deathdate1,deathdate2,on='eid').min(axis=1)
    return result

def get_icd9diagnosis_source_date(codes_list,collapse_list,nrows=None):
    eid,icd9,covariates = ukbb_parser.create_dataset(pheno_info.ICD9_FIELDS,nrows=nrows,parse_dataset_covariates_kwargs={'use_genotyping_metadata':False},no_kinship=False, only_caucasians=False,code='87')
    icd9['eid'] = eid
    icd9.set_index('eid',inplace=True)
    icd9.columns = icd9.columns.str.replace('41271', 'Diagnosis')
    icd9.columns = icd9.columns.str.replace('40013', 'Cancer')
    icd9.columns = icd9.columns.str.replace('41281', 'Diagnosis_date')
    icd9.columns = icd9.columns.str.replace('40005', 'Cancer_date')
    
    result = pd.DataFrame(columns=pd.MultiIndex.from_product([collapse_list,['hospital','hospital_date']]),index=icd9.index)
    for codes,collapse in zip(codes_list,collapse_list):
        result[collapse,'hospital_date'] = get_date_diagnosis(icd9.filter(regex='Diagnosis-'),icd9.filter(regex='Diagnosis_date-'),codes=codes,collapse=collapse)
        result[collapse,'hospital'] = get_diagnosis(icd9.filter(regex='Diagnosis-'),codes=codes,collapse=collapse)
    return result

def get_selfreported_diagnoses(demographics,collapse_list=[],nrows=None,names=[]):
    '''load the datafields associated with self-reported no cancer illnesses and extract for each subject which disorder ever reported at a visit and the minimal age which was ever reported'''
    eid,selfreport,covariates = ukbb_parser.create_dataset(pheno_info.SELF_FIELDS,nrows=nrows,
                                                               parse_dataset_covariates_kwargs={'use_genotyping_metadata':False},
                                                               no_kinship=False, only_caucasians=False,code='6')
    selfreport['eid'] = eid
    selfreport.set_index('eid',inplace=True)
    # coding used for selfreport
    codes = pd.read_csv(os.path.join(ukbb_paths.CODINGS_DIR, 'coding6.tsv'),sep='\t',dtype={'coding':str})
    codes = codes.set_index('coding')
    codes = rename_codes(codes)
    codes = codes.drop(index=codes.index[codes.index=='-1'])
    if len(collapse_list) == 0:
        collapse_list = codes['meaning'].values
    if len(names) == 0:
        names = collapse_list
    # rename the columns to which kind of data they represent
    selfreport.columns = selfreport.columns.str.replace('20002', 'selfreported')
    selfreport.columns = selfreport.columns.str.replace('87', 'selfreported_date')
    selfreport.columns = selfreport.columns.str.replace('20013', 'selfreported_method')
    # prepare condensed dataframe
    result = pd.DataFrame(columns=[f'selfreported_{c}_age' for c in names],index=selfreport.index)
    dates = selfreport.filter(regex='selfreported_date-')
    method = selfreport.filter(regex='selfreported_method-')
    # sometimes report age, sometimes report date, so condense into age info
    method.columns = dates.columns
    # set the year as the date in the middle of the year (2nd July)
    years = dates[method==-5].apply(pd.to_datetime,format='%Y') + datetime.timedelta(days=182)
    years = years.apply(lambda x: (x - demographics.loc[x.index,'date_birth']) / np.timedelta64(1,'Y'))
    years[method==-4] = dates[method==-4]
    years[method.isin([-3,-1])] = np.nan
    diagnoses = selfreport.filter(regex='selfreported-')
    diagnoses.columns = dates.columns
    for name,diag in zip(names,collapse_list):
        result[f'selfreported_{name}'] = (diagnoses==codes[codes['meaning']==diag].index.values[0]).astype(int).max(axis=1)
        result[f'selfreported_{name}_age'] = years[diagnoses==codes[codes['meaning']==diag].index.values[0]].min(axis=1)
    result[diagnoses.isna().all(axis=1)] = np.nan
    return result

def clean_predictors(merged_clean,predictors,scale_predictors,
                     thresh=0.2):
    #drop columns with too many nan
    pred_drop = merged_clean[predictors].isna().sum(axis=0) > int(thresh * merged_clean.shape[0])
    print("Drop these predictors as too many NaN ",predictors[pred_drop])
    predictors = predictors[~pred_drop]
    scale_predictors = scale_predictors[~pred_drop]
    return predictors, scale_predictors

def clean_subjects(merged_clean,predictors,thresh=0):
    # drop subjects with any NaN
    subj_drop = merged_clean[predictors].isna().sum(axis=1) > int(thresh * len(predictors))
    print('Subjects get dropped due to too many NaN ',subj_drop.sum())
    merged_clean = merged_clean.loc[~subj_drop]
    return merged_clean

def drop_low_frequency_predictors(df,predictors,scale_predictors,perc=0.05):
    drop = []
    drop_idx = []
    for idx,predictor in enumerate(predictors):
        if ~(scale_predictors[idx]):
            if df[predictor].value_counts(normalize=True).loc[1] < perc:
                drop.append(predictor)
                drop_idx.append(idx)
    return drop,drop_idx

def recode_family(df):
    codes = pd.read_csv(os.path.join(ukbb_paths.CODINGS_DIR, 'coding1010.tsv'),sep='\t')
    codes = codes.set_index('coding')
    codes = rename_codes(codes)
    # define those that are healthy
    orig_cols = df.columns
    for c in orig_cols:
        df.loc[df[c]<0,c] = np.nan
    df = df.replace(codes.to_dict(orient='dict')['meaning'])
    dummy = pd.get_dummies(df,columns=df.columns)
    for illness in codes['meaning'][6:]:
        df[f'family_{illness}'] = dummy.loc[:, [x for x in dummy.columns if x.endswith(illness)]].max(axis=1)
        # set those that do not have info or did not give to nan
        df.loc[df[orig_cols].isna().all(axis=1),f'family_{illness}'] = np.nan
    return df.filter(regex='family_')

def recode_pesticides(df,coding):
    """standard recoding: set negative to nan and pivot to categories"""
    codes = pd.read_csv(os.path.join(ukbb_paths.CODINGS_DIR, f'coding{coding}.tsv'),sep='\t')
    codes = codes.set_index('coding')
    codes = rename_codes(codes)
    df = df.replace(codes.to_dict(orient='dict')['meaning'])
    dummy = pd.get_dummies(df,columns=df.columns)
    dummy = add_missing_dummy_columns(dummy,codes['meaning'],name='Pesticides')
    dummy['Pesticides_Yes'] = np.logical_or(dummy['Pesticides_Often'],dummy['Pesticides_Sometimes']).astype(int)
    dummy.loc[np.logical_or(df['Pesticides']=='Do not know',df['Pesticides'].isna()),dummy.columns] = np.nan
    return dummy

def rename_codes(codes):
    codes['meaning'] = codes['meaning'].str.replace(' ','')
    codes['meaning'] = codes['meaning'].str.replace('-','_')
    codes['meaning'] = codes['meaning'].str.replace(',','')
    codes['meaning'] = codes['meaning'].str.replace('/','')
    codes['meaning'] = codes['meaning'].str.replace("'",'')
    return codes

def recode_not_dummyencode(df,coding,nona=True):
    codes = pd.read_csv(os.path.join(ukbb_paths.CODINGS_DIR, f'coding{coding}.tsv'),sep='\t')
    codes = codes.set_index('coding')
    codes = rename_codes(codes)
    # define those that are healthy
    if nona:
        codes = codes.drop(index=codes.index[codes.index<0])
        for c in df.columns:
            df.loc[df[c]<0,c] = np.nan
    df = df.replace(codes.to_dict(orient='dict')['meaning'])
    return df
    
def recode(df,coding,nona=True):
    """standard recoding: set negative to nan and pivot to categories"""
    codes = pd.read_csv(os.path.join(ukbb_paths.CODINGS_DIR, f'coding{coding}.tsv'),sep='\t')
    codes = codes.set_index('coding')
    codes = rename_codes(codes)
    # define those that are healthy
    if nona:
        codes = codes.drop(index=codes.index[codes.index<0])
        for c in df.columns:
            df.loc[df[c]<0,c] = np.nan
    df = df.replace(codes.to_dict(orient='dict')['meaning'])
    dummy = pd.get_dummies(df,columns=df.columns)
    dummy = add_missing_dummy_columns(dummy,codes['meaning'],name=df.columns[0].split('_')[0])
    for c in df.columns:
        dummy.loc[df[c].isna(),dummy.filter(regex=c).columns] = np.nan
    return dummy

def recode_ethnicity(df,coding=1001):
    '''extract who is of white race'''
    import math
    codes = pd.read_csv(os.path.join(ukbb_paths.CODINGS_DIR, f'coding{coding}.tsv'),sep='\t')
    codes = codes.set_index('coding')
    codes = rename_codes(codes)
    codes = codes.drop(index=codes.index[codes.index<0])
    for c in df.columns:
        df.loc[df[c]<0,c] = np.nan
        #df.loc[~(df[c].isna()),c] = df.loc[~(df[c].isna()),c].apply(lambda d: d // 10 ** (int(math.log(d, 10)) - 1))
        df.loc[~(df[c].isna()),c] = df.loc[~(df[c].isna()),c].replace(codes.to_dict(orient='dict')['meaning'])
    dummy = pd.get_dummies(df,columns=df.columns)
    for c in df.columns:
        dummy.loc[df[c].isna(),dummy.filter(regex=c).columns] = np.nan
    return pd.merge(dummy,df,on='eid')