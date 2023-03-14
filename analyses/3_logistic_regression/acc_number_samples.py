import numpy as np
import pandas as pd
import glob
import re
import os
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn import preprocessing,pipeline,linear_model,model_selection,metrics,multiclass

import pylab as plt
import seaborn as sns
from statannot import add_stat_annotation

import pickle
import sys
sys.path.insert(1,'/scratch/c.c21013066/software/biobankAccelerometerAnalysis/accelerometer')
sys.path.insert(1,'/scratch/c.c21013066/UKBIOBANK_DataPreparation/phenotypes')
import utils
import plots
import load_modalities

import yaml

img_path = '/scratch/c.c21013066/images/ukbiobank/accelerometer/models'
model_path = '/scratch/c.c21013066/data/ukbiobank/analyses/acc_models/noOsteo'
data_path = '/scratch/c.c21013066/data/ukbiobank'

# def read_traits_file(input_path: str):
#     with open(input_path, 'r') as f:
#         traits_data = yaml.load(f, Loader=yaml.BaseLoader)
#     return traits_data

# # get data set of PD, prod PD and matched controls
# dfs = pd.read_csv(f'{data_path}/phenotypes/accelerometer/matched_all_HCnoOsteo_genebloodrisk_prodromalsigns.csv',index_col=0)
# name = 'ParkinsonDisease'
# subset = dfs[dfs['diagnosis']==name]
# covs = ['accelerometry_age','male']
# allfeatures = np.hstack([subset.columns[:38],subset.columns[38:98],covs]) 
# scale_allfeatures = np.hstack([subset.columns[:38],subset.columns[38:98],covs[:1]]) 
# #features = np.hstack(['No_wear_time_bias_adjusted_average_acceleration',covs,'PRS'])
# #scale_features = np.hstack(['No_wear_time_bias_adjusted_average_acceleration',covs[:1],'PRS'])
# targets = ['diagnosis','Status','eid']
# # train for robustness
# # include all other diseases, but only once
# unique = dfs[dfs['diagnosis']!="ParkinsonDisease"].dropna(subset=allfeatures,how='any')
# unique = unique[~unique.index.duplicated(keep='last')]
# unique = pd.concat([unique,subset])
# unique = unique[~unique.index.duplicated(keep='last')]
# unique['diag_PDHC'] = (unique['diagnosis'] == 'ParkinsonDisease').astype(int) * unique['Status'].replace(['Prodromal','Diseased','Healthy'],[0,1,0])
# unique['diag_ProdHC'] = (unique['diagnosis'] == 'ParkinsonDisease').astype(int) * unique['Status'].replace(['Prodromal','Diseased','Healthy'],[1,0,0])
# unique['diag_PDProdPopulation'] = (unique['diagnosis'] == 'ParkinsonDisease').astype(int) * unique['Status'].replace(['Prodromal','Diseased','Healthy'],[1,1,0])
# unique['diag_PDPopulation'] = (unique['diagnosis'] == 'ParkinsonDisease').astype(int) * unique['Status'].replace(['Prodromal','Diseased','Healthy'],[0,1,0])
# unique['diag_ProdPopulation'] = (unique['diagnosis'] == 'ParkinsonDisease').astype(int) * unique['Status'].replace(['Prodromal','Diseased','Healthy'],[1,0,0])
# unique['diag_ProdPopulationNoPD'] = (unique['diagnosis'] == 'ParkinsonDisease').astype(int) * unique['Status'].replace(['Prodromal','Diseased','Healthy'],[1,0,0])
# unique.loc[np.logical_and(unique['diagnosis']=='ParkinsonDisease',unique['Status']=='Diseased'),'diag_ProdPopulationNoPD'] = np.nan
# unique['diag_PDPopulationNoProd'] = (unique['diagnosis'] == 'ParkinsonDisease').astype(int) * unique['Status'].replace(['Prodromal','Diseased','Healthy'],[0,1,0])
# unique.loc[np.logical_and(unique['diagnosis']=='ParkinsonDisease',unique['Status']=='Prodromal'),'diag_PDPopulationNoProd'] = np.nan

# # get data of PRS
# traits = read_traits_file('/scratch/c.c21013066/Paper/ProdromalUKBB/resources/genetics/traits.yaml')
# traits = pd.DataFrame(traits)
# score1 = pd.read_csv(f'{data_path}/ukb52375.csv').set_index('eid')
# trait='26260-0.0'
# score_best = score1[trait]
# score1.columns = score1.columns.str.replace('-0.0','')
# PRSs = score1[traits.columns]
# PRSs.columns = traits.loc['full_name',PRSs.columns]
# genetics = PRSs.columns
# genetics_scale = genetics

# # merge data
# merged = pd.merge(unique,score_best,right_index=True,left_index=True,how='left').rename(columns={trait:'PRS'})
# merged = pd.merge(merged,PRSs,right_index=True,left_index=True,how='left',suffixes=['_drop',''])
# merged = merged.drop(columns=merged.filter(regex="_drop").columns)
# # get subset of features
# # drop features with too many nan
# drop = allfeatures[subset[allfeatures].isna().sum() > 0]
# subset = subset.drop(columns=drop)
# allfeatures = list(set(allfeatures).difference(set(drop)))
# scale_allfeatures = list(set(scale_allfeatures).difference(set(drop)))

# # add BMI etc features
# lifestyle = pd.read_csv(f'{data_path}/phenotypes/accelerometer/matched_all_HCnoOsteo_acc_riskblood_genebloodrisk.csv',index_col=0)
# lifestylePD = lifestyle[lifestyle['diagnosis']=='PD']
# lifestyle = lifestyle[lifestyle['diagnosis']!="ParkinsonDisease"].dropna(subset=allfeatures,how='any')
# lifestyle = lifestyle[~lifestyle.index.duplicated(keep='last')]
# lifestyle = pd.concat([lifestyle,lifestylePD])
# lifestyle = lifestyle[~lifestyle.index.duplicated(keep='last')]
# life_cols = lifestyle.columns[173:]
# lifestyle = lifestyle[life_cols]
# life_cols = np.hstack([lifestyle.columns[:17],'TownsendDeprivationIndex'])
# life_scale = life_cols[11:]
# blood_cols = lifestyle.columns[17:]
# blood_scale = blood_cols
# family_cols = ['family_Stroke', 'family_Diabetes', 'family_Severedepression',
#        'family_Alzheimersdiseasedementia', 'family_Parkinsonsdisease']
# family_scale = family_cols
# life_nofam_cols = np.hstack([life_cols[:6],life_cols[11:]])
# life_nofam_scale = life_nofam_cols

# prodromal = ['UrinaryIncontinence','Constipation','ErectileDysfunction','Anxiety','RBD','Hyposmia','OrthostaticHypotension',
#                 'Depression']
# prod_col = [f'{p}_beforePD' for p in prodromal]
# prod_acc = [f'{p}_beforeacc' for p in prodromal]

# merged = pd.merge(merged,lifestyle,right_index=True,left_index=True,how='left',suffixes=['_drop',''])
# merged = merged.drop(columns=merged.filter(regex="_drop").columns)

merged = pd.read_csv(f'{data_path}/merged_data/unaffectedNoOsteoAllHC.csv').set_index('eid')
# get modalities
covs,allfeatures,allfeatures_scale,blood,blood_scale,lifestyle,lifestyle_scale,genetics,genetics_scale,prod,prod_acc = load_modalities.load_features(f'{data_path}')

# logistic reg model
models =['diag_ProdHC','diag_PDHC','diag_PDProdHC']
#models = ['diag_PDPopulationNoProd','diag_ProdPopulationNoPD','diag_PDProdPopulation']

fnames = ['all_acc_features+blood+lifestyle+genetics+prodromalsigns_beforeacc','prodromalsigns_beforeacc',
          'all_acc_features+blood+lifestyle+genetics+prodromalsigns_beforePD','prodromalsigns_beforePD',
          'genetics+family','lifestyle_nofam','blood','PRS','acc','all_acc_features','covariates',
         'genetics+family+all_acc_features','lifestyle+all_acc_features','blood+all_acc_features','prodromalsigns_beforePD+all_acc_features']
cols = [np.hstack([allfeatures,blood,lifestyle,genetics,prod_acc]),np.hstack([covs,prod_acc]),
        np.hstack([allfeatures,blood,lifestyle,genetics,prod]),np.hstack([covs,prod]),
        np.hstack([covs,genetics]),np.hstack([covs,lifestyle]),
       np.hstack([covs,blood]),np.hstack([covs,"Parkinson's disease"]),np.hstack(['No_wear_time_bias_adjusted_average_acceleration',covs]),
       allfeatures,covs,
       np.hstack([genetics,allfeatures]),np.hstack([lifestyle,allfeatures]),
       np.hstack([blood,allfeatures]),np.hstack([prod,allfeatures])]
scale_cols = [np.hstack([allfeatures_scale,blood_scale,lifestyle_scale,genetics_scale]),covs[:1],
              np.hstack([allfeatures_scale,blood_scale,lifestyle_scale,genetics_scale]),covs[:1],
              np.hstack([covs[:1],genetics_scale]), 
             np.hstack([covs[:1],lifestyle_scale]),np.hstack([covs[:1],blood_scale]),
             np.hstack([covs[:1],"Parkinson's disease"]),
              np.hstack(['No_wear_time_bias_adjusted_average_acceleration',covs[:1]]),allfeatures_scale,
             covs[:1],
            np.hstack([genetics_scale,allfeatures_scale]),np.hstack([lifestyle_scale,allfeatures_scale]),
       np.hstack([blood_scale,allfeatures_scale]),allfeatures_scale]

features_all = np.hstack([allfeatures,blood,lifestyle,genetics,prod])

# # get population cohort which uses all available HC
# hc = pd.read_csv(f'{data_path}/phenotypes/accelerometer/allHCnoOsteo_prodromalsigns.csv',
#                       index_col=0)
# hc = pd.merge(hc,score_best,right_index=True,left_index=True,how='left').rename(columns={trait:'PRS'})
# hc = pd.merge(hc,PRSs,right_index=True,left_index=True,how='left',suffixes=['_drop',''])
# hc = hc.drop(columns=hc.filter(regex="_drop").columns)
# hc[models] = 0
# merged = pd.concat([merged[np.hstack([models,features_all])],hc[np.hstack([models,features_all])]])
# merged = merged.drop_duplicates(keep='first')

for stacked in ['_stacked']:
    overview = pd.DataFrame(columns=['cases','controls','features','N features'],
                                   index=pd.MultiIndex.from_product([models,fnames],names=['model','feature']))
    print(overview)
    for fname,features in zip(fnames,cols):
        for y in models:
            print(f'run model for {y} with features {fname}')
            if stacked=='_stacked':
                nona = merged.dropna(subset=np.hstack([y,features_all]),how='any',axis='rows')
            else:
                nona = merged.dropna(subset=np.hstack([y,features]),how='any',axis='rows')
            #samples.loc[(y,fname),'total'] = nona.shape[0]
            #samples.loc[(y,fname),'cases'] = nona[y].sum()
            #samples.loc[(y,fname),'controls'] = nona.shape[0] - nona[y].sum()
            overview.loc[(y,fname),'cases'] = nona[y].sum()
            overview.loc[(y,fname),'controls'] = nona.shape[0] - nona[y].sum()
            overview.loc[(y,fname),'features'] = features
            overview.loc[(y,fname),'N features'] = len(features)           
        
    #samples.to_csv(f'{model_path}/popmodels_samples.csv')
    print(overview)
    overview.to_csv(f'{model_path}/AllHCmodelsNoPD_overview{stacked}.csv')
    
