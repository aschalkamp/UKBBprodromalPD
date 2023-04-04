import numpy as np
import pandas as pd
import glob
import re
import os
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn import preprocessing,pipeline,linear_model,model_selection,metrics,multiclass,inspection

import pylab as plt
import seaborn as sns
from statannot import add_stat_annotation
import missingno as msn
from lifelines.plotting import plot_lifetimes
from sksurv.datasets import load_veterans_lung_cancer
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.preprocessing import OneHotEncoder
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored,cumulative_dynamic_auc
from sksurv import ensemble

import pickle
import joblib
import sys
sys.path.insert(1,'/scratch/c.c21013066/software/biobankAccelerometerAnalysis/accelerometer')
sys.path.insert(1,'/scratch/c.c21013066/UKBIOBANK_DataPreparation/phenotypes')
sys.path.insert(1,'/scratch/c.c21013066/Paper/ProdromalUKBB/analyses/3_logistic_regression')
import utils
import plots
import load_modalities
import datetime

import yaml

img_path = '/scratch/c.c21013066/images/ukbiobank/accelerometer/models'
model_path = '/scratch/c.c21013066/data/ukbiobank/analyses/survival/prodromal/noOsteo'
data_path = '/scratch/c.c21013066/data/ukbiobank'

covs,allfeatures,allfeatures_scale,blood,blood_scale,lifestyle,lifestyle_scale,genetics,genetics_scale,prod,prod_acc = load_modalities.load_features(f'{data_path}')

models =['diag_ProdHC']

fnames = ['intercept','covariates',
          'genetics+family','lifestyle_nofam','blood','acc','all_acc_features','prodromalsigns_beforePD',
         'genetics+family+all_acc_features','lifestyle+all_acc_features','blood+all_acc_features','prodromalsigns_beforePD+all_acc_features',
'all_acc_features+blood+lifestyle+genetics+prodromalsigns_beforePD','prodromalsigns_beforeacc','prodromalsigns_beforeacc+all_acc_features',
         'all_acc_features+blood+lifestyle+genetics+prodromalsigns_beforeacc']
cols = [['Intercept'],np.hstack(['Intercept',covs]),
        np.hstack([covs,genetics,'Intercept']),np.hstack([covs,lifestyle,'Intercept']),
       np.hstack([covs,blood,'Intercept']),np.hstack(['No_wear_time_bias_adjusted_average_acceleration',covs,'Intercept']),
       np.hstack([allfeatures,'Intercept']),np.hstack([covs,prod,'Intercept']),
       np.hstack([genetics,allfeatures,'Intercept']),np.hstack([lifestyle,allfeatures,'Intercept']),
       np.hstack([blood,allfeatures,'Intercept']),np.hstack([prod,allfeatures,'Intercept']),np.hstack([allfeatures,blood,lifestyle,genetics,prod,'Intercept']),
       np.hstack([covs,prod_acc,'Intercept']),np.hstack([prod_acc,allfeatures,'Intercept']),np.hstack([allfeatures,blood,lifestyle,genetics,prod_acc,'Intercept'])]
scale_cols = [[],covs[:1],
              np.hstack([covs[:1],genetics_scale]), 
             np.hstack([covs[:1],lifestyle_scale]),np.hstack([covs[:1],blood_scale]),
              np.hstack(['No_wear_time_bias_adjusted_average_acceleration',covs[:1]]),allfeatures_scale,
             covs[:1],
            np.hstack([genetics_scale,allfeatures_scale]),np.hstack([lifestyle_scale,allfeatures_scale]),
       np.hstack([blood_scale,allfeatures_scale]),allfeatures_scale,np.hstack([allfeatures_scale,blood_scale,lifestyle_scale,genetics_scale]),
              covs[:1],allfeatures_scale,np.hstack([allfeatures_scale,blood_scale,lifestyle_scale,genetics_scale])]

features_all = np.hstack([allfeatures,blood,lifestyle,genetics,prod,prod_acc])

for kind in ['_matched','_allHC']:
    if kind == '_matched':
        merged = pd.read_csv(f'{data_path}/merged_data/unaffectedNoOsteoMatchedHC.csv').set_index('eid')
    elif kind == '_allHC':
        merged = pd.read_csv(f'{data_path}/merged_data/unaffectedNoOsteoAllHC.csv').set_index('eid')
        prodage = pd.read_csv(f'{data_path}/merged_data/unaffectedNoOsteoMatchedHC.csv').set_index('eid')
        merged.loc[prodage[prodage['diag_ProdHC']==1].index,'acc_time_to_diagnosis'] = prodage.loc[prodage['diag_ProdHC']==1,'acc_time_to_diagnosis'].values
    else:
        print('undefined condition')
        break
    merged.loc[merged['diag_ProdHC']==0,'acc_time_to_diagnosis'] = (pd.Timestamp(datetime.datetime(2021,3,1)) - pd.to_datetime(merged.loc[merged['diag_ProdHC']==0,'date_accelerometry']) )/ np.timedelta64(1,'Y')
    merged['Intercept'] = 1
    print(merged.groupby('diag_ProdHC')['acc_time_to_diagnosis'].agg(['min','max']))
#     eids = pd.DataFrame(index=merged[np.logical_and(merged[features_all].isna().sum(axis=1)>1,merged['Status'].isin(['Prodromal','Diseased']))].index,
#                          columns=['control_match'])
#     match_cols = ['accelerometry_age','male']
#     for key,row in merged[merged[features_all].isna().sum(axis=1)>1].iterrows():
#         control = merged[np.logical_and(merged['diagnosis']==row['diagnosis'],merged['Status']=='Healthy')]
#         if row['Status']=='Prodromal':
#             control = control[control['Group']=='Healthy_Prodromal']
#         elif row['Status']=='Diseased':
#             control = control[control['Group']=='Healthy_Diseased']
#         print(key)
#         match = control[(control[match_cols[0]].round()==np.round(row[match_cols[0]])) & (control[match_cols[1]]==row[match_cols[1]])].sample(n=1)
#         print(match.index)
#         eids.loc[key,'control_match'] = match.index.values[0]
#         control = control[~control.index.isin(eids['control_match'])]
#         merged = merged.drop(index=match.index)

    for name in models:
        df = merged.dropna(subset=[name])

        for features,scale,fname in zip([cols[13]],[scale_cols[13]],[fnames[13]]):
            try:
                os.mkdir(f'{model_path}/{fname}')
            except OSError as error:
                print(error)
            try:
                os.mkdir(f'{model_path}/{fname}/{name}')
            except OSError as error:
                print(error)
            # Only using the subset of the columns present in the original data
            df = df.dropna(subset=features_all,how='any',axis='rows')
            pred = np.hstack([name,'acc_time_to_diagnosis',features])

            df_r= df.loc[:,pred]
            #df_r = df_r.dropna(subset=allfeatures,how='any',axis='rows')
            outer_cv = model_selection.StratifiedKFold(n_splits=5, random_state=4,shuffle=True)
            time_points = np.arange(2.5, 7,0.1)
            print(time_points)
            #time_points = np.percentile(df["acc_time_to_diagnosis"], np.linspace(5, 81, 15))
            cph_aucs = pd.DataFrame(index=np.arange(5),columns=['mean'])
            for cv,(train_id,test_id) in enumerate(outer_cv.split(df[pred],df[name])):
                X_train = df.iloc[train_id][pred]
                X_test = df.iloc[test_id][pred]
                y_train = df.iloc[train_id][name]
                y_test = df.iloc[test_id][name]
                if len(scale)>0:
                    scaler = preprocessing.StandardScaler().fit(X_train[scale])
                    X_train[scale] = scaler.transform(X_train[scale])
                    X_test[scale] = scaler.transform(X_test[scale])
                df_dummy = X_train.copy()#pd.get_dummies(X_train ,drop_first=True)
                df_dummy[name] = df_dummy[name].astype('?')
                df_dummy['acc_time_to_diagnosis'] = df_dummy['acc_time_to_diagnosis'].astype('<f8')
                df_dummy = df_dummy.rename(columns={name:'Status','acc_time_to_diagnosis':'Survival_in_years'})
                dt=dtype=[('Status', '?'), ('Survival_in_years', '<f8')]
                data_y = np.array([tuple(row) for row in df_dummy[['Status','Survival_in_years']].values], dtype=dt)
                df_dummy_test = pd.get_dummies(X_test, drop_first=True)
                df_dummy_test[name] = df_dummy_test[name].astype('?')
                df_dummy_test['acc_time_to_diagnosis'] = df_dummy_test['acc_time_to_diagnosis'].astype('<f8')
                df_dummy_test = df_dummy_test.rename(columns={name:'Status','acc_time_to_diagnosis':'Survival_in_years'})
                data_y_test = np.array([tuple(row) for row in df_dummy_test[['Status','Survival_in_years']].values], dtype=dt)

                rsf = ensemble.RandomSurvivalForest(n_estimators=1000,
                                           min_samples_split=10,
                                           min_samples_leaf=15,
                                           n_jobs=-1,
                                           random_state=123)
                rsf.fit(df_dummy[pred[2:]], data_y)

                cph_risk_scores = rsf.predict(df_dummy_test[pred[2:]])
                cph_auc, cph_mean_auc = cumulative_dynamic_auc(
                    data_y, data_y_test, cph_risk_scores,time_points
                )
                cph_aucs.loc[cv,time_points] = cph_auc
                cph_aucs.loc[cv,'mean'] = cph_mean_auc
                joblib.dump(rsf, f'{model_path}/{fname}/{kind}modelrsf_CV{cv}.joblib') 
                pred_surv = rsf.predict_survival_function(df_dummy_test[pred[2:]],return_array=True)
                np.save(f'{model_path}/{fname}/rsf_{kind}testpred_CV{cv}.csv',pred_surv)
                df_dummy_test.to_csv(f'{model_path}/{fname}/{kind}rsftest_cv{cv}.csv')
                df_dummy_test['y_risk'] = cph_risk_scores
                df_dummy_test[['Status','Survival_in_years','y_risk']].to_csv(f'{model_path}/{fname}/{kind}rsf_testrisk_CV{cv}.csv')
                #plt.savefig('/scratch/c.c21013066/images/ukbiobank/accelerometer/models/coxphsurvival_prodhc_auc_time_test.png',bbox_inches="tight")
                #result = inspection.permutation_importance(
                #                rsf, df_dummy_test[pred[2:]], data_y_test, n_repeats=15, random_state=123)
                #features = pd.DataFrame(
                #            {k: result[k] for k in ("importances_mean", "importances_std",)},
                #            index=X_test.columns).sort_values(by="importances_mean", ascending=False)
                #features.to_csv(f'{model_path}/{fname}/rsffeatures_cv{cv}.csv')
            cph_aucs.to_csv(f'{model_path}/{fname}/{kind}rsf_aucs_5cv.csv')