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
import missingno as msn
from lifelines.plotting import plot_lifetimes
from sksurv.datasets import load_veterans_lung_cancer
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.preprocessing import OneHotEncoder
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored,cumulative_dynamic_auc

import pickle
import sys
sys.path.insert(1,'/scratch/c.c21013066/software/biobankAccelerometerAnalysis/accelerometer')
#sys.path.insert(1,'/scratch/c.c21013066/UKBIOBANK_DataPreparation/phenotypes')
import utils
import load_modalities

import yaml

img_path = '/scratch/c.c21013066/images/ukbiobank/accelerometer/models'
model_path = '/scratch/c.c21013066/data/ukbiobank/analyses/acc_models/noOsteo'
data_path = '/scratch/c.c21013066/data/ukbiobank'

covs,allfeatures,allfeatures_scale,blood,blood_scale,lifestyle,lifestyle_scale,genetics,genetics_scale,prod,prod_acc = load_modalities.load_features(f'{data_path}')

# logistic reg model
models =['diag_ProdHC','diag_PDHC','diag_PDProdHC']

fnames = ['intercept',
          'genetics+family','lifestyle_nofam','blood','PRS','acc','all_acc_features','covariates','prodromalsigns_beforePD',
         'genetics+family+all_acc_features','lifestyle+all_acc_features','blood+all_acc_features','prodromalsigns_beforePD+all_acc_features',
'all_acc_features+blood+lifestyle+genetics+prodromalsigns_beforePD']
cols = [['Intercept'],
        np.hstack([covs,genetics]),np.hstack([covs,lifestyle]),
       np.hstack([covs,blood]),np.hstack([covs,"Parkinson's disease"]),np.hstack(['No_wear_time_bias_adjusted_average_acceleration',covs]),
       allfeatures,covs,np.hstack([covs,prod]),
       np.hstack([genetics,allfeatures]),np.hstack([lifestyle,allfeatures]),
       np.hstack([blood,allfeatures]),np.hstack([prod,allfeatures]),np.hstack([allfeatures,blood,lifestyle,genetics,prod])]
scale_cols = [[],
              np.hstack([covs[:1],genetics_scale]), 
             np.hstack([covs[:1],lifestyle_scale]),np.hstack([covs[:1],blood_scale]),
             np.hstack([covs[:1],"Parkinson's disease"]),
              np.hstack(['No_wear_time_bias_adjusted_average_acceleration',covs[:1]]),allfeatures_scale,
             covs[:1],covs[:1],
            np.hstack([genetics_scale,allfeatures_scale]),np.hstack([lifestyle_scale,allfeatures_scale]),
       np.hstack([blood_scale,allfeatures_scale]),allfeatures_scale,np.hstack([allfeatures_scale,blood_scale,lifestyle_scale,genetics_scale])]

features_all = np.hstack([allfeatures,blood,lifestyle,genetics,prod])

for kind,length in zip(['_matched','_allHC'],[200,6000]):
    if kind == '_matched':
        merged = pd.read_csv(f'{data_path}/merged_data/unaffectedNoOsteoMatchedHC.csv').set_index('eid')
    elif kind == '_allHC':
        merged = pd.read_csv(f'{data_path}/merged_data/unaffectedNoOsteoAllHC.csv').set_index('eid')
    else:
        print('undefined condition')
        break
    for stacked in ['_stacked']:
        performance_features = []
        results_features = []
        params_features = []
        preds_features = []
        for fname,features,scale_features in zip([fnames[0]],[cols[0]],[scale_cols[0]]):
            if fname == 'intercept':
                merged['Intercept'] = 1
                fit_intercept = False
            else:
                fit_intercept = True
            performance = pd.DataFrame(columns=pd.MultiIndex.from_product([['train','test'],['ROCAUC','accuracy',
                                                                                             'precision recall AUC','fpr','tpr','recall',
                                                                                             'precision','C']],names=['kind','metric']),
                                       index=pd.MultiIndex.from_product([models,np.arange(10)],names=['model','cv_fold'])) 
            results = []
            params = pd.DataFrame(columns=features,index=pd.MultiIndex.from_product([models,np.arange(10)],names=['model','cv_fold']))
            preds = pd.DataFrame(index=pd.MultiIndex.from_product([merged.index,np.arange(10)],
                                                                  names=['eid','cv_fold']),
                                 columns=pd.MultiIndex.from_product([models,['train','test']],names=['model','kind']))
            curve = pd.DataFrame(index=pd.MultiIndex.from_product([models,np.arange(5)],names=['model','cv']),
                            columns=pd.MultiIndex.from_product([['tpr','fpr','recall','precision'],np.arange(length)]))
            for y in models:
                nona = merged.dropna(subset=[y])
                print(f'run model for {y} with features {fname}')
                # outer split for hold-out dataset
                outer_cv = model_selection.StratifiedKFold(n_splits=5,random_state=44,shuffle=True)
                for fold,(train_id,test_id) in enumerate(outer_cv.split(nona[features],nona[y])):
                    X_train = nona.iloc[train_id][features]
                    X_test = nona.iloc[test_id][features]
                    y_train = nona.iloc[train_id][y]
                    y_test = nona.iloc[test_id][y]
                    if len(scale_features) >0:
                        scaler = preprocessing.StandardScaler().fit(X_train[scale_features])
                        X_train[scale_features] = scaler.transform(X_train[scale_features])
                        X_test[scale_features] = scaler.transform(X_test[scale_features])
                    # inner CV to find best alpha parameter for Lasso
                    Cs = np.logspace(-5, 4, 10)
                    logregcv = linear_model.LogisticRegressionCV(cv=model_selection.StratifiedKFold(n_splits=5, random_state=3,shuffle=True),
                                                                                                                       Cs=Cs,penalty='l1',
                                                                                                                       solver='saga',refit=True,max_iter=1000,scoring='average_precision',class_weight='balanced',n_jobs=10,fit_intercept=fit_intercept).fit(X_train,y_train)
                    results.append(logregcv)
                    performance.loc[(y,fold),('test','C')] = logregcv.C_
                    performance.loc[(y,fold),('train','C')] = logregcv.C_
                    
                    # evaluate test
                    y_pred_proba = logregcv.predict_proba(X_test)[::,1]
                    preds.loc[(X_test.index,fold),(y,'test')] = y_pred_proba
                    auc = metrics.roc_auc_score(y_test, y_pred_proba)
                    performance.loc[(y,fold),('test','ROCAUC')] = auc
                    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
                    performance.loc[(y,fold),('test','fpr')] = fpr
                    performance.loc[(y,fold),('test','tpr')] = tpr
                    pre,rec,_ = metrics.precision_recall_curve(y_test,y_pred_proba)
                    performance.loc[(y,fold),('test','precision')] = pre
                    performance.loc[(y,fold),('test','recall')] = rec
                    performance.loc[(y,fold),('test','precision recall AUC')] = metrics.average_precision_score(y_test,y_pred_proba)
                    curve.loc[(y,cv),('tpr',slice(None))] = np.append(tpr, np.zeros(length-len(tpr)) + np.nan)
                    curve.loc[(y,cv),('fpr',slice(None))]= np.append(fpr, np.zeros(length-len(fpr)) + np.nan)
                    curve.loc[(y,cv),('recall',slice(None))]= np.append(rec, np.zeros(length-len(rec)) + np.nan)
                    curve.loc[(y,cv),('precision',slice(None))]= np.append(pre, np.zeros(length-len(pre)) + np.nan)
                    # evaluate train
                    y_pred_proba = logregcv.predict_proba(X_train)[::,1]
                    preds.loc[(X_train.index,fold),(y,'train')] = y_pred_proba
                    fpr, tpr, _ = metrics.roc_curve(y_train,  y_pred_proba)
                    auc = metrics.roc_auc_score(y_train, y_pred_proba)
                    performance.loc[(y,fold),('train','ROCAUC')] = auc
                    performance.loc[(y,fold),('train','fpr')] = fpr
                    performance.loc[(y,fold),('train','tpr')] = tpr
                    pre,rec,_ = metrics.precision_recall_curve(y_train,y_pred_proba)
                    performance.loc[(y,fold),('train','precision')] = pre
                    performance.loc[(y,fold),('train','recall')] = rec
                    performance.loc[(y,fold),('train','precision recall AUC')] = metrics.average_precision_score(y_train,y_pred_proba)

                    params.loc[(y,fold),:] = logregcv.coef_

                    # test on external data
                    if y == 'diag_PDHC' or y =='diag_ProdHC':
                        if y == 'diag_PDHC':
                            external_test = merged[merged['Status']=='Prodromal']
                        elif y =='diag_ProdHC':
                            external_test = merged[merged['Status']=='Diseased']
#                         if stacked=='_stacked':
#                             external_test = external_test.dropna(subset=features_all)[features]
#                         else:
#                             external_test = external_test.dropna(subset=features)[features]
                        if len(scale_features) >0:
                            external_test[scale_features] = scaler.transform(external_test[scale_features])
                        y_pred_proba = logregcv.predict_proba(external_test[features])[::,1]
                        preds.loc[(external_test.index,fold),(y,'test')] = y_pred_proba


                try:
                    os.mkdir(f'{model_path}/{y}_{fname}')
                except OSError as error:
                    print(error)
                try:
                    os.mkdir(f'{model_path}/{fname}')
                except OSError as error:
                    print(error) 
                pickle.dump(results[0], open(f'{model_path}/{y}_{fname}/result{stacked}{kind}.sav', 'wb'))
                performance.to_csv(f'{model_path}/{fname}/HCmodels_perf{stacked}{kind}.csv')
                params.to_csv(f'{model_path}/{fname}/HCmodels_params{stacked}{kind}.csv')
                preds.to_csv(f'{model_path}/{fname}/HCmodels_pred{stacked}{kind}.csv')
                curve.to_csv(f'{model_path}/{fname}/HCmodels_curves{stacked}{kind}.csv')
            performance.to_csv(f'{model_path}/{fname}/HCmodels_perf{stacked}{kind}.csv')
            params.to_csv(f'{model_path}/{fname}/HCmodels_params{stacked}{kind}.csv')
            preds.to_csv(f'{model_path}/{fname}/HCmodels_pred{stacked}{kind}.csv')
            curve.to_csv(f'{model_path}/{fname}/HCmodels_curves{stacked}{kind}.csv')

            performance_features.append(performance)
            results_features.append(results)
            params_features.append(params)
            preds_features.append(preds)
