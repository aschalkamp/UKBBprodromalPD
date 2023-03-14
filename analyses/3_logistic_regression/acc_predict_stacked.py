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
sys.path.insert(1,'/scratch/c.c21013066/UKBIOBANK_DataPreparation/phenotypes')
import utils
import plots
import yaml

model_path = '/scratch/c.c21013066/data/ukbiobank/analyses/acc_models/noOsteo/'
data_path = '/scratch/c.c21013066/data/ukbiobank/'

# stacked models
# retrieve single source models:
diagnames = ['PDHC','ProdHC','PDProdHC','PDallHC','ProdallHC','PDProdallHC','PDPopulationNoProd','ProdPopulationNoPD','PDProdPopulation']
diags = ['PDHC','ProdHC','PDProdHC','PDHC','ProdHC','PDProdHC','PDPopulationNoProd','ProdPopulationNoPD','PDProdPopulation']
kinds = ['HC','HC','HC','allHC','allHC','allHC','pop','pop','pop']
features = ['all_acc_features','blood','genetics+family','lifestyle_nofam','prodromalsigns_beforePD']
preds_models = []

for subset,sname in zip([['all_acc_features','blood'],['all_acc_features','genetics'],
                             ['all_acc_features','lifestyle'],['all_acc_features','prodromal_symptoms'],['all_acc_features','blood','genetics','lifestyle','prodromal_symptoms']],
                            ['blood+all_acc_features','genetics+family+all_acc_features','lifestyle+all_acc_features','prodromalsigns_beforePD+all_acc_features',
                            'all_acc_features+blood+lifestyle+genetics+prodromalsigns_beforePD']):

    perf = pd.DataFrame(index=pd.MultiIndex.from_product([diagnames,np.arange(5)],names=['model','cv']),
                                columns=pd.MultiIndex.from_product([['train','test'],['AUROC','AUPRC','fpr','tpr','precision','recall','C']],names=['kind','metric']))

    for kind,diag,diagname in zip(kinds,diags,diagnames):
        # add prediction output from single-modality models to dataframe
        for i,feature in enumerate(features):
            if kind == 'pop':
                print('population model')
                path = f'{model_path}/{feature}/{kind}modelsNoPD_pred_stacked_allHC.csv'
                popkind = '_allHC'
            elif kind=='allHC':
                print('all HC model')
                path = f'{model_path}/{feature}/HCmodels_pred_stacked_allHC.csv'
                diag = diag.replace('all','')
                popkind = '_allHC'
            elif kind=='HC':
                print('matched HC model')
                path = f'{model_path}/{feature}/{kind}models_pred_stacked_matched.csv'
                popkind = '_matched'
            print(kind,diag,diagname)
            preds = pd.read_csv(path,index_col=[0,1],header=[0,1])
            print(preds.head())
            preds = preds.loc[:,(f'diag_{diag}',slice(None))].droplevel(level=0,axis=1)
            if i == 0:
                pred_models = pd.DataFrame(columns=pd.MultiIndex.from_product([features,['train','test']],names=['model','kind']),
                                          index=preds.index)
            pred_models.loc[preds.index,(feature,'train')] = preds['train'].values
            pred_models.loc[preds.index,(feature,'test')] = preds['test'].values
        preds_models.append(pred_models)
        if kind == 'HC':
            merged = pd.read_csv(f'{data_path}/merged_data/unaffectedNoOsteoMatchedHC.csv').set_index('eid')
            #merged = merged.dropna(subset=allfeatures,axis='rows',how='any')
            merged_pred = pd.merge(merged,pred_models,left_index=True,right_index=True,how='inner')
        elif kind =='allHC':
            merged = pd.read_csv(f'{data_path}/merged_data/unaffectedNoOsteoAllHC.csv').set_index('eid')
            merged_pred = pd.merge(merged,pred_models,left_index=True,right_index=True,how='inner')
        elif kind=='pop':
            merged = pd.read_csv(f'{data_path}/merged_data/populationNoOsteoAllHC.csv').set_index('eid')
            merged_pred = pd.merge(merged,pred_models,left_index=True,right_index=True,how='inner')
        print(merged_pred.shape)
        print(merged_pred[f'diag_{diag}'].value_counts())
        preds = pd.DataFrame(index=pd.MultiIndex.from_product([merged_pred.index,np.arange(5)],names=['eid','cv']),
                                columns=['train','test'])
        dftrain = merged_pred.dropna(subset=np.hstack([merged_pred.filter(regex='train').columns,f'diag_{diag}']),how='any')
        dftrain = dftrain.drop(columns=np.hstack([merged_pred.filter(regex='test').columns]))
        dftrain.columns = [*dftrain.columns[:-5],'all_acc_features','blood','genetics','lifestyle','prodromal_symptoms']

        dftest = merged_pred.dropna(subset=np.hstack([merged_pred.filter(regex='test').columns,f'diag_{diag}']),how='any')
        dftest = dftest.drop(columns=np.hstack([merged_pred.filter(regex='train').columns]))
        dftest.columns = [*dftest.columns[:-5],'all_acc_features','blood','genetics','lifestyle','prodromal_symptoms']
        print(dftrain[['all_acc_features','blood','genetics','lifestyle','prodromal_symptoms']].describe())

        coefs = pd.DataFrame(index=np.arange(5),columns=pd.MultiIndex.from_product([np.hstack(['Intercept',subset]),['coef','pval','CI_upper',
                                                                                                                       'CI_lower']],
                                                                                  names=['modality','statistic']))
        for cv in np.arange(5):
            #grab all train from CV
            dftr = dftrain.loc[(slice(None),cv),:].dropna(subset=subset,
                                                        how='any')
            dfte = dftest.loc[(slice(None),cv),:].dropna(subset=subset,
                                                        how='any')
            #model = smf.logit(f"diag_{diag} ~  all_acc_features + blood + genetics + lifestyle + prodromal_symptoms", data=dftr).fit_regularized(method='l1', alpha=0.1)
            Cs = np.logspace(-3, 1, 5)
            model = linear_model.LogisticRegressionCV(cv=model_selection.StratifiedKFold(n_splits=5, random_state=3,shuffle=True),Cs=Cs,penalty='l1',
                                                      solver='saga',refit=True,max_iter=1000,scoring='average_precision',
                                                      class_weight='balanced',n_jobs=-1,fit_intercept=True).fit(dftr[subset],dftr[f'diag_{diag}'])

            dfte['pred'] = model.predict_proba(dfte[subset])[::,1]
            preds.loc[(dfte.index,cv),'test'] = dfte['pred']
            preds.loc[(dftr.index,cv),'train']= model.predict_proba(dftr[subset])[::,1]
            coefs.loc[cv,(slice(None),'coef')] = np.hstack([model.intercept_,model.coef_[0]])
            #coefs.loc[cv,(slice(None),'pval')] = model.pvalues.values
            #coefs.loc[cv,(slice(None),'CI_upper')] = model.conf_int(alpha=0.05).values[:,1]
            #coefs.loc[cv,(slice(None),'CI_lower')] = model.conf_int(alpha=0.05).values[:,0]
            perf.loc[(diagname,cv),('test','C')] = model.C_
            fpr, tpr, _ = metrics.roc_curve(dftest.loc[dfte.index,f'diag_{diag}'],dfte['pred'])
            perf.loc[(diagname,cv),('test','AUROC')] = metrics.roc_auc_score(dftest.loc[dfte.index,f'diag_{diag}'],dfte['pred'])
            perf.loc[(diagname,cv),('test','fpr')] = fpr
            perf.loc[(diagname,cv),('test','tpr')] = tpr
            perf.loc[(diagname,cv),('test','AUPRC')] = metrics.average_precision_score(dftest.loc[dfte.index,f'diag_{diag}'],dfte['pred'])
            pre,rec,_ = metrics.precision_recall_curve(dftest.loc[dfte.index,f'diag_{diag}'],dfte['pred'])
            perf.loc[(diagname,cv),('test','precision')] = pre
            perf.loc[(diagname,cv),('test','recall')] = rec

        coefs.to_csv(f'{model_path}/{sname}/{diag}{popkind}_stacked.csv')
        perf.to_csv(f'{model_path}/{sname}/{diag}{popkind}_stacked_perf.csv')
        preds.to_csv(f'{model_path}/{sname}/{diag}{popkind}_stacked_pred.csv')
    perf.to_csv(f'{model_path}/{sname}/stacked_perf.csv')