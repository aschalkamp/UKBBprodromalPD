import numpy as np
import pandas as pd
import joblib
import datetime
from sksurv.metrics import cumulative_dynamic_auc, cumulative_dynamic_auprc, brier_score, integrated_brier_score, concordance_index_ipcw

data_path = '/scratch/c.c21013066/data/ukbiobank'
model_path = '/scratch/c.c21013066/data/ukbiobank/analyses/survival/prodromal/noOsteo'
time_points = np.arange(2.5, 7,0.1)
dt=dtype=[('Status', '?'), ('Survival_in_years', '<f8')]

features = ['intercept','covariates',
          'genetics+family','lifestyle_nofam','blood','acc','all_acc_features','prodromalsigns_beforePD',
         'genetics+family+all_acc_features','lifestyle+all_acc_features','blood+all_acc_features','prodromalsigns_beforePD+all_acc_features',
'all_acc_features+blood+lifestyle+genetics+prodromalsigns_beforePD','prodromalsigns_beforeacc','prodromalsigns_beforeacc+all_acc_features',
         'all_acc_features+blood+lifestyle+genetics+prodromalsigns_beforeacc']
names = ['_matched','_allHC','diag_ProdPopulationNoPD_allHC']
diags = ['diag_ProdHC','diag_ProdHC','diag_ProdPopulationNoPD']

for name,diag in zip(names,diags):
    for feature in features:
        cph_auprcs = pd.DataFrame(index=np.arange(5),columns=['mean'])
        for cv in np.arange(5):
            print(diag,name,feature,cv)
            # load preds
            if name == 'diag_ProdPopulationNoPD_allHC':
                merged = pd.read_csv(f'{data_path}/merged_data/populationNoOsteoAllHC.csv').set_index('eid')
                prodage = pd.read_csv(f'{data_path}/merged_data/unaffectedNoOsteoMatchedHC.csv').set_index('eid')
                merged.loc[prodage[prodage['diag_ProdHC']==1].index,'acc_time_to_diagnosis'] = prodage.loc[prodage['diag_ProdHC']==1,'acc_time_to_diagnosis'].values

                merged.loc[merged[diag]==0,'acc_time_to_diagnosis'] = (pd.Timestamp(datetime.datetime(2021,3,1)) - pd.to_datetime(merged.loc[merged[diag]==0,'date_accelerometry']) )/ np.timedelta64(1,'Y')
                merged = merged.dropna(subset=[diag])
                # load pred and true
                true = pd.read_csv(f'{model_path}/{feature}/{name}/rsf_testrisk_CV{cv}.csv')
                surv = np.load(f'{model_path}/{feature}/{name}/rsf_testpred_CV{cv}.csv.npy')
            else:
                if name == '_matched':
                    merged = pd.read_csv(f'{data_path}/merged_data/unaffectedNoOsteoMatchedHC.csv').set_index('eid')
                elif name == '_allHC':
                    merged = pd.read_csv(f'{data_path}/merged_data/unaffectedNoOsteoAllHC.csv').set_index('eid')
                    prodage = pd.read_csv(f'{data_path}/merged_data/unaffectedNoOsteoMatchedHC.csv').set_index('eid')
                    merged.loc[prodage[prodage['diag_ProdHC']==1].index,'acc_time_to_diagnosis'] = prodage.loc[prodage['diag_ProdHC']==1,'acc_time_to_diagnosis'].values
                else:
                    print('undefined condition')
                    break
                merged.loc[merged[diag]==0,'acc_time_to_diagnosis'] = (pd.Timestamp(datetime.datetime(2021,3,1)) - pd.to_datetime(merged.loc[merged[diag]==0,'date_accelerometry']) )/ np.timedelta64(1,'Y')
                merged = merged.dropna(subset=[diag])
                # load pred
                true = pd.read_csv(f'{model_path}/{feature}/{name}rsf_testrisk_CV{cv}.csv')
                surv = np.load(f'{model_path}/{feature}/rsf_{name}testpred_CV{cv}.csv.npy')
                
            train = merged[ ~merged.index.isin(true.eid) ]
            train[diag] = train[diag].astype('?')
            train['acc_time_to_diagnosis'] = train['acc_time_to_diagnosis'].astype('<f8')
            try:
                train = train.drop(columns=['Status'])
            except:
                pass
            train = train.rename(columns={diag:'Status','acc_time_to_diagnosis':'Survival_in_years'})
                
            data_y = np.array([tuple(row) for row in train[['Status','Survival_in_years']].values], dtype=dt)
            data_y_test = np.array([tuple(row) for row in true[['Status','Survival_in_years']].values], dtype=dt)

            cph_auprc, cph_mean_auprc = cumulative_dynamic_auprc(
                data_y, data_y_test, true['y_risk'],time_points
            )

            cph_auprcs.loc[cv,time_points] = cph_auprc
            cph_auprcs.loc[cv,'mean'] = cph_mean_auprc

            
            if diag == 'diag_ProdPopulationNoPD':
                cph_auprcs.to_csv(f'{model_path}/{feature}/{name}/rsf_auprcs_5cv.csv')
            else:
                cph_auprcs.to_csv(f'{model_path}/{feature}/{name}rsf_auprcs_5cv.csv')
