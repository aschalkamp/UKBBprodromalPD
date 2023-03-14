import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

def load_hc(path='/scratch/c.c21013066/data/ukbiobank/sample/withGP',disease='ParkinsonDisease',drop_healthy='neurology',df_names=['risk','acc_QC']):
    for i,df in enumerate(df_names):
        hc_ = pd.read_csv(f'{path}/{disease}_controlNo{drop_healthy}_{df}.csv').set_index('eid')
        if i == 0:
            hc = hc_.copy(deep=True)
        else:
            hc = pd.merge(hc_,hc,how='outer',right_index=True,left_index=True,suffixes=[f'_{df}',''])

        hc = hc.drop(columns=hc.filter(regex=f'_{df}').columns)
    hc = hc[hc[disease]==0]
    return hc

def clean_hc(hc,covariates=['accelerometry_age','male','BMI'],target='No_wear_time_bias_adjusted_average_acceleration'):
    hc = hc.dropna(subset=np.hstack([covariates,target]),how='any',axis='rows')
    return hc

def fit_model(data,formula='No_wear_time_bias_adjusted_average_acceleration ~ male + accelerometry_age + BMI'):
    mod = smf.ols(formula=formula, data=data)
    res = mod.fit()
    print(res.summary())
    return res

def get_residuals(model,data,res_name="average acceleration residual_bmi",covariates=['accelerometry_age','male','BMI'],target='No_wear_time_bias_adjusted_average_acceleration',intercept=True):
    params = model.params
    data['pred'] = 0
    for cov in covariates:
        data['pred'] += params[cov] * data[cov]
    if intercept:
        data['pred'] += params['Intercept']
    data[res_name] = data[target] - data['pred']
    return data

def fit_get_res(hc,data,target='No_wear_time_bias_adjusted_average_acceleration',covariates=['accelerometry_age','male','BMI'],res_name="average acceleration residual_bmi",
                save='/scratch/c.c21013066/data/ukbiobank/analyses/acc_models/hc_residuals',intercept=True):
    formula = f'{target} ~ ' + ' + '.join(covariates)
    res = fit_model(hc,formula)
    if len(save)>1:
        res.save(f"{save}/{res_name}_model.pickle")
    data = get_residuals(res,data,res_name,covariates,target,intercept=intercept)
    return res,data