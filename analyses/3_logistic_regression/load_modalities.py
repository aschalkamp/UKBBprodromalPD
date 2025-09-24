import numpy as np 
import pandas as pd



def load_features(data_path):
    merged = pd.read_csv(f'{data_path}/merged_data/unaffectedNoOsteoMatchedHC.csv').set_index('eid')
    # get modalities
    covs = ['accelerometry_age','male']
    allfeatures = np.load(f'{data_path}/analyses/acc_models/allfeatures.npy').tolist()
    allfeatures_scale = np.load(f'{data_path}/analyses/acc_models/allfeatures.npy').tolist()
    allfeatures_scale.remove('male')
    blood = merged.columns[256:]
    blood_scale = merged.columns[256:]
    lifestyle = np.hstack(['TownsendDeprivationIndex',merged.columns[239:245],merged.columns[250:256]])
    lifestyle_scale = ['BMI', 'Waist_Circumference', 'Hip_Circumference',
           'Diastolic_BloodPressure', 'PulseRate', 'BodyFat_Percentage',
           'TownsendDeprivationIndex']
    genetics = np.hstack([merged.columns[205:239],merged.columns[245:250]])
    genetics_scale = merged.columns[205:239]
    prod_acc = merged.filter(regex='_beforeacc').columns
    prod = merged.filter(regex='_beforePD').columns
    
    return covs,allfeatures,allfeatures_scale,blood,blood_scale,lifestyle,lifestyle_scale,genetics,genetics_scale,prod,prod_acc