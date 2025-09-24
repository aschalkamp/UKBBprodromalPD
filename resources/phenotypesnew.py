import pandas as pd
import numpy as np
import itertools
from simpledbf import Dbf5
import re
import sys
import os
sys.path.insert(1,'/scratch/c.c21013066/software/ukbb_parser/ukbb_parser')
import ukbb_parser as ukbb_parser

ukbb_paths = ukbb_parser._load_ukbb_paths()
print(ukbb_paths.CODINGS_DIR)
visit = 0

def get_FieldsfromCategory(category_id,field_type='continuous'):
    # get info about field ID and encodings
    overview = pd.read_csv(os.path.join(ukbb_paths.CODINGS_DIR,'Data_Dictionary_Showcase.csv'),sep=',',
                       encoding='latin-1',header=0,usecols=np.arange(17))
    # get the one for catgory of interest
    IDPs = overview.loc[overview['Category']==category_id,['FieldID','Field']]
    IDPs['source'] = 'field'
    IDPs['field_type'] = field_type
    IDPs = IDPs.rename(columns={'FieldID':'field_id','Field':'name'})
    # rename so no whitespace of special characters
    IDPs['name'] = IDPs['name'].str.replace(' ','')
    IDPs['name'] = IDPs['name'].str.replace(',','_')
    IDPs['name'] = IDPs['name'].str.replace("'",'')
    IDPs['name'] = IDPs['name'].str.replace('(','_')
    IDPs['name'] = IDPs['name'].str.replace(')','')
    return IDPs.to_dict('records')

def get_accelerometer_data(field_type='continuous'):
    # get info about field ID and encodings
    overview = pd.read_csv(os.path.join(ukbb_paths.CODINGS_DIR,'Data_Dictionary_Showcase.csv'),sep=',',
                       encoding='latin-1',header=0,usecols=np.arange(17))
    # get the one for catgory of interest
    IDPs = overview.loc[overview['Category'].isin([1009,]),['FieldID','Field']]
    IDPs['source'] = 'field'
    IDPs['field_type'] = field_type
    IDPs = IDPs.rename(columns={'FieldID':'field_id','Field':'name'})
    # rename so no whitespace of special characters
    IDPs['name'] = IDPs['name'].str.replace(' of ','')
    IDPs['name'] = IDPs['name'].str.replace(' acceleration ','')
    IDPs['name'] = IDPs['name'].str.replace(":",'')
    IDPs['name'] = IDPs['name'].str.replace('-','_')
    IDPs['name'] = IDPs['name'].str.replace(' ','_')
    IDPs = IDPs.to_dict('records')
    IDPs.append({'name':'date_accelerometry',
                 'source':'field',
                 'field_id':90003,
                 'field_type':'continuous'})
    IDPs.append({'name':'weartime_QC',
                 'source':'field',
                 'field_id':90015,
                 'field_type':'continuous'})
    return IDPs

        
def recode_ICD10(codes):
    add = []
    codings = pd.DataFrame(codes['codings'],columns=['ICD10'])
    codings['addX'] = codings['ICD10'].str.len() == 3
    addedx = [f'{c}X' for c in codings.loc[codings['addX'],'ICD10']]
    add.append(addedx)
    codes['codings_add'] = np.union1d(addedx,codings['ICD10'])
    return codes
    
def recode_reads(reads):
    reads = reads.set_index('READ_CODE')
    reads = pd.merge(reads,reads['TARG_CODE'].str.split(r'\+| ',expand=True).add_prefix('TARG_CODE_split_'),right_index=True,left_index=True).reset_index()
    reads = pd.melt(reads,id_vars=['READ_CODE','TARG_CODE','MAP_STAT','REF_FLAG','ADD_FLAG','ELEM_NUM','BLOCK_NUM'],var_name='TARG_CODE_alt',value_vars=reads.filter(regex='^TARG_CODE_split_').columns)
    reads['value'] = reads['value'].str.replace(r'(A|D)$','')
    reads = reads.dropna(axis='rows',how='all',subset=['value'])
    # drop duplicates
    reads = reads.drop_duplicates(subset=['READ_CODE','value'])
    return reads

def get_gpdiagnosis_from_ICD10_3code(data,icd10,name):
    codemap2 = pd.read_csv(os.path.join(ukbb_paths.CODINGS_DIR,f'coding1834.tsv'),sep='\t')
    codemap3 = pd.read_csv(os.path.join(ukbb_paths.CODINGS_DIR,f'coding1835.tsv'),sep='\t')
    r2 = codemap2[codemap2['meaning'].isin(icd10)]
    r3 = codemap3[codemap3['meaning'].isin(icd10)]
    r = data[np.logical_or(data['read_3'].isin(r3['coding']),data['read_2'].isin(r2['coding']))]
    r = r.sort_values(['eid','event_dt'])
    r = r.groupby('eid').first().reset_index()
    df = pd.DataFrame(index=data['eid'].unique())
    df[name] = 0
    df.loc[r['eid'],name] = 1
    df[f'{name}_date'] = pd.to_datetime('1900-01-01')
    df.loc[r['eid'],f'{name}_date'] = r['event_dt'].values
    df.loc[df[f'{name}_date']==pd.to_datetime('1900-01-01'),f'{name}_date'] = np.nan
    df = df.reset_index()
    df = df.rename(columns={'index':'eid'})
    return df

def get_gpdiagnosis_from_ICD10(data,icd10,name):
    dbf3 = Dbf5(os.path.join(ukbb_paths.CODINGS_DIR,'readbrowser/Standard/V3/ICD10.DBF'))
    df3 = dbf3.to_dataframe()
    dbf2 = Dbf5(os.path.join(ukbb_paths.CODINGS_DIR,'readbrowser/Standard/V2/ICD10.DBF'))
    df2 = dbf2.to_dataframe()
    df3 = recode_reads(df3)
    df2 = recode_reads(df2)
    print('Extract relevant read codes for ',name)
    r2 = df2[df2['value'].isin(icd10)]
    r3 = df3[df3['value'].isin(icd10)]
    r3.to_csv(os.path.join(ukbb_paths.CODINGS_DIR,f'readv3ICD10_{name}.csv'))
    r2.to_csv(os.path.join(ukbb_paths.CODINGS_DIR,f'readv2ICD10_{name}.csv'))
    print('Get subjects with relevant codes')
    r = data[np.logical_or(data['read_3'].isin(r3['READ_CODE']),data['read_2'].isin(r2['READ_CODE']))]
    r = r.sort_values(['eid','event_dt'])
    r = r.groupby('eid').first().reset_index()
    df = pd.DataFrame(index=data['eid'].unique())
    df[name] = 0
    df.loc[r['eid'],name] = 1
    df[f'{name}_date'] = pd.to_datetime('1900-01-01')
    df.loc[r['eid'],f'{name}_date'] = r['event_dt'].values
    df.loc[df[f'{name}_date']==pd.to_datetime('1900-01-01'),f'{name}_date'] = np.nan
    df = df.reset_index()
    df = df.rename(columns={'index':'eid'})
    return df

def run_gpdiagnosis(codes):
    data = pd.read_csv(f'{ukbb_paths.RECORD_DIR}/gp_clinical.txt',nrows=None,sep='\t',encoding = "ISO-8859-1")
    data['event_dt'] = pd.to_datetime(data['event_dt'])
    print('loaded GP clinical data')
    for code in codes:
        print('Processing for ',code['name'])
        code = recode_ICD10(code)
        name = code['name']
        print(name)
        # strip code to 3 character version
        #code['codings_3'] = np.unique([c[:3] for c in code['codings']])
        print(code)
        #if name == 'neurology':
        #print('using 3code')
        df = get_gpdiagnosis_from_ICD10(data,code['codings_add'],code['name'])
        #else:
        #    df = get_gpdiagnosis_from_ICD10(data,code['codings'],code['name'])
        df.to_csv(f'{ukbb_paths.RECORD_DIR}/{name}.csv',index=False)
        
        
def run_gpmedication(codes):
    data = pd.read_csv(f'{ukbb_paths.RECORD_DIR}/gp_scripts.txt',nrows=None,sep='\t',encoding = "ISO-8859-1")
    data['event_dt'] = pd.to_datetime(data['issue_date'])
    print('loaded GP script data')
    for code in codes:
        print('Processing for ',code['name'])
        print(code['name'])
        name = code['name']
        df,r = get_gpmedication(data,code['codings'],name)
        df.to_csv(f'{ukbb_paths.RECORD_DIR}/{name}_first.csv',index=False)
        r.to_csv(f'{ukbb_paths.RECORD_DIR}/{name}_raw.csv',index=False)
        

def get_gpmedication(data,codes,name):
    r = data[data['read_2'].isin(codes)]
    r = r.sort_values(['eid','issue_date'])
    r['quantity'] = r['quantity'].astype(float)
    first = r.groupby('eid').first().reset_index()
    df = pd.DataFrame(index=data['eid'].unique())
    df[name] = 0
    df.loc[r['eid'],name] = 1
    df[f'{name}_date'] = pd.to_datetime('1900-01-01')
    df.loc[first['eid'],f'{name}_date'] = first['issue_date'].values
    df.loc[df[f'{name}_date']==pd.to_datetime('1900-01-01'),f'{name}_date'] = np.nan
    df = df.reset_index()
    df = df.rename(columns={'index':'eid'})
    return df,r
                      

ACCELEROMETER = get_accelerometer_data(visit)


TEST = [
    {
        'name': 'self_dementia',
        'source': 'ICD-10',
        'codings': ['1074','1065']
    },
]

GPDRUGS = [
    {
        'name': 'gp_AntiParkinsonism',
        'codings': pd.read_excel(f'{ukbb_paths.CODINGS_DIR}/AntiParkinsonismDrugs.xls',header=None,names=['codings']).values[:,0]
    }
]

ICD10_FIELDS = [
    # (field_name, field_id, field_type)
    ('Diagnoses', 41270, 'raw'),
    ('CancerType', 40006, 'raw'),
    ('PrimaryCauseDeath', 40001, 'raw'),
    ('SecondaryCauseDeath', 40002, 'raw'),
    ('Diagnoses_date', 41280, 'raw'),
    ('CancerType_date', 40005, 'raw'),
    ('PrimaryCauseDeath_date', 40000, 'raw')
]
ICD9_FIELDS = [
    # (field_name, field_id, field_type)
    ('Diagnoses', 41271, 'raw'),
    ('CancerType', 40013, 'raw'),
    ('Diagnoses_date', 41281, 'raw'),
    ('CancerType_date', 40005, 'raw'),
]

SELF_FIELDS = [
    # (field_name, field_id, field_type)
    ('selfreported', 20002, 'raw'),
    ('selfreported_date', 87, 'raw'),
    ('selfreported_method', 20013, 'raw'),
]

MEDICATION = [
    ('medication', 20003, 'raw')
]

GeneticPCs = [
    ('PC', 22009, 'raw')
]

DEATH = [
    {
        'name': 'death_date',
        'source': 'field',
        'field_id': 40000,
        'field_type': 'raw',
    },
    {
        'name': 'death_age',
        'source': 'field',
        'field_id': 40007,
        'field_type': 'raw',
    },
]
GENES = [
    {
        'name': 'sex_male',
        'source': 'field',
        'field_id': 22001,
        'field_type': 'continuous',
    },
    {
        'name': 'Heterozygoty_corrected',
        'source': 'field',
        'field_id': 22004,
        'field_type': 'continuous'
    },
    {
        'name': 'Missingness',
        'source': 'field',
        'field_id': 22005,
        'field_type': 'continuous',
    },
    {
        'name': 'Kinship', #coding 682: -1 excluded,0 no kinship, 1 at least 1 relative, 10 >=10 3rd degree relatives
        'source': 'field',
        'field_id': 22021,
        'field_type': 'continuous',
    },
    {
        'name': 'Gene_QualityIssues', # outliers for heterozygoty or missingness
        'source': 'field',
        'field_id': 22027,
        'field_type': 'continuous'
    },    {
        'name': 'Gene_ethnicity', # self-reported white british and PC aligns with it coding 1002
        'source': 'field',
        'field_id': 22006,
        'field_type': 'continuous'
    }
]

PHYSICAL = [
    {
        'name': 'Height',
        'source': 'field',
        'field_id': 50,
        'field_type': visit,
    },
    {
        'name': 'BMI',
        'source': 'field',
        'field_id': 21001,
        'field_type': visit,
    },
    {
        'name': 'Waist_Circumference',
        'source': 'field',
        'field_id': 48,
        'field_type': visit,
    },
    {
        'name': 'Hip_Circumference',
        'source': 'field',
        'field_id': 49,
        'field_type': visit,
    },
    {
        'name': 'Diastolic_BloodPressure',
        'source': 'field',
        'field_id': 4079,
        'field_type': visit,
    },
    {
        'name': 'Systolic_BloodPressure',
        'source': 'field',
        'field_id': 4080,
        'field_type': visit,
    },
    {
        'name': 'PulseRate',
        'source': 'field',
        'field_id': 102,
        'field_type': visit,
    },
    {
        'name': 'ArmFat_Percentage',
        'source': 'aggregation',
        'subspecs': [
            {
                'name': 'ArmFat_Percentage_l',
                'source': 'field',
                'field_id': 23123,
                'field_type': visit,
            },
            {
                'name': 'ArmFat_Percentage_r',
                'source': 'field',
                'field_id': 23119,
                'field_type': visit,
            },
        ],
        'aggregation_function': lambda right, left: pd.concat([right, left], axis = 1).mean(axis = 1), # We take the mean between the two fields.
    },
    {
        'name': 'LegFat_Percentage',
        'source': 'aggregation',
        'subspecs': [
            {
                'name': 'LegFat_Percentage_l',
                'source': 'field',
                'field_id': 23115,
                'field_type': visit,
            },
            {
                'name': 'LegFat_Percentage_r',
                'source': 'field',
                'field_id': 23111,
                'field_type': visit,
            },
        ],
        'aggregation_function': lambda right, left: pd.concat([right, left], axis = 1).mean(axis = 1), # We take the mean between the two fields.
    },
    {
        'name': 'BodyFat_Percentage',
        'source': 'field',
        'field_id': 23099,
        'field_type': visit
    },    
    {
        'name': 'HandGrip_strength',
        'source': 'aggregation',
        'subspecs': [
            {
                'name': 'HandGrip_strength_l',
                'source': 'field',
                'field_id': 46,
                'field_type': visit,
            },
            {
                'name': 'HandGrip_strength_r',
                'source': 'field',
                'field_id': 47,
                'field_type': visit,
            },
        ],
        'aggregation_function': lambda left, right: pd.concat([left, right], axis = 1).max(axis = 1), # We take the maximum between the two fields.
    }
]

def get_children(tree,idx=[],pattern='',selectable=False):
    children = []
    for i in idx:
        ids = list(tree.loc[i,'children_ids'])
        if len(ids) >= 1:
            if selectable:
                if tree.loc[i,'selectable']=='Y':
                    children.append([tree.loc[i,'coding']])
            if len(pattern)>0:
                if bool(re.match(pattern,tree.loc[i,'coding'])):
                    children.append([tree.loc[i,'coding']])               
            children.append(get_children(tree,ids,pattern=pattern,selectable=selectable))
        else:
            children.append([tree.loc[i,'coding']])
    return list(set(itertools.chain(*children)))


DIAGNOSESSELF = [
    {
        'name': 'selfreported_AllCauseDementia',
        'source': 'ICD-10',
        'codings': ['1263']
    },
    {
        'name': 'selfreported_ParkinsonDisease',
        'source': 'ICD-10',
        'codings': ['1262']
    },
    {
        'name': 'selfreported_neurology',
        'source': 'ICD-10',
        'codings': get_children(ukbb_parser.construct_ICD10_tree(code='6'),[1076],selectable=True,pattern='')
    },
    {
        'name': 'selfreported_Osteoarthritis',
        'source': 'ICD-10',
        'codings': ['1465']
    },
    {
        'name': 'selfreported_Depression',
        'source': 'ICD-10',
        'codings': ['1286']
    },
    {
        'name': 'selfreported_UrinaryIncontinence',
        'source': 'ICD-10',
        'codings': ['1202']
    },
    {
        'name': 'selfreported_ErectileDysfunction',
        'source': 'ICD-10',
        'codings': ['1518']
    },
    {
        'name': 'selfreported_Constipation',
        'source': 'ICD-10',
        'codings': ['1599']
    },
    {
        'name': 'selfreported_Anxiety',
        'source': 'ICD-10',
        'codings': ['1287']
    },
    {
        'name': 'selfreported_Multiple_sclerosis',
        'source': 'ICD-10',
        'codings': ['1261']
    },
    {
        'name': 'selfreported_nonHC',
        'source': 'ICD-10',
        'codings': list(set(itertools.chain(*[get_children(ukbb_parser.construct_ICD10_tree(code='6'),[1076],selectable=True,pattern=''),['1465']])))
    },
]

DIAGNOSESICD9 = [
    {
        'name': 'icd9_ParkinsonDisease',
        'source': 'ICD-10',
        'codings': ['3320'],
    },
    {
        'name': 'icd9_OtherParkinsonism',
        'source': 'ICD-10',
        'codings': ['3321','3330'],
    },
    {
        'name': 'icd9_AllCauseParkinsonism',
        'source': 'ICD-10',
        'codings': ['3320','3321','3330'],
    },
    {
        'name': 'icd9_AlzheimerDisease',
        'source': 'ICD-10',
        'codings': ['3310'],
    },
    {
        'name': 'icd9_VascularDementia',
        'source': 'ICD-10',
        'codings': ['2904'],
    },
    {
        'name': 'icd9_FrontoTemporalDementia',
        'source': 'ICD-10',
        'codings': ['3311'],
    },
    {
        'name': 'icd9_AllCauseDementia',
        'source': 'ICD-10',
        'codings': ['2902','2903','2904','2912','2941','3310','3311','3312','3315'],
    },
    {
        'name': 'icd9_neurology',
        'source': 'ICD-10',
        'codings': list(set(itertools.chain(*[get_children(ukbb_parser.construct_ICD10_tree(code='87'),[5],selectable=True),get_children(ukbb_parser.construct_ICD10_tree(code='87'),[6],selectable=True)])))
    },
    {
        'name': 'icd9_Dystonia',
        'source': 'ICD-10',
        'codings': ['3337','3338','3336'],
    },
    {
        'name': 'icd9_Depression',
        'source': 'ICD-10',
        'codings': ['2962', '2963', '3004', '3119'],
    },
    {
        'name': 'icd9_Osteoarthritis',
        'source': 'ICD-10',
        'codings': ['7153'],
    },
    {
        'name': 'icd9_UrinaryIncontinence',
        'source': 'ICD-10',
        'codings': ['7883']
    },
    {
        'name': 'icd9_ErectileDysfunction',
        'source': 'ICD-10',
        'codings': ['3027']
    },
    {
        'name': 'icd9_Constipation',
        'source': 'ICD-10',
        'codings': ['56409','56402']
    },
    {
        'name': 'icd9_Anxiety',
        'source': 'ICD-10',
        'codings': ['3000','3001','3002','3005','3009']
    },
    {
        'name': 'icd9_OrthostaticHypotension',
        'source': 'ICD-10',
        'codings': ['4580']
    },
    {
        'name': 'icd9_Hyposmia',
        'source': 'ICD-10',
        'codings': ['7811']
    },
    {
        'name': 'icd9_Multiple_sclerosis',
        'source': 'ICD-10',
        'codings': ['3409'],
    },
    {
        'name': 'icd9_nonHC',
        'source': 'ICD-10',
        'codings': list(set(itertools.chain(*[get_children(ukbb_parser.construct_ICD10_tree(code='87'),[5],selectable=True),get_children(ukbb_parser.construct_ICD10_tree(code='87'),[6],selectable=True),['7153']])))
    },
]

DIAGNOSESICD10 = [
    {
        'name': 'icd10_Breast_cancer',
        'source': 'ICD-10',
        'codings': ['C50'],
        'sex_filter': 'F',
    },
    {
        'name': 'icd10_EpithelialOvarian_cancer',
        'source': 'ICD-10',
        'codings': ['C56'],
        'sex_filter': 'F',
    },
    {
        'name': 'icd10_Prostate_cancer',
        'source': 'ICD-10',
        'codings': ['C61'],
        'sex_filter': 'M',
    },
    {
        'name': 'icd10_Colorectal_cancer',
        'source': 'ICD-10',
        'codings': ['C18'],
    },
    {
        'name': 'icd10_Lung_cancer',
        'source': 'ICD-10',
        'codings': ['C34'],
    },
    {
        'name': 'icd10_ChronicLymphocytic_leukemia',
        'source': 'ICD-10',
        'codings': ['C91'],
    },
    {
        'name': 'icd10_Pancreatic_cancer',
        'source': 'ICD-10',
        'codings': ['C25'],
    },
    {
        'name': 'icd10_Melanoma',
        'source': 'ICD-10',
        'codings': ['C43'],
    },
    {
        'name': 'icd10_Schizophrenia',
        'source': 'ICD-10',
        'codings': ['F20'],
    },
    {
        'name': 'icd10_BipolarDisorder',
        'source': 'ICD-10',
        'codings': ['F31'],
    },
    {
        'name': 'icd10_MajorDepressiveDisorder',
        'source': 'ICD-10',
        'codings': ['F33'],
    },
    {
        'name': 'icd10_ParkinsonDisease',
        'source': 'ICD-10',
        'codings': ['G20'],
    },
    {
        'name': 'icd10_neurology',
        'source': 'ICD-10',
        'codings': list(set(itertools.chain(*[get_children(ukbb_parser.construct_ICD10_tree(code='19'),[6],'^[A-Z]{1}\d{2,}$'),
                                              get_children(ukbb_parser.construct_ICD10_tree(code='19'),[5],'^[A-Z]{1}\d{2,}$'),['A810','I673']])))
        
    },
    {
        'name': 'icd10_AllCauseParkinsonism',
        'source': 'ICD-10',
        'codings': ['G20','G21','G210','G211','G212','G213','G214','G218','G219',"G22",'G230',
                    'G231','G232','G233','G238','G239','G259','G26','G903'],
    },
    {
        'name': 'icd10_OtherParkinsonism',
        'source': 'ICD-10',
        'codings': ['G21','G210','G211','G212','G213','G214','G218','G219',"G22",'G230',
                    'G231','G232','G233','G238','G239','G259','G26','G903'],
    },
    {
        'name': 'icd10_MultipleSystemAtrophy',
        'source': 'ICD-10',
        'codings': ['G232','G233','G903'],
    },
    {
        'name': 'icd10_ProgressiveSupranuclearPalsy',
        'source': 'ICD-10',
        'codings': ['G231'],
    },
    {
        'name': 'icd10_Dystonia',
        'source': 'ICD-10',
        'codings': ['G24'],
    },
    {
        'name': 'icd10_SleepDisorders',
        'source': 'ICD-10',
        'codings': ['G47','G470','G471','G472','G473','G474','G478','G479'],
    },
    {
        'name': 'icd10_RBD',
        'source': 'ICD-10',
        'codings': ['G478'],
    },
    {
        'name': 'icd10_AlzheimerDisease',
        'source': 'ICD-10',
        'codings': ['F00','F000','F001','F002','F009','G30','G300','G301','G308','G309'],
    },
    {
        'name': 'icd10_AllCauseDementia',
        'source': 'ICD-10',
        'codings': ['A810','F00','F000','F001','F002','F009','F01','F010','F011','F012','F013','F018','F019',
        'F02','F020','F021','F022','F023','F024','F028','F03','F051','F106','G30','G300','G301','G308',
        'G309','G310','G311','G318','I673'],
    },
    {
        'name': 'icd10_NotAlzheimers',
        'source': 'ICD-10',
        'codings': ['A810','F01','F010','F011','F012','F013','F018','F019',
        'F02','F020','F021','F022','F023','F024','F028','F03','F051','F106','G310','G311','G318','I673'],
    },
    {
        'name': 'icd10_VascularDementia',
        'source': 'ICD-10',
        'codings': ['F01','F010','F011','F012','F013','F018','F019','I673'],
    },
    {
        'name': 'icd10_FrontoTemporalDementia',
        'source': 'ICD-10',
        'codings': ['F020','G310'],
    },    
    {
        'name': 'icd10_Stroke',
        'source': 'ICD-10',
        'codings': ['I63'],
    },
    {
        'name': 'icd10_Hypertension',
        'source': 'ICD-10',
        'codings': ['I10'],
    },
    {
        'name': 'icd10_SuddenCardiacArrest',
        'source': 'ICD-10',
        'codings': ['I46'],
    },
    {
        'name': 'icd10_Type1_diabetes',
        'source': 'ICD-10',
        'codings': ['E100', 'E101', 'E102', 'E103', 'E104', 'E105', 'E106', 'E107', 'E108', 'E109'],
    },
    {
        'name': 'icd10_Type2_diabetes',
        'source': 'ICD-10',
        'codings': ['E110', 'E111', 'E112', 'E113', 'E114', 'E115', 'E116', 'E117', 'E118', 'E119'],
    },
    {
        'name': 'icd10_Systemic_sclerosis',
        'source': 'ICD-10',
        'codings': ['M34'],
    },
    {
        'name': 'icd10_Multiple_sclerosis',
        'source': 'ICD-10',
        'codings': ['G35'],
    },
    {
        'name': 'icd10_SystemicLupusErythematosus',
        'source': 'ICD-10',
        'codings': ['M32'],
    },
    {
        'name': 'icd10_RheumatoidArthritis',
        'source': 'ICD-10',
        'codings': ['M05', 'M06'],
    },
    {
        'name': 'icd10_Asthma',
        'source': 'ICD-10',
        'codings': ['J45'],
    },
    {
        'name': 'icd10_CrohnColitis',
        'source': 'ICD-10',
        'codings': ['K50', 'K51'],
    },
    {
        'name': 'icd10_Constipation',
        'source': 'ICD-10',
        'codings': ['K590']
    },
    {
        'name': 'icd10_Depression',
        'source': 'ICD-10',
        'codings': ['F204','F32','F328','F33','F330','F331','F332','F333','F334','F338']
    },
    {
        'name': 'icd10_Anxiety',
        'source': 'ICD-10',
        'codings': ['F412','F413','F418','F419','F480','F488','F489','F064']
    },
    {
        'name': 'icd10_Migraine',
        'source': 'ICD-10',
        'codings': ['G430','G431']
    },
    {
        'name': 'icd10_Hyposmia',
        'source': 'ICD-10',
        'codings': ['R430']
    },
    {
        'name': 'icd10_HeadInjury',
        'source': 'ICD-10',
        'codings': ['S020','S021','S022','S023','S024','S026','S027','S028','S029',
                    'S060','S061','S062','S063','S064','S065','S066','S068','S069']
    },
    {
        'name': 'icd10_GastricUlcer',
        'source': 'ICD-10',
        'codings': ['K250','K251','K252','K253','K254','K255','K256','K257','K259']
    },
    {
        'name': 'icd10_Inflammatory',
        'source': 'ICD-10',
        'codings': ['J31','J35','J37','J41','J42','J43','J44','J47']
    },
    {
        'name': 'icd10_Allergies',
        'source': 'ICD-10',
        'codings': ['J30','J31','J45']
    },
    {
        'name': 'icd10_Osteoarthritis',
        'source': 'ICD-10',
        'codings': ['M16','M163','M169','M189']
    },
    {
        'name': 'icd10_UrinaryIncontinence',
        'source': 'ICD-10',
        'codings': ['N394','R32']
    },
    {
        'name': 'icd10_ErectileDysfunction',
        'source': 'ICD-10',
        'codings': ['N484','F522']
    },
    {
        'name': 'icd10_OrthostaticHypotension',
        'source': 'ICD-10',
        'codings': ['I951']
    },
    {
        'name': 'icd10_HuntingtonDisease',
        'source': 'ICD-10',
        'codings': ['G10']
    },
    {
        'name': 'icd10_nonHC',
        'source': 'ICD-10',
        'codings': list(set(itertools.chain(*[get_children(ukbb_parser.construct_ICD10_tree(code='19'),[6],'^[A-Z]{1}\d{2,}$'),
                                              get_children(ukbb_parser.construct_ICD10_tree(code='19'),[5],'^[A-Z]{1}\d{2,}$'),['A810','I673','M16','M163','M169','M189']])))
        
    },
]

codes = pd.DataFrame.from_dict(DIAGNOSESICD10)

DIAGNOSESGPADD = [
    {
        'name': 'gp_nonHC',
        'codings': codes.loc[codes['name']=='icd10_neurology','codings'].values.tolist()[0]
    },
    #{
    #    'name': 'gp_HuntingtonDisease',
    #    'codings': codes.loc[codes['name']=='icd10_HuntingtonDisease','codings'].values.tolist()[0]
    #},
    #{
    #    'name': 'gp_Multiple_sclerosis',
    #    'codings': codes.loc[codes['name']=='icd10_Multiple_sclerosis','codings'].values.tolist()[0]
    #},
]

DIAGNOSESGP = [
    {
        'name': 'gp_AllCauseDementia',
        'codings': codes.loc[codes['name']=='icd10_AllCauseDementia','codings'].values.tolist()[0]
    },
    {
        'name': 'gp_AllCauseParkinsonism',
        'codings': codes.loc[codes['name']=='icd10_AllCauseParkinsonism','codings'].values.tolist()[0]
    },
    {
        'name': 'gp_OtherParkinsonism',
        'codings': codes.loc[codes['name']=='icd10_OtherParkinsonism','codings'].values.tolist()[0]
    },
    {
        'name': 'gp_MultipleSystemAtrophy',
        'codings': codes.loc[codes['name']=='icd10_MultipleSystemAtrophy','codings'].values.tolist()[0]
    },
    {
        'name': 'gp_ProgressiveSupranuclearPalsy',
        'codings': codes.loc[codes['name']=='icd10_ProgressiveSupranuclearPalsy','codings'].values.tolist()[0]
    },
    {
        'name': 'gp_FrontoTemporalDementia',
        'codings': codes.loc[codes['name']=='icd10_FrontoTemporalDementia','codings'].values.tolist()[0]
    },
    {
        'name': 'gp_VascularDementia',
        'codings': codes.loc[codes['name']=='icd10_VascularDementia','codings'].values.tolist()[0]
    },
    {
        'name': 'gp_ParkinsonDisease',
        'codings': codes.loc[codes['name']=='icd10_ParkinsonDisease','codings'].values.tolist()[0]
    },
    {
        'name': 'gp_neurology',
        'codings': codes.loc[codes['name']=='icd10_neurology','codings'].values.tolist()[0]
    },
    {
        'name': 'gp_Depression',
        'codings': codes.loc[codes['name']=='icd10_Depression','codings'].values.tolist()[0]
    },
    {
        'name': 'gp_AlzheimerDisease',
        'codings': codes.loc[codes['name']=='icd10_AlzheimerDisease','codings'].values.tolist()[0]
    },
    {
        'name': 'gp_RBD',
        'codings': codes.loc[codes['name']=='icd10_RBD','codings'].values.tolist()[0]
    },
    {
        'name': 'gp_Dystonia',
        'codings': codes.loc[codes['name']=='icd10_Dystonia','codings'].values.tolist()[0]
    },
    {
        'name': 'gp_Osteoarthritis',
        'codings': codes.loc[codes['name']=='icd10_Osteoarthritis','codings'].values.tolist()[0]
    },
        {
        'name': 'gp_UrinaryIncontinence',
        'codings': codes.loc[codes['name']=='icd10_UrinaryIncontinence','codings'].values.tolist()[0]
    },
    {
        'name': 'gp_ErectileDysfunction',
        'codings': codes.loc[codes['name']=='icd10_ErectileDysfunction','codings'].values.tolist()[0]
    },
    {
        'name': 'gp_Constipation',
        'codings': codes.loc[codes['name']=='icd10_Constipation','codings'].values.tolist()[0]
    },
    {
        'name': 'gp_Anxiety',
        'codings': codes.loc[codes['name']=='icd10_Anxiety','codings'].values.tolist()[0]
    },
    {
        'name': 'gp_OrthostaticHypotension',
        'codings': codes.loc[codes['name']=='icd10_OrthostaticHypotension','codings'].values.tolist()[0]
    },
    {
        'name': 'gp_Hyposmia',
        'codings': codes.loc[codes['name']=='icd10_Hyposmia','codings'].values.tolist()[0]
    },
    {
        'name': 'gp_nonHC',
        'codings': codes.loc[codes['name']=='icd10_neurology','codings'].values.tolist()[0]
    },
]

DEMOGRAPHICS = [
    {
        'name': 'male',
        'source':'field',
        'field_id': 31,
        'field_type': 'continuous'
    },
    {
        'name': 'year_birth',
        'source': 'field',
        'field_id': 34,
        'field_type': 'continuous'
    },
    {
        'name': 'month_birth',
        'source': 'field',
        'field_id': 52,
        'field_type': 'continuous',
    },
    {
        'name': 'country_birth',
        'source':'field',
        'field_id': 1647,
        'field_type': 'continuous'   
    },
    {
        'name': 'handedness',
        'source':'field',
        'field_id': 1707,
        'field_type': 'continuous'   
    },
    {
        'name': 'skin_color',
        'source':'field',
        'field_id': 1717,
        'field_type': 'continuous'   
    },   
    {
        'name': 'ethnicity', # data coding: 1001
        'source': 'field',
        'field_id': 21000,
        'field_type': 'continuous'
    },
    {
        'name': 'TownsendDeprivationIndex',
        'source': 'field',
        'field_type': 'continuous',
        'field_id': 189
    },
    {
        'name': 'EducationAge',
        'source': 'field',
        'field_type': visit,
        'field_id': 845
    },
]

ASSESSMENTS = [
    {
        'name': 'date_visit',
        'source': 'field',
        'field_id': 53,
        'field_type': visit
    },
    {
        'name': 'site',
        'source': 'field',
        'field_id': 54,
        'field_type': visit
    }
]

CognitionOnlineFU = [
    {
        'name': 'TMT_date',
        'source': 'field',
        'field_id': 20136,
        'field_type': 0
    },
    {
        'name': 'TMT_B_sec',
        'source': 'field',
        'field_id': 20157,
        'field_type': 0
    },
    {
        'name': 'TMT_A_sec',
        'source': 'field',
        'field_id': 20156,
        'field_type': 0
    },
    {
        'name': 'TMT_A_err',
        'source': 'field',
        'field_id': 20247,
        'field_type': 0
    },
    {
        'name': 'TMT_B_err',
        'source': 'field',
        'field_id': 20248,
        'field_type': 0
    },
    {
        'name': 'TMT_completion',
        'source': 'field',
        'field_id': 20246,
        'field_type': 0
    },
    {
        'name': 'SymbolDigit_date',
        'source': 'field',
        'field_id': 20137,
        'field_type': 0
    },
    {
        'name': 'SymbolDigit_completion',
        'source': 'field',
        'field_id': 20245,
        'field_type': 0
    },
    {
        'name': 'SymbolDigit_correct',
        'source': 'field',
        'field_id': 20159,
        'field_type': 0
    },
    {
        'name': 'SymbolDigit_trials',
        'source': 'field',
        'field_id': 20195,
        'field_type': 0
    },
    {
        'name': 'SymbolDigit_sec',
        'source': 'field',
        'field_id': 20230,
        'field_type': 0
    },
    {
        'name': 'NumMemory_date',
        'source': 'field',
        'field_id': 20138,
        'field_type': 0
    },
    {
        'name': 'NumMemory_correct',
        'source': 'field',
        'field_id': 20240,
        'field_type': 0
    },
    
    {
        'name': 'FluidInt_date',
        'source': 'field',
        'field_id': 20135,
        'field_type': 0
    },
    {
        'name': 'FluidInt_status',
        'source': 'field',
        'field_id': 20242,
        'field_type': 0
    },
    {
        'name': 'FluidInt_score',
        'source': 'field',
        'field_id': 20191,
        'field_type': 0
    },
    {
        'name': 'Pairs_date',
        'source': 'field',
        'field_id': 20134,
        'field_type': 0
    },
    {
        'name': 'Pairs_status',
        'source': 'field',
        'field_id': 20244,
        'field_type': 0
    },
    {
        'name': 'Pairs_correct',
        'source': 'field',
        'field_id': 20131,
        'field_type': 'set'
    },
    {
        'name': 'Pairs_incorrect',
        'source': 'field',
        'field_id': 20132,
        'field_type': 'set'
    },
    {
        'name': 'Pairs_rows',
        'source': 'field',
        'field_id': 20130,
        'field_type': 'set'
    },
    {
        'name': 'Pairs_columns',
        'source': 'field',
        'field_id': 20129,
        'field_type': 'set'
    },
    {
        'name': 'Pairs_time',
        'source': 'field',
        'field_id': 20133,
        'field_type': 'set'
    }
]

CognitionTouchScreen = [
    {
        'name': 'NumericMemory',
        'source': 'field',
        'field_id': 4282,
        'field_type': visit
    },
    {
        'name': 'ReactionTime',
        'source': 'field',
        'field_id': 20023,
        'field_type': visit
    },
    {
        'name': 'ProspectiveMemory',
        'source': 'field',
        'field_id': 20018,
        'field_type': visit
    },
    {
        'name': 'ProspectiveMemory_firstanswer',
        'source': 'field',
        'field_id': 4292,
        'field_type': visit
    },
    {
        'name': 'Reasoning',
        'source': 'field',
        'field_id': 20016,
        'field_type': visit
    },
#     {
#         'name': 'TMT_B_sec',
#         'source': 'field',
#         'field_id': 6350,
#         'field_type': visit
#     },
#     {
#         'name': 'TMT_A_sec',
#         'source': 'field',
#         'field_id': 6348,
#         'field_type': visit
#     },
#     {
#         'name': 'TMT_B_err',
#         'source': 'field',
#         'field_id': 6351,
#         'field_type': visit
#     },
#     {
#         'name': 'TMT_A_err',
#         'source': 'field',
#         'field_id': 6349,
#         'field_type': visit
#     },
#     {
#         'name': 'Pairs',
#         'source': 'field',
#         'field_id': 399,
#         'field_type': visit
#     },
#     {
#         'name': 'SymbolDigit_correct',
#         'source': 'field',
#         'field_id': 23324,
#         'field_type': visit
#     },
#     {
#         'name': 'SymbolDigit_trials',
#         'source': 'field',
#         'field_id': 23323,
#         'field_type': visit
#     },
    {
        'name': 'Pairs_correct',
        'source': 'field',
        'field_id': 398,
        'field_type': 'set'
    },
    {
        'name': 'Pairs_incorrect',
        'source': 'field',
        'field_id': 399,
        'field_type': 'set'
    },
    {
        'name': 'Pairs_rows',
        'source': 'field',
        'field_id': 397,
        'field_type': 'set'
    },
    {
        'name': 'Pairs_columns',
        'source': 'field',
        'field_id': 396,
        'field_type': 'set'
    },
    {
        'name': 'Pairs_time',
        'source': 'field',
        'field_id': 400,
        'field_type': 'set'
    }
]

IDPsFIRST = [
    {
        'name': 'Accumbens_l',
        'source': 'field',
        'field_id': 25023,
        'field_type': visit,
    },
    {
        'name': 'Accumbens_r',
        'source': 'field',
        'field_id': 25024,
        'field_type': visit,
    },
    {
        'name': 'Amygdala_l',
        'source': 'field',
        'field_id': 25021,
        'field_type': visit,
    },
    {
        'name': 'Amygdala_r',
        'source': 'field',
        'field_id': 25022,
        'field_type': visit,
    },
    {
        'name': 'Caudate_l',
        'source': 'field',
        'field_id': 25013,
        'field_type': visit,
    },
    {
        'name': 'Caudate_r',
        'source': 'field',
        'field_id': 25014,
        'field_type': visit,
    },
    {
        'name': 'Hippocampus_l',
        'source': 'field',
        'field_id': 25019,
        'field_type': visit,
    },
    {
        'name': 'Hippocampus_r',
        'source': 'field',
        'field_id': 25020,
        'field_type': visit,
    },
    {
        'name': 'Pallidum_l',
        'source': 'field',
        'field_id': 25017,
        'field_type': visit,
    },
    {
        'name': 'Pallidum_r',
        'source': 'field',
        'field_id': 25018,
        'field_type': visit,
    },
    {
        'name': 'Putamen_l',
        'source': 'field',
        'field_id': 25015,
        'field_type': visit,
    },
    {
        'name': 'Putamen_r',
        'source': 'field',
        'field_id': 25017,
        'field_type': visit,
    },
    {
        'name': 'Thalamus_l',
        'source': 'field',
        'field_id': 25011,
        'field_type': visit,
    },
    {
        'name': 'Thalamus_r',
        'source': 'field',
        'field_id': 25012,
        'field_type': visit,
    }
]

TRANSCRIPTOMICS = [
    {
        'name': 'Platelet_count',
        'source': 'field',
        'field_id': 30080,
        'field_type': visit,
    },
    {
        'name': 'Monocyte_count',
        'source': 'field',
        'field_id': 30130,
        'field_type': visit,
    },
    {
        'name': 'RedBloodCell_count',
        'source': 'field',
        'field_id': 30010,
        'field_type': visit,
    },
    {
        'name': 'WhiteBloodCell_count',
        'source': 'field',
        'field_id': 30000,
        'field_type': visit,
    },
    {
        'name': 'HighLightScatterReticulocyte_count',
        'source': 'field',
        'field_id': 30300,
        'field_type': visit,
    },
    {
        'name': 'Eosinophil_count',
        'source': 'field',
        'field_id': 30150,
        'field_type': visit,
    },
    {
        'name': 'Reticulocyte_count',
        'source': 'field',
        'field_id': 30250,
        'field_type': visit,
    },
    {
        'name': 'Lymphocyte_count',
        'source': 'field',
        'field_id': 30120,
        'field_type': visit,
    },
    {
        'name': 'Mean_platelet_volume',
        'source': 'field',
        'field_id': 30100,
        'field_type': visit,
    },
    {
        'name': 'Mean_corpuscular_volume',
        'source': 'field',
        'field_id': 30040,
        'field_type': visit,
    },
    {
        'name': 'Mean_corpuscular_hemoglobin',
        'source': 'field',
        'field_id': 30050,
        'field_type': visit,
    },
    {
        'name': 'Neutrophil_count',
        'source': 'field',
        'field_id': 30140,
        'field_type': visit,
    },
    {
        'name': 'RedCell_distributionwidth',
        'source': 'field',
        'field_id': 30070,
        'field_type': visit,
    },
    {
        'name': 'Platelet_distributionwidth',
        'source': 'field',
        'field_id': 30110,
        'field_type': visit,
    },
    {
        'name': 'HighLightScatterReticulocyte_percentage_RedCells',
        'source': 'field',
        'field_id': 30290,
        'field_type': visit,
    },
]

def get_IDPsFAST(visit=2):
    # get info about field ID and encodings
    overview = pd.read_csv(os.path.join(ukbb_paths.CODINGS_DIR,'Data_Dictionary_Showcase.csv'),sep=',',
                       encoding='latin-1',header=0,usecols=np.arange(17))
    # get the one for catgory of interest
    IDPsFAST = overview.loc[overview['Category']==1101,['FieldID','Field']]
    IDPsFAST['source'] = 'field'
    IDPsFAST['field_type'] = visit
    IDPsFAST = IDPsFAST.rename(columns={'FieldID':'field_id','Field':'name'})
    # rename so no whitespace of special characters
    IDPsFAST['name'] = IDPsFAST['name'].str.replace('Volume of grey matter in ','')
    IDPsFAST['name'] = IDPsFAST['name'].str.replace(' ','')
    IDPsFAST['name'] = IDPsFAST['name'].str.replace(',','_')
    IDPsFAST['name'] = IDPsFAST['name'].str.replace("'",'')
    IDPsFAST['name'] = IDPsFAST['name'].str.replace('(','_')
    IDPsFAST['name'] = IDPsFAST['name'].str.replace(')','')
    return IDPsFAST.to_dict('records')

IDPsFAST = get_IDPsFAST(visit)

def get_IDPsDKT(visit=2):
    # get info about field ID and encodings
    overview = pd.read_csv(os.path.join(ukbb_paths.CODINGS_DIR,'Data_Dictionary_Showcase.csv'),sep=',',
                       encoding='latin-1',header=0,usecols=np.arange(17))
    # get the one for catgory of interest
    IDPs = overview.loc[overview['Category']==196,['FieldID','Field']]
    IDPs['source'] = 'field'
    IDPs['field_type'] = visit
    IDPs = IDPs.rename(columns={'FieldID':'field_id','Field':'name'})
    # rename so no whitespace of special characters
    IDPs['name'] = IDPs['name'].str.replace(' of ','_')
    IDPs['name'] = IDPs['name'].str.replace('hemisphere','')
    IDPs['name'] = IDPs['name'].str.replace(',','_')
    IDPs['name'] = IDPs['name'].str.replace("'",'')
    IDPs['name'] = IDPs['name'].str.replace(' ','')
    IDPs['name'] = IDPs['name'].str.replace('(','_')
    IDPs['name'] = IDPs['name'].str.replace(')','')
    return IDPs.to_dict('records')

IDPsDKT = get_IDPsDKT(visit)

def get_IDPsDesikianGW(visit=2):
    # get info about field ID and encodings
    overview = pd.read_csv(os.path.join(ukbb_paths.CODINGS_DIR,'Data_Dictionary_Showcase.csv'),sep=',',
                       encoding='latin-1',header=0,usecols=np.arange(17))
    # get the one for catgory of interest
    IDPs = overview.loc[overview['Category']==194,['FieldID','Field']]
    IDPs['source'] = 'field'
    IDPs['field_type'] = visit
    IDPs = IDPs.rename(columns={'FieldID':'field_id','Field':'name'})
    # rename so no whitespace of special characters
    IDPs['name'] = IDPs['name'].str.replace(' of ','_')
    IDPs['name'] = IDPs['name'].str.replace('hemisphere','')
    IDPs['name'] = IDPs['name'].str.replace('Grey-white contrast in ','')
    IDPs['name'] = IDPs['name'].str.replace(',','_')
    IDPs['name'] = IDPs['name'].str.replace("'",'')
    IDPs['name'] = IDPs['name'].str.replace(' ','')
    IDPs['name'] = IDPs['name'].str.replace('(','_')
    IDPs['name'] = IDPs['name'].str.replace(')','')
    return IDPs.to_dict('records')

IDPsDesikianGW = get_IDPsDesikianGW(visit)

def get_IDPsDWIskeleton(visit=2):
    # get info about field ID and encodings
    overview = pd.read_csv(os.path.join(ukbb_paths.CODINGS_DIR,'Data_Dictionary_Showcase.csv'),sep=',',
                       encoding='latin-1',header=0,usecols=np.arange(17))
    # get the one for catgory of interest
    IDPs = overview.loc[overview['Category']==134,['FieldID','Field']]
    IDPs['source'] = 'field'
    IDPs['field_type'] = visit
    IDPs = IDPs.rename(columns={'FieldID':'field_id','Field':'name'})
    # rename so no whitespace of special characters
    IDPs['name'] = IDPs['name'].str.replace('on FA skeleton','')
    IDPs['name'] = IDPs['name'].str.replace(' in ','')
    IDPs['name'] = IDPs['name'].str.replace(',','_')
    IDPs['name'] = IDPs['name'].str.replace("'",'')
    IDPs['name'] = IDPs['name'].str.replace(' ','')
    IDPs['name'] = IDPs['name'].str.replace('(','_')
    IDPs['name'] = IDPs['name'].str.replace(')','')
    return IDPs.to_dict('records')

IDPsDWIskeleton = get_IDPsDWIskeleton(visit)

def get_IDPsDWImean(visit=2):
    # get info about field ID and encodings
    overview = pd.read_csv(os.path.join(ukbb_paths.CODINGS_DIR,'Data_Dictionary_Showcase.csv'),sep=',',
                       encoding='latin-1',header=0,usecols=np.arange(17))
    # get the one for catgory of interest
    IDPs = overview.loc[overview['Category']==135,['FieldID','Field']]
    IDPs['source'] = 'field'
    IDPs['field_type'] = visit
    IDPs = IDPs.rename(columns={'FieldID':'field_id','Field':'name'})
    # rename so no whitespace of special characters
    IDPs['name'] = IDPs['name'].str.replace(' in tract ','')
    IDPs['name'] = IDPs['name'].str.replace(',','_')
    IDPs['name'] = IDPs['name'].str.replace("'",'')
    IDPs['name'] = IDPs['name'].str.replace(' ','')
    IDPs['name'] = IDPs['name'].str.replace('(','_')
    IDPs['name'] = IDPs['name'].str.replace(')','')
    return IDPs.to_dict('records')

IDPsDWImean = get_IDPsDWImean(visit)

def get_IDPsSWI(visit=2):
    # get info about field ID and encodings
    overview = pd.read_csv(os.path.join(ukbb_paths.CODINGS_DIR,'Data_Dictionary_Showcase.csv'),sep=',',
                       encoding='latin-1',header=0,usecols=np.arange(17))
    # get the one for catgory of interest
    IDPs = overview.loc[overview['Category']==109,['FieldID','Field']].iloc[2:-2,:]
    IDPs['source'] = 'field'
    IDPs['field_type'] = visit
    IDPs = IDPs.rename(columns={'FieldID':'field_id','Field':'name'})
    # rename so no whitespace of special characters
    IDPs['name'] = IDPs['name'].str.replace(' in ','')
    IDPs['name'] = IDPs['name'].str.replace(',','_')
    IDPs['name'] = IDPs['name'].str.replace("'",'')
    IDPs['name'] = IDPs['name'].str.replace(' ','')
    IDPs['name'] = IDPs['name'].str.replace('(','_')
    IDPs['name'] = IDPs['name'].str.replace(')','')
    return IDPs.to_dict('records')

IDPsSWI = get_IDPsSWI(visit)

def get_BloodCounts(visit=2):
    # get info about field ID and encodings
    overview = pd.read_csv(os.path.join(ukbb_paths.CODINGS_DIR,'Data_Dictionary_Showcase.csv'),sep=',',
                       encoding='latin-1',header=0,usecols=np.arange(17))
    # get the one for catgory of interest
    IDPs = overview.loc[overview['Category']==100081,['FieldID','Field']]
    IDPs['source'] = 'field'
    IDPs['field_type'] = visit
    IDPs = IDPs.rename(columns={'FieldID':'field_id','Field':'name'})
    # rename so no whitespace of special characters
    IDPs['name'] = IDPs['name'].str.replace(' percentage','_percentage')
    IDPs['name'] = IDPs['name'].str.replace(' count','_count')
    IDPs['name'] = IDPs['name'].str.replace(' width','_width')
    IDPs['name'] = IDPs['name'].str.replace(' volume','_volume')
    IDPs['name'] = IDPs['name'].str.replace(' concentration','_concentration')
    IDPs['name'] = IDPs['name'].str.replace(' crit','_crit')
    IDPs['name'] = IDPs['name'].str.replace(',','_')
    IDPs['name'] = IDPs['name'].str.replace("'",'')
    IDPs['name'] = IDPs['name'].str.replace(' ','')
    IDPs['name'] = IDPs['name'].str.replace('(','_')
    IDPs['name'] = IDPs['name'].str.replace(')','')
    return IDPs.to_dict('records')

CellCounts = get_BloodCounts(visit)

def get_BloodChemistry(visit=2):
    # get info about field ID and encodings
    overview = pd.read_csv(os.path.join(ukbb_paths.CODINGS_DIR,'Data_Dictionary_Showcase.csv'),sep=',',
                       encoding='latin-1',header=0,usecols=np.arange(17))
    # get the one for catgory of interest
    IDPs = overview.loc[overview['Category']==17518,['FieldID','Field']]
    IDPs['source'] = 'field'
    IDPs['field_type'] = visit
    IDPs = IDPs.rename(columns={'FieldID':'field_id','Field':'name'})
    # rename so no whitespace of special characters
    IDPs['name'] = IDPs['name'].str.replace(',','_')
    IDPs['name'] = IDPs['name'].str.replace('-','_')
    IDPs['name'] = IDPs['name'].str.replace("'",'')
    IDPs['name'] = IDPs['name'].str.replace(' ','')
    IDPs['name'] = IDPs['name'].str.replace('(','_')
    IDPs['name'] = IDPs['name'].str.replace(')','')
    return IDPs.to_dict('records')

BloodChemistry = get_BloodChemistry(visit)

LIFESTYLE = [
    {
        'name': 'AlcoholFrequency', # encoding 100402: 1 Daily or almost daily 2 Three or four times a week 3 Once or twice a week 4 One to three times a month 5 Special occasions only 6 Never -3 Prefer not to answer
        'source': 'field',
        'field_type': visit,
        'field_id': 1558
    },
    {
        'name': 'AlcoholStatus', # encoding 90: -3 Prefer not to answer 0 Never 1 Previous 2 Current
        'source': 'field',
        'field_type': visit,
        'field_id': 20117
    },
    {
        'name': 'SmokePackyears',
        'source': 'field',
        'field_type': visit,
        'field_id': 20161
    },
    {
        'name': 'SmokeStatus', # encoding 90: -3 Prefer not to answer 0 Never 1 Previous 2 Current
        'source': 'field',
        'field_type': visit,
        'field_id': 20116
    },
    {
        'name': 'DaytimeSleepiness', # encoding: 100346
        'source': 'field',
        'field_type': visit,
        'field_id': 1220
    },
    {
        'name': 'Pesticides', # encoding 493: -141 Often -131 Sometimes 0 Rarely/never -121 Do not know
        'source': 'field',
        'field_type': visit,
        'field_id': 22614
    },
    
]

FAMILY = [
    {
        'name': 'IllnessFather', # encoding 1010
        'source': 'field',
        'field_type': visit,
        'field_id': 20107
    },
        {
        'name': 'IllnessMother', # encoding 1010
        'source': 'field',
        'field_type': visit,
        'field_id': 20110
    },
        { 
        'name': 'IllnessSiblings', # encoding 1010
        'source': 'field',
        'field_type': visit,
        'field_id': 20111
    },
]