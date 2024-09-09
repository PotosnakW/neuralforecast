import pandas as pd
import numpy as np
import pickle
import os

def make_start_end_df(df, feature, freq):

    df.ts_end = df.ts_end - pd.Timedelta(freq)

    df_periods = pd.DataFrame()
    for row in df.iterrows():
        test = pd.DataFrame(np.repeat(row[1][feature], 2), index=[row[1].ts_begin, row[1].ts_end])
        test = test.resample(freq).ffill()
        df_periods = pd.concat([df_periods, test], axis=0)
     
    df_periods.columns = [feature]
    df_periods.index.name = 'ts'
    return df_periods

def get_bolus_timestamp_doses(square_dual):
    
    square_dual = square_dual.copy()
    square_dual.dose = square_dual.dose / (square_dual.ts_end - \
                                           square_dual.ts_begin).astype('timedelta64[m]')
    
    bolus_periods = make_start_end_df(square_dual, 'dose')
    bolus_periods.index.name = 'ts_begin'
    bolus_periods.columns = ['dose']
    
    return bolus_periods

def preprocess_insulin_ts_df(df):
    df.ts_begin = pd.to_datetime(df.ts_begin, format='%d-%m-%Y %H:%M:%S')
    df.ts_end = pd.to_datetime(df.ts_end, format='%d-%m-%Y %H:%M:%S')
    df.ts_begin = df.ts_begin.round('min')
    df.ts_end = df.ts_end.round('min')
    
    return df
    
def preprocess_ts_df(df):
    df.ts = pd.to_datetime(df.ts, format='%d-%m-%Y %H:%M:%S')
    df.ts = df.ts.round('min')
    df.set_index('ts', inplace=True)
    
    return df


data_dir = './OhioT1DM'
os.makedirs('./data', exist_ok=True)

pat_ids = ['559', '563', '570', '575', '588', '591', '540', '544', '552', '567', '584', '596']
years = ['2018']*6+['2020']*6

pat_dataset = pd.DataFrame()
test_size = {}

for pat_id, year in zip(pat_ids, years):
    print(pat_id, year)
    
    for dataset_type in ['train', 'test']:
        print(dataset_type)
    
        id_ = pd.read_xml(data_dir+f'/{year}/{dataset_type}/{pat_id}-ws-{dataset_type}ing.xml', xpath="/patient")[['id']]

        #--------------------------------------------------------------CGM_DATA--------------------------------------------------------------
        
        cgm = pd.read_xml(data_dir+f'/{year}/{dataset_type}/{pat_id}-ws-{dataset_type}ing.xml', xpath="./glucose_level/event")
        cgm = preprocess_ts_df(cgm)

        cgm.set_index(cgm.index.ceil('5Min'), inplace=True)
        cgm = cgm[~cgm.index.duplicated(keep='last')]
        cgm_timestamps = cgm.index.values
        
        t_index = pd.DatetimeIndex(pd.date_range(start=cgm.index[0], 
                                                 end=cgm.index[-1], 
                                                 freq="5Min"))
        same_index = cgm.index.intersection(t_index).values
        
        cgm_resamp = pd.DataFrame(index=t_index)
        cgm_resamp.index.name = 'ts'
        cgm_resamp['y'] = np.nan
        cgm_resamp.loc[same_index] = cgm.loc[same_index].value.values.reshape(-1, 1)
        cgm_resamp = cgm_resamp.ffill()
        cgm_resamp['available_mask'] = 0
        cgm_resamp.loc[same_index, 'available_mask'] = 1
        
        cgm_resamp_start_time = cgm_resamp.index[0]
        cgm_resamp_end_time = cgm_resamp.index[-1]
        
        #print('Day Count:', cgm_resamp.shape[0]/288)

        #--------------------------------------------------------------MEAL_DATA--------------------------------------------------------------
        
        if (pat_id == '567') & (dataset_type=='test'):
            meal = pd.DataFrame(np.zeros(cgm_resamp.shape[0]), index=cgm_resamp.index)
            meal.columns = ['CHO']
            
        else:
            meal = pd.read_xml(data_dir+f'/{year}/{dataset_type}/{pat_id}-ws-{dataset_type}ing.xml', xpath="./meal/event")[['ts', 'carbs']]
            meal = preprocess_ts_df(meal)
            meal.columns = ['CHO']
            meal.set_index(meal.index.ceil('5Min'), inplace=True)
            meal = meal[(meal.index>=cgm_resamp_start_time) & (meal.index<=cgm_resamp_end_time)]
            meal.index.name = 'ts'
            meal = meal.groupby('ts').sum()

        #--------------------------------------------------------------BASAL_DATA--------------------------------------------------------------
        
        basal = pd.read_xml(data_dir+f'/{year}/{dataset_type}/{pat_id}-ws-{dataset_type}ing.xml', xpath="./basal/event")
        basal = preprocess_ts_df(basal)
        
        if basal.index[-1]<cgm_resamp_end_time:
            new_end_time = pd.DataFrame([basal.value.values[-1]], index=[cgm_resamp_end_time], columns=['value'])
            new_end_time.index.name = 'ts'
            basal = pd.concat([basal, new_end_time], axis=0)
        
        basal.set_index(basal.index.ceil('1H'), inplace=True)
        basal = basal[~basal.index.duplicated(keep='last')]
        basal = basal.resample('1H').ffill()
        basal_idx_overlap = basal.index.intersection(cgm_resamp.index).values
        basal = basal.loc[basal_idx_overlap]
        
        if ((dataset_type == 'test') & (pat_id not in ['563', '570', '584'])) | (dataset_type == 'train'):
            temp_basal = pd.read_xml(data_dir+f'/{year}/{dataset_type}/{pat_id}-ws-{dataset_type}ing.xml', xpath="./temp_basal/event")
            temp_basal = preprocess_insulin_ts_df(temp_basal)
            temp_basal = make_start_end_df(temp_basal, 'value', '1Min')
            
            #basal rate may change, so use last rate. Basal is generally continuous so do not sum.
            temp_basal = temp_basal[~temp_basal.index.duplicated(keep='last')]
            temp_basal_idx_overlap = basal.index.intersection(temp_basal.index).values
            basal.loc[temp_basal_idx_overlap] = temp_basal

        basal.columns = ['basal_insulin']
        basal = basal.round(4)
        basal = basal[(basal.index>=cgm_resamp_start_time) &\
                      (basal.index<=cgm_resamp_end_time)]

        #--------------------------------------------------------------BOLUS_DATA--------------------------------------------------------------
        
        bolus = pd.read_xml(data_dir+f'/{year}/{dataset_type}/{pat_id}-ws-{dataset_type}ing.xml', xpath="./bolus/event")
        bolus = preprocess_insulin_ts_df(bolus)
        
        # Normal dual doses have same begin and end timestamps for all patients with normal dual doses.
        # Normal dual are thus treated like normal insulin boluses
        normal_bolus = bolus[(bolus.type == 'normal')|(bolus.type == 'normal dual')][['ts_begin', 'dose']]
        normal_bolus.set_index('ts_begin', inplace=True)
        normal_bolus.set_index(normal_bolus.index.ceil('5Min'), inplace=True)
        
        square_dual = bolus[(bolus.type == 'square dual')][['ts_begin', 'ts_end', 'dose']]
        if square_dual.shape[0] > 0:
            square_dual.ts_begin = square_dual.set_index('ts_begin').index.ceil('5Min')
            square_dual.ts_end = square_dual.set_index('ts_end').index.ceil('5Min')
            square_dual.dose = square_dual.dose/((square_dual.ts_end - square_dual.ts_begin)/pd.Timedelta('5Min'))
            square_dual = make_start_end_df(square_dual, 'dose', '5min')

        resamp_bolus = pd.DataFrame(pd.concat([normal_bolus.dose, square_dual.dose], axis=0))
        resamp_bolus.columns = ['bolus_insulin']
        resamp_bolus.index.name = 'ts'
        resamp_bolus = resamp_bolus.groupby('ts').sum()
        resamp_bolus.sort_index(inplace=True)
        resamp_bolus = resamp_bolus[(resamp_bolus.index>=cgm_resamp_start_time) &\
                                    (resamp_bolus.index<=cgm_resamp_end_time)]

        #--------------------------------------------------------------COMBINE_DATA--------------------------------------------------------------
        
        pat_data = pd.concat([cgm_resamp, meal, basal, resamp_bolus], axis=1)
        pat_data.index.name = 'ds'
        pat_data.reset_index(inplace=True, drop=False)
        
        pat_data.loc[pat_data.CHO.isnull(), 'CHO']=0
        pat_data.loc[pat_data.basal_insulin.isnull(), 'basal_insulin']=0
        pat_data.loc[pat_data.bolus_insulin.isnull(), 'bolus_insulin']=0
    
        pat_data.sort_index(inplace=True)
        pat_data['unique_id'] = '#'+str(id_.values[0][0])

        if dataset_type == 'test':
            pat_data = pat_data.iloc[:2691, :]
            
            if pat_data.ds[0] == pat_dataset.ds.iloc[-1]:
                pat_dataset.drop(index=[pat_dataset.iloc[-1].name], inplace=True)

        pat_dataset = pd.concat([pat_dataset, pat_data], axis=0)

pat_dataset = pat_dataset[['unique_id', 'ds', 'y', 'available_mask', 'CHO', 'basal_insulin', 'bolus_insulin']]
pat_dataset.to_csv('./data/ohiot1dm_exog_9_day_test.csv', index=False)


## Evaluation: Mark cutoff times with no new info
data = pat_dataset.copy()
df = []
unique_ids = data['unique_id'].unique()
for unique_id in unique_ids:
    df_uid = data[data['unique_id'] == unique_id].reset_index(drop=True)
    df_uid["sum_av_mask"] = df_uid['available_mask'].rolling(window=120, min_periods=1).sum() # ADJUSTABLE: MODEL INPUT SIZE == 120
    df.append(df_uid)
av_mask = pd.concat(df).reset_index(drop=True)
av_mask = av_mask.rename(columns={'ds': 'cutoff'})
av_mask.head()
av_mask.to_csv('./data/ohiot1dm_exog_9_day_test_avmask.csv', index=False)


## Static dataset
it = {}
for pat_id, year in zip(pat_ids, years):
    t = pd.read_xml(data_dir+f'/{year}/train/{pat_id}-ws-training.xml', xpath="/patient")[['insulin_type']].values[0][0]
    t = t.replace(' 200', '')
    it['#'+pat_id] = t

insulin_type = pd.DataFrame(it.values(), index=it.keys())
insulin_type.columns = ['insulin_type_novalog']
insulin_type.index.name = 'unique_id'
insulin_type.index = insulin_type.index.astype('str')
insulin_type.loc[insulin_type.insulin_type_novalog == 'Humalog'] = 0
insulin_type.loc[insulin_type.insulin_type_novalog == 'Novalog'] = 1

onehot = np.diag((np.repeat(1, len(pat_dataset.unique_id.unique()))))
df = pd.DataFrame(onehot, index=pat_dataset.unique_id.unique())
df.columns = pat_dataset.unique_id.unique()
df.index.name = 'unique_id'
df = df.iloc[:, :-1]

static = pd.concat([df, insulin_type], axis=1)

static['female'] = np.zeros(static.shape[0])
static.loc[['#567', '#559', '#575', '#588', '#591'], 'female'] = 1

static['age_20_40'] = np.zeros(static.shape[0])
static.loc[['#540', '#552', '#567'], 'age_20_40'] = 1

static['age_40_60'] = np.zeros(static.shape[0])
static.loc[['#544', '#584', '#559', '#563', '#570', '#575', '#588', '#591'], 'age_40_60'] = 1

static['pump_model_630G'] = np.zeros(static.shape[0])
static.loc[['#540', '#552', '#567'], 'pump_model_630G'] = 1

static.reset_index(inplace=True, drop=False)
static.to_csv('./data/ohiot1dm_static.csv', index=False)
