import pandas as pd
import numpy as np
import random
from statsmodels.tsa.stattools import acf, adfuller
import torch
import heapq


def get_data_samples(data, zero_count_threshold, alpha):
    _ids = data.unique_id.unique()
    lags = 200
    
    macs = {}
    y_subsets = {}
    ds_subsets = {}
    for fi, id_ in enumerate(_ids):
        if fi % 100 == 0: print(fi)
    
        signal = data[data.unique_id==str(id_)].y.values
        timestamps = data[data.unique_id==str(id_)].ds.values
        timestamps = pd.to_datetime(timestamps)
        timestamps = timestamps.astype('int64') // 10**6 #convert to milliseconds
        unfolded_y = torch.tensor(signal).unfold(dimension=-1, size=1056, step=528)
        unfolded_ds = torch.tensor(timestamps).unfold(dimension=-1, size=1056, step=528)
        
        for fi, (fold_y, fold_ds) in enumerate(zip(unfolded_y, unfolded_ds)):
            # Check for flatline periods
            rolling_std = pd.Series(fold_y).rolling(window=30).std()
            if 0 in rolling_std.value_counts().index.values:
                zero_counts = rolling_std.value_counts()[0]
                if zero_counts > zero_count_threshold: continue

            pval= Augmented_Dickey_Fuller_Test_func(fold_y.detach().numpy())['p-value']
            if pval >= alpha: continue 
            
            acf_values = acf(fold_y, nlags=lags, fft=True) 
            mac = np.mean(np.abs(acf_values[1:]))  # Exclude lag 0 (always 1.0)

            macs[id_+f'_{fi}'] = mac
            y_subsets[id_+f'_{fi}'] = fold_y
            ds_subsets[id_+f'_{fi}'] = pd.to_datetime(fold_ds, unit='ms')

    return macs, y_subsets, ds_subsets

def Augmented_Dickey_Fuller_Test_func(series):
    dftest = adfuller(series, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','No Lags Used','Number of observations used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value

    return dfoutput

def make_df(random_sample, ts, ds):
    df = pd.DataFrame()
    for i in random_sample:
        i_df = pd.DataFrame([ds[i], ts[i].detach().numpy()], index=['ds', 'y']).T
        i_df['unique_id'] = i
        i_df = i_df[['unique_id', 'ds', 'y']]
        df = pd.concat([df, i_df], axis=0)

    return df


N_series = 100

# ettm2 dataset
data = pd.read_csv('./data/ETTm2_df_y.csv')
data = data[(data.unique_id=='OT')|(data.unique_id=='HUFL')|(data.unique_id=='HULL')|(data.unique_id=='MUFL')]
macs, y_subsets, ds_subsets = get_data_samples(data, 10, 0.001)
random_sample = heapq.nlargest(N_series, macs, key=macs.get)
df = make_df(random_sample, y_subsets, ds_subsets)
df.to_csv('./data/ETTm2_100_series.csv', index=False)

# ecl dataset
data = pd.read_csv('./data/ECL_df_y.csv')
macs, y_subsets, ds_subsets= get_data_samples(data, 10, 0.001)
random_sample = heapq.nlargest(N_series, macs, key=macs.get)
df = make_df(random_sample, y_subsets, ds_subsets)
df.to_csv('./data/ECL_100_series.csv', index=False)

# solar dataset
data = pd.read_csv('./data/solar_1h_dataset.csv')
unique_id_subset = data.unique_id.unique()[:3000]
data = data.set_index('unique_id').loc[unique_id_subset]
data.reset_index(inplace=True, drop=False)
macs, y_subsets, ds_subsets = get_data_samples(data, 30, 0.001)
random_sample = heapq.nlargest(N_series, macs, key=macs.get)
df = make_df(random_sample, y_subsets, ds_subsets)
df.to_csv('./data/solar1h_100_series2.csv', index=False)

# subseasonal dataset
data = pd.read_csv('./data/subseasonal.csv')
data['ds'] = pd.to_datetime(data.ds, format='%Y-%m-%d %H:%M:%S')
data = data.groupby('unique_id').resample('W', on='ds')['y'].mean().reset_index()
macs, y_subsets, ds_subsets = get_data_samples(data, 10, 0.001)
random_sample = heapq.nlargest(N_series, macs, key=macs.get)
df = make_df(random_sample, y_subsets, ds_subsets)
df.to_csv('./data/subseasonal_100_series.csv', index=False)

# loop seattle dataset
data = pd.read_csv('./data/loopseattle.csv')
data['ds'] = pd.to_datetime(data.ds, format='%Y-%m-%d %H:%M:%S')
data = data.groupby('unique_id').resample('W', on='ds')['y'].mean().reset_index()
macs, y_subsets, ds_subsets = get_data_samples(data, 10, 0.001)
random_sample = heapq.nlargest(N_series, macs, key=macs.get)
df = make_df(random_sample, y_subsets, ds_subsets)
df.to_csv('./data/loopseattle_100_series.csv', index=False)
