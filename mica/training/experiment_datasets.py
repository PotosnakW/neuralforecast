import pandas as pd
import numpy as np
from gift_eval.data import Dataset
from sklearn.decomposition import PCA, FastICA


gifteval_eval_dataset_names = [
 'LOOP_SEATTLE/5T',
 'LOOP_SEATTLE/D',
 'LOOP_SEATTLE/H',
 'M_DENSE/D',
 'M_DENSE/H',
 'SZ_TAXI/15T',
 'SZ_TAXI/H',
 'bitbrains_fast_storage/5T',
 'bitbrains_fast_storage/H',
 'bitbrains_rnd/5T',
 'bitbrains_rnd/H',
 'bizitobs_application',
 'bizitobs_l2c/5T',
 'bizitobs_l2c/H',
 'bizitobs_service',
 #'car_parts_with_missing', #Not enough samples for training
 'covid_deaths',
 'electricity/15T',
 'electricity/D',
 'electricity/H',
 'electricity/W',
 'ett1/15T',
 'ett1/D',
 'ett1/H',
 'ett1/W',
 'ett2/15T',
 'ett2/D',
 'ett2/H',
 'ett2/W',
 'hierarchical_sales/D',
 'hierarchical_sales/W',
 'hospital',
 'jena_weather/10T',
 'jena_weather/D',
 'jena_weather/H',
 'kdd_cup_2018_with_missing/D',
 'kdd_cup_2018_with_missing/H',
 'm4_daily',
 'm4_hourly',
 'm4_monthly',
 'm4_quarterly',
 'm4_weekly',
 'm4_yearly',
 'restaurant',
 'saugeenday/D',
 'saugeenday/M',
 'saugeenday/W',
 'solar/10T',
 'solar/D',
 'solar/H',
 'solar/W',
 'temperature_rain_with_missing',
 'us_births/D',
 'us_births/M',
 'us_births/W'
]


def get_datasets(args):
    
    if args.dataset_name in gifteval_eval_dataset_names:
        to_univariate = False  # Whether to convert the data to univariate
        term = "short"  # Term of the dataset
        dataset = Dataset(name=args.dataset_name, term=term, to_univariate=to_univariate)

        train_df = preprocess_gifteval_dataset_for_neuralforecast(args.dataset_name, 
                                                                  dataset_type='train',
        )
        test_df = preprocess_gifteval_dataset_for_neuralforecast(args.dataset_name, 
                                                                 dataset_type='test',
                                                                 ids=train_df.unique_id.unique(),
        )

        df = pd.concat([train_df, test_df], axis=0)
        df.sort_values(by=['unique_id', 'ds'], ascending=True, inplace=True)
        df = df.drop_duplicates(subset=["unique_id", "ds"], keep="last")
        h = dataset.prediction_length
        val_size = dataset.prediction_length
        test_size = int(dataset.prediction_length*dataset.windows)
        freq = dataset.freq

    elif args.dataset_name in gifteval_pretrain_dataset_names:
        to_univariate = False  # Whether to convert the data to univariate
        term = "short"  # Term of the dataset
        dataset = Dataset(name=args.dataset_name, term=term, to_univariate=to_univariate)

        df = preprocess_gifteval_dataset_for_neuralforecast(args.dataset_name, 
                                                                  dataset_type='train',
        )
        h = dataset.prediction_length
        val_size = dataset.prediction_length
        test_size = int(dataset.prediction_length*dataset.windows)
        freq = dataset.freq

    elif args.dataset_name in gluonts_datasets.keys():
        df_train = gluonts_to_long_dataframe(args.dataset_name, split='train')
        df_test = gluonts_to_long_dataframe(args.dataset_name, split='test')

        df = pd.concat([df_train, df_test], axis=0)
        df.sort_values(by=['unique_id', 'ds'], ascending=True, inplace=True)
        df = df.drop_duplicates(subset=["unique_id", "ds"], keep="last")

        h = gluonts_datasets[args.dataset_name]['prediction_length']
        val_size = gluonts_datasets[args.dataset_name]['prediction_length']
        test_size = int(h*2-1) # Enable n_windows=h for rolling window predictions
        freq = gluonts_datasets[args.dataset_name]['freq']

    elif args.dataset_name == 'simglucose':
        df = pd.read_csv('../datasets/simglucose_90_days.csv')
        h = 6
        val_size = 2592 # 9 days (10% of data)
        test_size = 2592 # 9 days (10% of data)
        freq = '5min'

    elif args.dataset_name == 'iowa_ihop_smex_windspeed':
        df = pd.read_csv('../datasets/preprocessed_iowa_ihop_smex02_dataset.csv')
        df.ds = pd.to_datetime(df.ds, format='%Y-%m-%d %H:%M:%S')
        df = df.groupby('unique_id').resample('5min', on='ds')['y'].mean().reset_index()
        h = 24
        val_size = 2304 #8 days (10% of data)
        test_size = 2304 #8 days (10% of data)
        freq = '5min'
        
    elif args.dataset_name == 'iowa_plows_windspeed':
        df = pd.read_csv('../datasets/preprocessed_iowa_plows_dataset.csv')
        df.ds = pd.to_datetime(df.ds, format='%Y-%m-%d %H:%M:%S')
        df = df.groupby('unique_id').resample('5min', on='ds')['y'].mean().reset_index()
        h = 24
        val_size = 3744 #13 days (10% of data)
        test_size = 3744 #13 days (10% of data)
        freq = '5min'
        
    elif args.dataset_name == 'wind_global_hourly':
        df = pd.read_csv('../datasets/wind_global_hourly.csv')
        h = 24
        val_size = 1752 # 1755 Timer-XL val size
        test_size = 3504 #3509 Timer-XL test size
        freq = 'H'
        
    elif args.dataset_name == 'temp_global_hourly':
        df = pd.read_csv('../datasets/temp_global_hourly.csv')
        h = 24
        val_size = 1752 # 1755 Timer-XL val size
        test_size = 3504 #3509 Timer-XL test size
        freq = 'H'

    elif args.dataset_name == 'synthetic_windspeed':
        df = pd.read_csv('../datasets/synthetic_windgust_dataset.csv')
        h = 24
        val_size = 68600 # (10% of 686000 timepoints)
        test_size = 68600 # (10% of 686000 timepoints)
        freq = 'min'

    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    return df, h, val_size, test_size, freq


def preprocess_gifteval_dataset_for_neuralforecast(
        dataset_name, 
        dataset_type, 
        ids=None, 
        to_univariate=False,
        term='short'
    ):
    '''
    inputs:
        - dataset_name: str, name of dataset.
        - dataset_type: str=['train','test'], Specify dataset.
        - ids: list of ids to keep in dataset (generally used for dataset_type='test'). 
    return: preprocessed Dataframe with 'unique_id', 'ds', and 'y' columns.
    '''
    dataset = Dataset(
        name=dataset_name, 
        term=term, 
        to_univariate=to_univariate
    )
    
    if dataset_type == 'train':
        data_list = list(dataset.validation_dataset)
    elif dataset_type == 'test':
        data_list = [i[1] for i in dataset.test_data]
    else:
        raise ValueError("dataset_type must be 'train' or 'test'")
    
    df = pd.DataFrame(data_list)
    
    # known series with all nans
    PROBLEMATIC_SERIES = {
        'electricity': ['MT_178'],
        'bitbrains_fast_storage': ['fastStorage_552']
    }
    
    for dataset_key, series_ids in PROBLEMATIC_SERIES.items():
        if dataset_key in dataset.name:
            df = df[~df.item_id.isin(series_ids)]
    
    n_targets = [i.shape for i in df.target.values]
    has_multiple_targets = any(len(shape) == 2 for shape in n_targets)
    
    if has_multiple_targets:
        print('more than one target')
        dfe_list = []
        for n, target in enumerate(n_targets):
            dfe = pd.DataFrame(df.iloc[n]).T.explode("target")
            dfe_n_series = dfe.shape[0]
            dfe['item_id'] = [f'{dfe.item_id.values[0]}_{j}' for j in range(dfe_n_series)]
            dfe_list.append(dfe)
        dfe_all = pd.concat(dfe_list, axis=0, ignore_index=True)
    else:
        print('one target')
        dfe_all = df
    
    # expand data to get stacked unique_id values
    df_expanded = dfe_all.explode("target")
    df_expanded.reset_index(inplace=True, drop=True)
    
    metadata = dfe_all.groupby('item_id')[['start', 'freq']].first()
    for uid in metadata.index:
        mask = df_expanded.item_id == uid
        start = metadata.loc[uid, 'start'].to_timestamp()
        freq = metadata.loc[uid, 'freq']
        periods = mask.sum()
        df_expanded.loc[mask, 'ds'] = pd.date_range(start=start, periods=periods, freq=freq)
    
    df_expanded.drop(columns=['start', 'freq'], inplace=True)
    df_expanded.rename(columns={'target': 'y', 'item_id':'unique_id'}, inplace=True)
    df_expanded = df_expanded[['unique_id', 'ds', 'y']]
    df_expanded.reset_index(drop=True, inplace=True)
    
    df_expanded['available_mask'] = 0
    one_idxs = np.where(df_expanded["y"].notnull())[0]
    df_expanded.loc[one_idxs, "available_mask"] = 1
    df_expanded.y = df_expanded.groupby('unique_id')['y'].ffill().fillna(0)
    
    if (dataset_type == 'test') and (ids is not None):
        df_expanded = df_expanded[df_expanded['unique_id'].isin(ids)]
    
    print('n_series length:', df_expanded.shape[0]/df_expanded.unique_id.unique().shape[0])
    
    return df_expanded

def decorrelate_data(df, val_size, test_size, method='pca', **kwargs):
    n_series = df['unique_id'].nunique()
    original_unique_ids = df['unique_id'].unique().tolist()
    
    # Split
    train_set = df.groupby('unique_id').apply(
        lambda x: x.iloc[:-(val_size+test_size)]
    ).reset_index(drop=True)
    end_set = df.groupby('unique_id').tail(val_size + test_size)
    validation_set = end_set.groupby('unique_id').head(val_size).reset_index(drop=True)
    test_set = end_set.groupby('unique_id').tail(test_size).reset_index(drop=True)

    # Check train set lengths and truncate if needed
    train_lengths = train_set.groupby('unique_id').size()
    if train_lengths.nunique() != 1:
        min_train_length = train_lengths.min()
        print(f"Warning: Train set has different lengths.")
        print(f"Truncating train set to {min_train_length} time steps (keeping most recent).")
        
        train_set = train_set.groupby('unique_id').tail(min_train_length).reset_index(drop=True)
    
        new_train_lengths = train_set.groupby('unique_id').size()
        assert new_train_lengths.nunique() == 1, "Train set truncation failed"
    
    # Preserve originals
    for s in (train_set, validation_set, test_set):
        s['y_original'] = s['y'].copy()
    
    # Add row indices for each split
    train_set['row'] = train_set.groupby('unique_id').cumcount()
    validation_set['row'] = validation_set.groupby('unique_id').cumcount()
    test_set['row'] = test_set.groupby('unique_id').cumcount()
    
    # Pivot to matrix form
    train_pivot = train_set.pivot(index='row', columns='unique_id', values='y')
    train_pivot = train_pivot[original_unique_ids]
    column_order = original_unique_ids
    
    # Standardize
    train_mean = train_pivot.mean(axis=0)
    train_std = train_pivot.std(axis=0)
    train_std = train_std.replace(0, 1e-4) # Avoid division by zero for pca
    train_standardized = (train_pivot - train_mean) / train_std
    
    # Choose transformer
    if method.lower() == 'pca':
        print('using PCA')
        if 'whiten' not in kwargs:
            kwargs['whiten'] = True
        transformer = PCA(n_components=n_series, **kwargs)
    elif method.lower() == 'ica':
        print('using ICA')
        # ICA defaults
        if 'max_iter' not in kwargs:
            kwargs['max_iter'] = 200
        transformer = FastICA(n_components=n_series, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'pca' or 'ica'")
    
    # Fit transformer
    transformer.fit(train_standardized.values)
    
    def transform_split(split):
        pivot = split.pivot(index='row', columns='unique_id', values='y')[column_order]
        standardized = (pivot - train_mean) / train_std
        transformed = transformer.transform(standardized.values)
        
        result = split.copy()
        for i, orig_id in enumerate(column_order):
            mask = result['unique_id'] == orig_id
            result.loc[mask, 'y'] = transformed[:, i]
        
        result.drop(columns=['row'], inplace=True)
        return result
    
    train_df = transform_split(train_set)
    val_df = transform_split(validation_set)
    test_df = transform_split(test_set)
    
    # Combine
    combined_set = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined_set.sort_values(['unique_id', 'ds'], ascending=True, inplace=True)
    combined_set.reset_index(drop=True, inplace=True)
    
    return combined_set, transformer, train_mean, train_std, column_order
