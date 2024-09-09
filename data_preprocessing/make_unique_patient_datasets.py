import pandas as pd
import os

def get_unique_id_datasets(dataset):
    data = pd.read_csv(f'../data/{dataset}_exog_9_day_test.csv')
    os.makedirs(f'../data/{dataset}_unique_id_data/', exist_ok=True)

    for id_ in data.unique_id.unique():
        pat_data = data[data.unique_id == id_]
        pat_data = pat_data[['unique_id', 'ds', 'y', 'CHO', 'basal_insulin', 'bolus_insulin']]
        pat_data.to_csv(f'../data/{dataset}_unique_id_data/{dataset}_{id_}_data.csv', index=False)

    data = pd.read_csv(f'./data/{dataset}_static.csv')
    for id_ in data.unique_id.unique():
        pat_data = data[data.unique_id == id_]
        pat_data = pat_data[['unique_id', 'Age', 'BW', 'adolescent', 'adult']]
        pat_data.to_csv(f'../data/{dataset}_unique_id_data/{dataset}_{id_}_static.csv', index=False)


get_unique_id_datasets('simglucose')
get_unique_id_datasets('ohiot1dm')