import pandas as pd
import numpy as np

from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario_gen_test import RandomScenario
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj, sim, batch_sim
from datetime import timedelta
from datetime import datetime


os.makedirs('../data', exist_ok=True)

ids = ['adult#001']
adolescent_ids = [f'adolescent#00{i}' for i in range(1, 10)]+['adolescent#010']
adult_ids = [f'adult#00{i}' for i in range(1, 10)]+['adult#010']
child_ids = [f'child#00{i}' for i in range(1, 10)]+['child#010']
ids = adolescent_ids+adult_ids+child_ids

glucose_monitor = 'GuardianRT'
duration_days = 54

np.random.seed(0)
random_seeds = np.random.randint(100, size=len(ids))

datasets = []
for id_, seed in zip(ids, random_seeds):
    print(id_)
    # Patient vitals do not flatline at these seeds
    if id_ == 'child#004': 
        seed = 0
    if id_ == 'child#008':
        seed = 5

    # specify start_time as the beginning of today
    now = datetime.now()
    start_time = datetime.combine(now.date(), datetime.min.time())

    # --------- Create Random Scenario --------------
    path = None
    
    # Create a simulation environment
    patient = T1DPatient.withName(id_)
    sensor = CGMSensor.withName(glucose_monitor, seed=seed)
    pump = InsulinPump.withName('Insulet')
    scenario = RandomScenario(start_time=start_time, patient=patient, seed=seed)
    env = T1DSimEnv(patient, sensor, pump, scenario)

    # Create a controller
    controller = BBController()

    # Put them together to create a simulation object
    s = SimObj(env, controller, timedelta(days=duration_days), animate=False, path=path)
    results = sim(s)
    results['unique_id'] = id_
    s.reset()
    
    # Additional Preprocessing
    df = pd.DataFrame(results.reset_index(), columns=['Time', 'unique_id', 'CGM', 'CHO', 'insulin'])
    df.columns = ['ds', 'unique_id', 'y', 'CHO', 'insulin']
    df = df.iloc[:-1, :]
    
    non_zero_insulin = results.insulin.value_counts()[results.insulin.value_counts().index!=0]
    basal_dose = non_zero_insulin.idxmax()
    insulin_idx = np.where(df.insulin!=0)[0]
    basal_insulin = np.zeros(df.shape[0])
    basal_insulin[insulin_idx] = basal_dose
    df['basal_insulin'] = basal_insulin
    df['bolus_insulin'] = df.insulin - basal_insulin
    df.drop(columns=['insulin'], inplace=True)
    
    datasets.append(df)

final_df = pd.concat(datasets).reset_index(drop=True)
final_df.to_csv(f'../data/simglucose_exog_9_day_test.csv', index=False)


Static Dataset
patient_params = pd.read_csv('../simglucose_data/simglucose/params/vpatient_params.csv')
patient_params.set_index('Name', inplace=True)
patient_features = patient_params[['Age', 'BW']]
patient_features.reset_index(drop=True, inplace=True)

za = np.diag((np.repeat(1, len(patient_params.index.unique()))))
df = pd.DataFrame(za, index=patient_params.index.unique())
df.columns = [df.index.values]
df.index.name = 'unique_id'
df = df.iloc[:, :-1]
df.reset_index(drop=False, inplace=True)
df = pd.concat([df, patient_features], axis=1)
df.rename(columns=''.join, inplace=True)

pat_type = np.zeros((len(df), 2))
adolescent_idxs = [fi for fi,i in enumerate(df.unique_id) if 'adolescent' in i]
adult_idxs = [fi for fi,i in enumerate(df.unique_id) if 'adult' in i]

pat_type[adolescent_idxs, 0] = 1
pat_type[adult_idxs, 1] = 1
pat_type_df = pd.DataFrame(pat_type, columns=['adolescent', 'adult'])

df = pd.concat([df, pat_type_df], axis=1)

df.to_csv('../data/simglucose_static.csv', index=False)
