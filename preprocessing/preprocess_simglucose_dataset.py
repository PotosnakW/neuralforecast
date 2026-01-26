import pandas as pd
import numpy as np
import os

from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.sim_engine import SimObj, sim, batch_sim
from datetime import timedelta
from datetime import datetime


ids = ['adult#001']
adolescent_ids = [f'adolescent#00{i}' for i in range(1, 10)]+['adolescent#010']
adult_ids = [f'adult#00{i}' for i in range(1, 10)]+['adult#010']
child_ids = [f'child#00{i}' for i in range(1, 10)]+['child#010']
ids = adolescent_ids+adult_ids+child_ids

glucose_monitor = 'GuardianRT'
duration_days = 90 #182

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
    
    datasets.append(df)

final_df = pd.concat(datasets).reset_index(drop=True)

os.makedirs('../datasets', exist_ok=True)
final_df.to_csv(f'../datasets/simglucose_{duration_days}_days.csv', index=False)
