import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed

# SMEX02: Automated Weather Observing System (AWOS) Iowa 1-min Data
# https://data.eol.ucar.edu/dataset/80.003

# IHOP_2002: Automated Weather Observing System (AWOS) Iowa 1-min Data
# https://data.eol.ucar.edu/dataset/77.099

location_dict = {'AXA':{'city':'ALGONA',
                      'latitude':43.0779106,
                      'longitude':-94.2719928,
                      'elev':371.6,
                      },
                 'IKV': {'city':'ANKENY',
                      'latitude':41.6912856,
                      'longitude':-93.5663033,
                      'elev':277.4,
                      },
                 'AIO': {'city':'ATLANTIC',
                      'latitude':41.4072672,
                      'longitude':-95.0469064,
                      'elev':360.3,
                      },
                 'ADU': {'city':'AUDUBON',
                      'latitude':41.7013756,
                      'longitude':-94.9205417,
                      'elev':392.3,
                      },
                 'BNW': {'city':'BOONE',
                      'latitude':42.0495694,
                      'longitude':-93.8475722,
                      'elev':353.6,
                      },
                 'CIN': {'city':'Carroll',
                      'latitude':42.0461944,
                      'longitude':-94.7890000,
                      'elev':367.0,
                      },
                 'CNC': {'city':'CHARITON',
                      'latitude':41.0196239,
                      'longitude':-93.3596803,
                      'elev':320.0,
                      },
                 'CCY': {'city':'CHARLES CITY',
                      'latitude':43.0726086,
                      'longitude':-92.6107783,
                      'elev':342.9,
                      },
                 'ICL': {'city':'CLARINDA',
                      'latitude':40.7217836,
                      'longitude':-95.0264267,
                      'elev':302,
                      },
                 'CAV': {'city':'CLARION',
                      'latitude':42.7419439,
                      'longitude':-93.7589094,
                      'elev':354.2,
                      },
                 'CBF': {'city':'COUNCIL BLUFFS',
                      'latitude':41.2594722,
                      'longitude':-95.7599722,
                      'elev':381.9,
                      },
                 'CSQ': {'city':'CRESTON',
                      'latitude':41.0214614,
                      'longitude':-94.3633192,
                      'elev':394.4,
                      },
                 'DEH': {'city':'DECORAH',
                      'latitude':43.2755014,
                      'longitude':-91.7393739,
                      'elev':352.7,
                      },
                 'DNS': {'city':'DENISON',
                      'latitude':41.9864325,
                      'longitude':-95.3807208,
                      'elev':388.9,
                      },
                 'FFL': {'city':'FAIRFIELD',
                      'latitude':41.0533242,
                      'longitude':-91.9789233,
                      'elev':243.5,
                      },
                 'FSW': {'city':'FORT MADISON',
                      'latitude':40.6592625,
                      'longitude':-91.3268175,
                      'elev':220.7,
                      },
                 'HNR': {'city':'HARLAN',
                      'latitude':41.5843889,
                      'longitude':-95.3396389,
                      'elev':375.2,
                      },
                 'EOK': {'city':'KEOKUK',
                      'latitude':40.4599078,
                      'longitude':-91.4285011,
                      'elev':204.5,
                      },
                 'OXV': {'city':'KNOXVILLE',
                      'latitude':41.2988856,
                      'longitude':-93.1138156,
                      'elev':282.9,
                      },
                 'LRJ': {'city':'LeMARS',
                      'latitude':42.7780178,
                      'longitude':-96.1936894,
                      'elev':364.5,
                      },
                 'MXO': {'city':'MONTICELLO',
                      'latitude':42.2240453,
                      'longitude':-91.1658219,
                      'elev':253.9,
                      },
                 'MPZ': {'city':'Mount Pleasant',
                      'latitude':40.9466139,
                      'longitude':-91.5110750,
                      'elev':223.7,
                      },
                 'MUT': {'city':'MUSCATINE',
                      'latitude':41.3678633,
                      'longitude':-91.1482164,
                      'elev':167.0,
                      },
                 'TNU': {'city':'NEWTON',
                      'latitude':41.6744297,
                      'longitude':-93.0217292,
                      'elev':290,
                      },
                 'OLZ': {'city':'OELWEIN',
                      'latitude':42.6808447,
                      'longitude':-91.9744783,
                      'elev':328.0,
                      },
                 'ORC': {'city':'ORANGE CITY',
                      'latitude':42.9902644,
                      'longitude':-96.0627967,
                      'elev':431,
                      },
                 'PEA': {'city':'Pella',
                      'latitude':41.4000667,
                      'longitude':-92.9458833,
                      'elev':269.7,
                      },
                 'RDK': {'city':'RED OAK',
                      'latitude':41.0105278,
                      'longitude':-95.2598611,
                      'elev':318.2,
                      },
                 'SHL': {'city':'SHELDON',
                      'latitude':43.2083936,
                      'longitude':-95.8334331,
                      'elev':432.5,
                      },
                 'SDA': {'city':'SHENANDOAH',
                      'latitude':40.7514817,
                      'longitude':-95.4134722,
                      'elev':296.0,
                      },
                 'SLB': {'city':'STORM LAKE',
                      'latitude':42.5971944,
                      'longitude':-95.2406667,
                      'elev':453.5,
                      },
                 'VTI': {'city':'Vinton',
                      'latitude':42.2186261,
                      'longitude':-92.0259281,
                      'elev':258,
                      },
                 'AWG': {'city':'WASHINGTON',
                      'latitude':41.2761008,
                      'longitude':-91.6734439,
                      'elev':229.8,
                      },
                 'EBS': {'city':'WEBSTER CITY',
                      'latitude':42.4366389,
                      'longitude':-93.8688611,
                      'elev':341.7,
                      }
                }

def make_id_df(data_dir, id_):
    file_list = os.listdir(data_dir)
    id_df = pd.DataFrame()
    id_files = [i for i in file_list if id_ in i]
    print(id_)

    for f in np.sort(np.array(id_files)):
        with open(f'{data_dir}/{f}', 'r') as file:
            lines = file.readlines()
            data = [line.split() for line in lines]
            df = pd.DataFrame(data)
            df = df.iloc[1:]
            
            date = f.split('_')[-1].split('.')[0]
            start_date = pd.to_datetime(date, format='%Y%m%d')
            end_date = pd.to_datetime(date, format='%Y%m%d')+pd.Timedelta('24H')-pd.Timedelta('1min')
            times = pd.DatetimeIndex(pd.date_range(start=start_date, 
                                                   end=end_date, 
                                                   freq="1Min"
                                                  )
                                    )
            
            if df.shape[0]==0:
                df_resamp = pd.DataFrame(np.full((times.shape[0], 8), np.nan),
                                  columns=['tmpf', 
                                           'dwpf', 
                                           'y', 
                                           'drct', 
                                           'gust', 
                                           'vsby', 
                                           'p01i',
                                           'alti'
                                          ]
                                         )
                df_resamp['ds'] = times
                df_resamp['unique_id'] = id_
                df_resamp['available_mask'] = 0
                df_resamp = df_resamp[['unique_id', 
                                       'ds', 
                                       'y', 
                                       'available_mask', 
                                       'tmpf', 
                                       'dwpf', 
                                       'drct', 
                                       'gust', 
                                       'vsby', 
                                       'p01i', 
                                       'alti'
                                      ]
                                     ]
            
            else:
                df.columns = ['id', 
                              'day', 
                              'minute', 
                              'tmpf', 
                              'dwpf', 
                              'y', 
                              'drct', 
                              'gust', 
                              'vsby', 
                              'p01i', 
                              'alti', 
                              'cl1', 
                              'ca1', 
                              'cl2', 
                              'ca2', 
                              'cl3', 
                              'ca3'
                             ]
              
                df['ds'] = [pd.to_datetime(df['day'].values[i]+' '+df['minute'].values[i]) 
                            for i in range(df.shape[0])]
                df.drop(columns=['id', 'day', 'minute', 'cl1', 'ca1', 'cl2', 'ca2', 'cl3', 'ca3'], 
                        inplace=True)
                df.set_index('ds', inplace=True)
                df = df[~df.index.duplicated(keep='last')]
                
                common_times = np.unique(df.index.intersection(times).values)
                
                df_resamp = pd.DataFrame(np.full((times.shape[0], 8), np.nan),
                                         index=times,
                                         columns=['tmpf', 
                                                  'dwpf', 
                                                  'y', 
                                                  'drct', 
                                                  'gust', 
                                                  'vsby', 
                                                  'p01i', 
                                                  'alti'
                                                 ]
                                        )
                df_resamp.index.name = 'ds'
                df_resamp.loc[common_times] = df.loc[common_times]
                df_resamp['unique_id'] = id_
                df_resamp['available_mask'] = 0
                df_resamp.loc[common_times, 'available_mask'] = 1
                df_resamp.reset_index(inplace=True, drop=False)
                             
                df_resamp = df_resamp[['unique_id', 
                                       'ds', 
                                       'y', 
                                       'available_mask', 
                                       'tmpf',
                                       'dwpf', 
                                       'drct', 
                                       'gust', 
                                       'vsby', 
                                       'p01i', 
                                       'alti'
                                      ]
                                     ]

            id_df = pd.concat([id_df, df_resamp], axis=0) 

        id_df = id_df.ffill()
        id_df = id_df.replace(np.nan, 0)
        
        id_df['elev'] = location_dict[id_]['elev']
        id_df['lat'] = location_dict[id_]['latitude']
        id_df['lon'] = location_dict[id_]['longitude']
        
        id_df.y = id_df.y.astype('float64')
        id_df.tmpf = id_df.tmpf.astype('float64')
        id_df.dwpf = id_df.dwpf.astype('float64')
        id_df.drct = id_df.drct.astype('float64')
        id_df.gust = id_df.gust.astype('float64')
        id_df.vsby = id_df.vsby.astype('float64')
        id_df.p01i = id_df.p01i.astype('float64')
        id_df.alti = id_df.alti.astype('float64')
            
    return id_df
  
data_dir1 = '../iowa_IHOP_data'
results1 = Parallel(n_jobs=10)(delayed(make_id_df)(data_dir1, id_) for id_ in location_dict.keys())   
final_df1 = pd.concat(results1)
final_df1.to_csv('../datasets/preprocessed_iowa_ihop_dataset.csv', index=False)

data_dir2 = '../iowa_SMEX02_data'
results2 = Parallel(n_jobs=10)(delayed(make_id_df)(data_dir2, id_) for id_ in location_dict.keys())   
final_df2 = pd.concat(results2)

os.makedirs('../datasets', exist_ok=True)
final_df2.to_csv('../datasets/preprocessed_iowa_smex02_dataset.csv', index=False)

# IHOP dataset overlaps with SMEX02 dataset
final_df1 = final_df1[(final_df1.ds>='2002-05-13 00:00:00')&(final_df1.ds<'2002-06-01 00:00:00')]

final_df = pd.DataFrame()
for id_ in final_df1.unique_id.unique():
    id_df1 = final_df1[final_df1.unique_id==id_]
    id_df2 = final_df2[final_df2.unique_id==id_]
    combined_df = pd.concat([id_df1, id_df2], axis=0)
    combined_df.sort_values(by='ds', ascending=True, inplace=True)
    combined_df.reset_index(drop=True, inplace=True)
    
    final_df = pd.concat([final_df, combined_df], axis=0)
    
final_df.to_csv('../datasets/preprocessed_iowa_ihop_smex02_dataset.csv', index=False)
