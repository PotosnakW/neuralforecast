import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed

# PLOWS: Iowa Automated Weather Observing System (AWOS) 1-minute Data
# https://data.eol.ucar.edu/dataset/113.038

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
                      },
                 'PRO': {'city':'PERRY',
                      'latitude':41.8386,
                      'longitude':-94.10718,
                      'elev':293.5,
                      },
                 'IFA': {'city':'IOWA FALLS',
                      'latitude':42.52248,
                      'longitude':-93.25131,
                      'elev':341.7,
                      },
                 'IIB': {'city':'INDEPENDENCE',
                      'latitude':42.4686,
                      'longitude':-91.88934,
                      'elev':284.9,
                      },
                 'GGI': {'city':'GRINNELL',
                      'latitude':41.74305,
                      'longitude':-92.72241,
                      'elev':309.3,
                      },
                 'CKP': {'city':'CHEROKEE',
                      'latitude':42.74943,
                      'longitude':-95.55167,
                      'elev':364.9,
                      },
                 'OOA': {'city':'OSKALOSSA',
                      'latitude':41.29639,
                      'longitude':-92.64436,
                      'elev':256.2,
                      },
                 'TVK': {'city':'CENTERVILLE',
                      'latitude':40.73418,
                      'longitude':-92.87409,
                      'elev':307.6,
                      },
                }


def make_id_df(data_dir, location_dict, id_):
    file_list = os.listdir(data_dir)
    id_df = pd.DataFrame()
    id_files = [i for i in file_list 
                if (location_dict[id_]['city'].upper().replace(' ', '')
                in i.split('_')[0].upper()) & ('_20091101_20100310' in i)]

    print(id_)
    print(id_files)

    for f in np.sort(np.array(id_files)):
        with open(f'{data_dir}/{f}', 'r') as file:
            lines = file.readlines()
            data = [line.split(',') for line in lines]
            df = pd.DataFrame(data)
            df = df.iloc[1:, :-1]
            
            start_date = pd.to_datetime('20091101', format='%Y%m%d')
            end_date = pd.to_datetime('20100310', format='%Y%m%d')+pd.Timedelta('24h')-pd.Timedelta('1min')
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
                              'city',
                              'date', 
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
                              'ca1cod',
                              'cl2', 
                              'ca2', 
                              'ca2cod',
                              'cl3', 
                              'ca3',
                              'ca3cod'
                             ]
              
                df['ds'] = [pd.to_datetime(df['date'].values[i])
                            for i in range(df.shape[0])]
                df.drop(columns=['id', 
                                 'city',
                                 'date', 
                                 'cl1', 
                                 'ca1', 
                                 'ca1cod',
                                 'cl2', 
                                 'ca2', 
                                 'ca2cod',
                                 'cl3', 
                                 'ca3',
                                 'ca3cod',
                                ], 
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
        
        id_df['elev'] = location_dict[id_]['elev']
        id_df['lat'] = location_dict[id_]['latitude']
        id_df['lon'] = location_dict[id_]['longitude']
        
        id_df.y = id_df.y.replace('', np.nan)
        id_df.tmpf = id_df.tmpf.replace('', np.nan)
        id_df.dwpf = id_df.dwpf.replace('', np.nan)
        id_df.drct = id_df.drct.replace('', np.nan)
        id_df.gust = id_df.gust.replace('', np.nan)
        id_df.vsby = id_df.vsby.replace('', np.nan)
        id_df.p01i = id_df.p01i.replace('', np.nan)
        id_df.alti = id_df.alti.replace('', np.nan)

        id_df = id_df.ffill()
        id_df = id_df.replace(np.nan, 0)
        
        id_df.y = id_df.y.astype('float64')
        id_df.tmpf = id_df.tmpf.astype('float64')
        id_df.dwpf = id_df.dwpf.astype('float64')
        id_df.drct = id_df.drct.astype('float64')
        id_df.gust = id_df.gust.astype('float64')
        id_df.vsby = id_df.vsby.astype('float64')
        id_df.p01i = id_df.p01i.astype('float64')
        id_df.alti = id_df.alti.astype('float64')
            
    return id_df

data_dir = '../iowa_PLOWS_data'
results = Parallel(n_jobs=10)(delayed(make_id_df)(data_dir, location_dict, id_) for id_ in location_dict.keys())   
final_df = pd.concat(results)

os.makedirs('../datasets', exist_ok=True)
final_df.to_csv('../datasets/preprocessed_iowa_plows_dataset.csv', index=False)
