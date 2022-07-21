
'''
This script processes event-based dataset and generates instance-based dataset for classification algorithms.
It also incorporates polar field features.
Should be run after generate_event_dataset.py

Input:  hmi_valid_events.txt
        mdi_valid_events.txt
        
Output: filt_plr_fused_data_b_mx_cyc.txt
        filt_plr_sharp_data_b_mx_cyc.txt
        filt_plr_smarp_data_b_mx_cyc.txt
        nofilt_plr_fused_data_b_mx_cyc.txt
        nofilt_plr_sharp_data_b_mx_cyc.txt
        nofilt_plr_smarp_data_b_mx_cyc.txt

'''


import numpy as np
import pickle
from dataset_generate import dataset_generate
import random
import pandas as pd
from datetime import timedelta
import math
from collections import defaultdict
from utils_polar import split_data
import os
from os.path import dirname
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()



path_cache = os.path.join(os.getcwd(), 'Cache\\')

def ma(dataset, window_width, feature):
    data = dataset[feature].astype(np.float32).values
    data = np.cumsum(np.insert(data, 0, 0)) 
    data = (data[window_width:] - data[:-window_width]) / window_width
    data = data[window_width+1:]
    t = dataset['T_REC'][window_width:-window_width]
    return data, t

def df_polar_process(df_polar, filt, window_width):
    df_polar = df_polar.drop([0,1], axis = 0)
    df_polar = df_polar[df_polar['CAPS2'].notna() & df_polar['CAPN2'].notna()]
    df_polar['T_REC'] = df_polar['T_REC'].str[:-4]
    df_polar['T_REC'] = df_polar['T_REC'].str.replace('_', ' ')
    df_polar['T_REC'] = pd.to_datetime(df_polar['T_REC'])
    #df_polar = df_polar.set_index('T_REC')
    df_polar['CAPS2'] = df_polar['CAPS2'].astype(np.float32)
    df_polar['CAPN2'] = df_polar['CAPN2'].astype(np.float32)
    if filt:
        df_polar['CAPS2'][window_width:-window_width] = ma(df_polar, window_width, 'CAPS2')[0]
        df_polar['CAPN2'][window_width:-window_width] = ma(df_polar, window_width, 'CAPN2')[0]
    return df_polar

def prep_features(pooled_events, dataset, df_polar_hmi, df_polar_mdi):
    x = []
    y = []
    for i in range(len(pooled_events)):
        if pooled_events[i][0][0] == 'T':
            feats = dataset.read_features(pooled_events[i], dataset.dataset_mdi)
            feats_polar = df_polar_mdi.loc[[(abs(df_polar_mdi['T_REC']- (dataset.pooled_events[i][2] - timedelta(hours=24) ) )).idxmin()]][['CAPN2', 'CAPS2']].values.astype(np.float64)
        else:
            feats = dataset.read_features(pooled_events[i], dataset.dataset_hmi)
            feats_polar = df_polar_hmi.loc[[(abs(df_polar_hmi['T_REC']- (dataset.pooled_events[i][2] - timedelta(hours=24) ) )).idxmin()]][['CAPN2', 'CAPS2']].values.astype(np.float64)
        #feats = np.concatenate((feats, feats_polar[0]))
        feats = np.concatenate((feats, np.tile(feats_polar, (feats.shape[0],1))), axis = 1)
        x.append(feats)
        y.append(pooled_events[i][1])
    x = np.array(x)
    y = np.array([0 if (y[i][:1] == 'B') or (y[i][:1] == 'C') else 1 for i in range(len(y))])
    return x,y

def gen_data_for_conds(): 
    # Overlapping region is not used

    for sets in ['sharp', 'smarp', 'fused']:
        for str_flt in ['filt_plr_', 'nofilt_plr_']:
            
            filt_polar = False
            if str_flt == 'filt_plr_':
                filt_polar = True
            
            path_polar = os.path.join(dirname(os.getcwd()), 'Dataset_solar\\POLAR\\')
            
            
            df_polar_hmi = pd.read_csv(path_polar + 'polar_field_hmi.txt', delimiter = "\t")
            df_polar_hmi = df_polar_process(df_polar_hmi, filt_polar, 20)
            df_polar_mdi = pd.read_csv(path_polar + 'polar_field_mdi.txt', delimiter = "\t")
            df_polar_mdi = df_polar_process(df_polar_mdi, filt_polar, 50)
    
            
            dataset = dataset_generate()
            dataset.use_cached_harps = True
            dataset.load_datasets()
            dataset.goes = dataset.goes_read_process()

            
            dataset.dataset_mdi.features = ['USFLUXL', 'MEANGBL', 'R_VALUE', 'AREA']
            dataset.dataset_hmi.features = ['USFLUXL', 'MEANGBL', 'R_VALUE', 'AREA', 'TOTUSJH', 'TOTUSJZ', 'SAVNCPP', 'ABSNJZH', 'TOTPOT', 'SIZE_ACR', 'NACR', 'MEANPOT', 'SIZE', 'MEANJZH', 'SHRGT45', 'MEANSHR', 'MEANJZD', 'MEANALP', 'MEANGBT', 'MEANGAM', 'MEANGBH', 'NPIX']
            if sets == 'sharp':
                dataset.pooled_events = dataset.dataset_hmi.valid_events
            elif sets == 'smarp':
                dataset.pooled_events = dataset.dataset_mdi.valid_events
            elif sets == 'fused':
                dataset.dataset_hmi.features = ['USFLUXL', 'MEANGBL', 'R_VALUE', 'AREA']
                dataset.pooled_events = dataset.dataset_hmi.valid_events + dataset.dataset_mdi.valid_events
            
            x, y = prep_features(dataset.pooled_events, dataset, df_polar_hmi, df_polar_mdi)
            with open(path_cache + str_flt + sets + "_data_b_mx_cyc.txt", "wb") as fp:
                pickle.dump([x,y,dataset.pooled_events], fp)





if __name__ == "__main__":
    
    gen_data_for_conds()
    