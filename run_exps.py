import numpy as np
import matplotlib.pyplot as plt
import pickle
from utils_polar import split_data, performance, hss2_score
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from sklearn.linear_model import LogisticRegression
from datetime import datetime
from os.path import dirname
import sys
import os
sys.path.insert(1, os.path.join(dirname(os.getcwd()), 'Library'))
from moe_std import moe_std
path_cache = os.path.join(os.getcwd(), 'Cache/')
path_output = os.path.join(os.getcwd(), 'Output/')
from nn import nn
import seaborn as sns
sns.set_theme(style="whitegrid")
import pandas as pd
import itertools
import time
from sklearn.model_selection import cross_validate, KFold
import time

from multiprocessing import Pool, cpu_count
from sklearn.metrics import make_scorer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Don't use GPU since CPU is faster and parallelizable for this setting


def func(comb):
    
    exps = 10
    metric = hss2_score
    cv = True
    
    
    polar_data = comb[0]
    polar_data_mod = comb[1]
    split = comb[2]
    data = comb[3]
    model_in = comb[4]
    
    score = make_scorer(metric)
    
    res_vec = []
    
    # Data load
    fn = path_cache + polar_data + '_plr_' + data
    with open(fn, "rb") as fp:
        x, y, pooled_events = pickle.load(fp)
        
    # Polar field features
    if polar_data_mod == 'cyc':
        x = np.array([np.concatenate((x[i,:,:-2], abs(x[i,:,-1] - x[i,:,-2])[None].T), axis = 1) for i in range(len(x))])
        n_field = 1
    elif polar_data_mod == 'raw':
        n_field = 2
    
    
    for e in range(exps):

        x_tr, y_tr, x_te, y_te, _, _ = split_data(x, y, pooled_events, False, split, data[0:5])
        
        
        if model_in == 'aug':
            x_in_tr = x_tr
            x_in_te = x_te
        else:
            x_in_tr = x_tr[:,:,:-n_field]
            x_in_te = x_te[:,:,:-n_field]
            
        # Logistic Regression
        if True:
            s_kv = [10]
            best_ind = 0
            if cv:
                s_kv = 10**np.arange(-1, 2, dtype=float)    
                cv_results = []
                for s_k in s_kv:
                    cv_result_in = []
                    for k in range(5):
                        ind = np.arange(len(x_in_tr))
                        ind_val = ind[k * int(len(x_in_tr) / 5) : (k+1) * int(len(x_in_tr) / 5)]
                        ind_tr = np.setdiff1d(ind, ind_val)
                        model = LogisticRegression(C = s_k)
                        yp_val = model.fit(x_in_tr[ind_tr,-1,:], y_tr[ind_tr,0]).predict(x_in_tr[ind_val,-1,:])
                        cv_result_in.append(metric(y_tr[ind_val,0], yp_val))
                    cv_results.append(np.mean(cv_result_in))
                best_ind = np.argmax(cv_results)
                    
            model = LogisticRegression(C = s_kv[best_ind])
            model.fit(x_in_tr[:,-1,:], y_tr[:,0])
            yp_te = model.predict(x_in_te[:,-1,:])
            res_vec.append([polar_data, polar_data_mod, split, data[0:5], 'lr-' + model_in, metric(y_te, yp_te)])
            
            
        # MLP
        if True:
            s_kv = [1]
            best_ind = 0
            if True:
                s_kv = 10**np.arange(-1, 2, dtype=float)  
    
                cv_results = []
                for s_k in s_kv:
                    cv_result_in = []
                    for k in range(5):
                        ind = np.arange(len(x_in_tr))
                        ind_val = ind[k * int(len(x_in_tr) / 5) : (k+1) * int(len(x_in_tr) / 5)]
                        ind_tr = np.setdiff1d(ind, ind_val)
                        model = nn( D = x_in_tr.shape[-1], M = 1, sk = s_k, obs = 'cat', cnv_th=1e-5, batch_size=0, expert_type = 'nn')
                        model.fit(x_in_tr[ind_tr,-1,:].astype(np.float32), y_tr[ind_tr,:].astype(np.float32))
                        yp_val = model.predict(x_in_tr[ind_val,-1,:].astype(np.float32))
                        cv_result_in.append(metric(y_tr[ind_val,:], yp_val))
                    cv_results.append(np.mean(cv_result_in))
                best_ind = np.argmax(cv_results)
            
            model = nn( D = x_in_tr.shape[-1], M = 1, sk = s_kv[best_ind], obs = 'cat', cnv_th=1e-5, batch_size=0, expert_type = 'nn')
            model.fit(x_in_tr[:,-1,:].astype(np.float32), y_tr.astype(np.float32))
            yp_te = model.predict(x_in_te[:,-1,:].astype(np.float32))
            res_vec.append([polar_data, polar_data_mod, split, data[0:5], 'nn-' + model_in, metric(y_te, yp_te)])
            
            
        # RNN
        if True:
            s_kv = [100]
            best_ind = 0
            if True:
                s_kv = 10**np.arange(-0, 3, dtype=float)  
    
                cv_results = []
                for s_k in s_kv:
                    cv_result_in = []
                    for k in range(5):
                        ind = np.arange(len(x_in_tr))
                        ind_val = ind[k * int(len(x_in_tr) / 5) : (k+1) * int(len(x_in_tr) / 5)]
                        ind_tr = np.setdiff1d(ind, ind_val)
                        model = nn( D = x_in_tr.shape[-1], M = 1, sk = s_k, obs = 'cat', cnv_th=1e-5, batch_size=0, expert_type = 'rn')
                        model.fit(x_in_tr[ind_tr].astype(np.float32), y_tr[ind_tr,:].astype(np.float32))
                        yp_val = model.predict(x_in_tr[ind_val].astype(np.float32))
                        cv_result_in.append(metric(y_tr[ind_val,:], yp_val))
                    cv_results.append(np.mean(cv_result_in))
                best_ind = np.argmax(cv_results)
            
            model = nn( D = x_in_tr.shape[-1], M = 1, sk = s_kv[best_ind], obs = 'cat', cnv_th=1e-5, batch_size=0, expert_type = 'rn')
            q_vec = model.fit(x_in_tr.astype(np.float32), y_tr.astype(np.float32))
            plt.plot(q_vec)
            yp_te = model.predict(x_in_te.astype(np.float32))
            res_vec.append([polar_data, polar_data_mod, split, data[0:5], 'rn-' + model_in, metric(y_te, yp_te)])
                
              
        # Mixture Model
        if True:

            s_kv = [10]
            s_sv = [10]
            best_ind = 0
            if True:
                s_kv = 10**np.arange(-1, 2, dtype=float)  
                s_sv = 10**np.arange(-2, 3, dtype=float)  
            
                cv_results = []
                #for s_k, s_s in zip(s_kv, s_sv):
                for s_k in s_kv:
                    cv_result_in = []
                    for k in range(5):
                        ind = np.arange(len(x_in_tr))
                        ind_val = ind[k * int(len(x_in_tr) / 5) : (k+1) * int(len(x_in_tr) / 5)]
                        ind_tr = np.setdiff1d(ind, ind_val)
                        model = moe_std(K = 3,
                                    Dx = x_in_tr.shape[-1],
                                    M = y_tr.shape[1],
                                    sk = s_k, ss = s_k,
                                    obs = 'cat')
                        model.fit(x_in_tr[ind_tr,-1,:].astype(np.float32), y_tr[ind_tr,:].astype(np.float32))
                        yp_val = model.predict(x_in_tr[ind_val,-1,:].astype(np.float32))
                        cv_result_in.append(metric(y_tr[ind_val,:], yp_val))
                    cv_results.append(np.mean(cv_result_in))
                best_ind = np.argmax(cv_results)
                    
            model = moe_std(K = 3,
                        Dx = x_in_tr.shape[-1],
                        M = y_tr.shape[1],
                        sk = s_kv[best_ind], ss = s_kv[best_ind],
                        obs = 'cat')
            model.fit(x_in_tr[:,-1,:].astype(np.float32), y_tr.astype(np.float32))
            yp_te = model.predict(x_in_te[:,-1,:].astype(np.float32))
            res_vec.append([polar_data, polar_data_mod, split, data[0:5], 'mx-' + model_in, metric(y_te, yp_te)])
        
    return res_vec
    

if __name__ == '__main__':
    
    
    polar_data_list = ['nofilt','filt']
    polar_data_mods = ['cyc', 'raw']
    data_list = [
        "sharp_data_b_mx_cyc.txt",
        #"smarp_data_b_mx_cyc.txt",
        #"fused_data_b_mx_cyc.txt",
        ]
    splits = ['random', 'year', 'ar']

    start = time.time()
    
    # Train all
    if True:
        models = ['bar', 'aug']
        combs = []
        for r in itertools.product(polar_data_list, polar_data_mods, splits, data_list, models):
            combs.append(r)
        p = Pool(cpu_count()-4)
        #res = func(combs[0])
        res_vec = p.map(func, combs)
        res_vec2 = [item for sublist in res_vec for item in sublist]
        with open(path_output + 'res_vec_all.txt', "wb") as fp:
            pickle.dump(res_vec2, fp)
            
    
    print(time.time() - start)
        
    