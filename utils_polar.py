
import numpy as np
import os
from os.path import dirname
import pickle
from sklearn.metrics import confusion_matrix

def split_data(x, y, pooled_events, regression = False, split = 'ar', data = 'fused'):
    
    if split == 'year':
        

        min_year = min([p[-1] for p in pooled_events]).year
        max_year = max([p[-1] for p in pooled_events]).year
        
        ind = np.arange(min_year, max_year + 1)
        np.random.shuffle(ind)
        ind_tr_y = ind[0: int(0.8*len(ind))]
        ind_te_y = ind[int(0.8*len(ind)):]
        
        ind_tr = []
        ind_te = []
        for i in range(len(pooled_events)):
            t = pooled_events[i][2]
            if t.year in ind_tr_y:
                ind_tr.append(i)
            else:
                ind_te.append(i)
            
    elif split == 'random':
        
        N = len(pooled_events)
        ind = np.arange(len(pooled_events))
        np.random.shuffle(ind)
        ind_tr = ind[0: int(0.8*N)]
        ind_te = ind[int(0.8*N):]
        
    elif split == 'ar':
        ars = [int(pooled_events[i][0][4:10]) for i in range(0,len(pooled_events))]
        ars = np.unique(ars)
        np.random.shuffle(ars)
        
        N = len(ars)
        ars_tr = ars[0:int(0.8*N)]
        ars_te = ars[int(0.8*N):]
        
        ind_tr = [i for i in range(len(pooled_events)) if int(pooled_events[i][0][4:10]) in ars_tr]
        ind_te = [i for i in range(len(pooled_events)) if int(pooled_events[i][0][4:10]) in ars_te]
        
        
            
    if regression:
        y = np.zeros(len(y))
        letter = np.array([pooled_events[i][1][0] for i in range(len(pooled_events))])
        intensity = np.array([pooled_events[i][1][1:] for i in range(len(pooled_events))]).astype(float)
        
        y[np.where(letter == 'B')[0]] = 1e-7
        y[np.where(letter == 'C')[0]] = 1e-6
        y[np.where(letter == 'M')[0]] = 1e-5
        y[np.where(letter == 'X')[0]] = 1e-4
        y = y * intensity
        y = np.log(y)
    
    
    x_tr   = x[ind_tr]
    x_te = x[ind_te]
    
    events_tr = [pooled_events[i] for i in ind_tr]
    events_te = [pooled_events[i] for i in ind_te]
    
    y_tr   = y[ind_tr][None].T
    y_te = y[ind_te][None].T
    
    # Normalization (dataset-wise)
    ind_tr_sharp = [i for i in range(len(events_tr)) if (events_tr[i][0][0] == 'H')]
    ind_te_sharp = [i for i in range(len(events_te)) if (events_te[i][0][0] == 'H')]
    
    ind_tr_smarp = [i for i in range(len(events_tr)) if (events_tr[i][0][0] == 'T')]
    ind_te_smarp = [i for i in range(len(events_te)) if (events_te[i][0][0] == 'T')]
    
    # Normalize Sharp
    if data == 'sharp' or data == 'fused': 
        mean_sharp = np.mean(x_tr[ind_tr_sharp,:, :-2], axis = (0,1))
        std_sharp = np.std(x_tr[ind_tr_sharp, :, :-2], axis = (0,1))
        x_tr[ind_tr_sharp,:, :-2] = (x_tr[ind_tr_sharp,:, :-2] - mean_sharp) / std_sharp
        x_te[ind_te_sharp,:, :-2] = (x_te[ind_te_sharp,:, :-2] - mean_sharp) / std_sharp
    
    # Normalize Smarp
    if data == 'smarp' or data == 'fused': 
        mean_smarp = np.mean(x_tr[ind_tr_smarp, :, :-2], axis = (0,1))
        std_smarp = np.std(x_tr[ind_tr_smarp, :, :-2], axis = (0,1))
        
        x_tr[ind_tr_smarp,:, :-2] = (x_tr[ind_tr_smarp,:, :-2] - mean_smarp) / std_smarp
        x_te[ind_te_smarp,:, :-2] = (x_te[ind_te_smarp,:, :-2] - mean_smarp) / std_smarp
    
    # Normalize Polar field
    mean_polar = np.mean(x_tr[:,:,-2:], axis = (0,1))
    std_polar = np.std(x_tr[:,:,-2:], axis = (0,1))
    x_tr[:,:,-2:] = (x_tr[:,:,-2:] - mean_polar) / std_polar
    x_te[:,:,-2:] = (x_te[:,:,-2:] - mean_polar) / std_polar

    # Normalize output
    if regression:
        mean = np.mean(np.concatenate(y_tr))
        std = np.std(np.concatenate(y_tr))
        
        y_tr = (y_tr - mean) / std 
        y_te = (y_te - mean) / std
    
    return x_tr, y_tr, x_te, y_te, ind_tr, ind_te


def performance(y, yp, regression):
    
    if regression:
        mse = np.mean((yp - y)**2)
        return mse
    else:
        y = np.concatenate((y, 1-y.sum(axis = 1)[None].T), axis = 1)
        yp = np.concatenate((yp, 1-yp.sum(axis = 1)[None].T), axis = 1)
        
        tn, fp, fn, tp = confusion_matrix(np.argmax(y, axis = 1), np.argmax(yp, axis = 1), labels = [0,1]).ravel()
        #acc = (tp + tn) / (tn + fp + fn + tp)
        hss2 = (2 * ((tp * tn) - (fn*fp))) / ((tp + fn) * (fn + tn) + (tn + fp) * (tp + fp))
        return hss2
    
def hss2_score(y, yp):
    
    if len(yp.shape) == 1:
        yp = yp[None].T
        
    if len(y.shape) == 1:
        y = y[None].T
    
    y = np.concatenate((y, 1-y.sum(axis = 1)[None].T), axis = 1)
    yp = np.concatenate((yp, 1-yp.sum(axis = 1)[None].T), axis = 1)
    
    tn, fp, fn, tp = confusion_matrix(np.argmax(y, axis = 1), np.argmax(yp, axis = 1), labels = [0,1]).ravel()
    #acc = (tp + tn) / (tn + fp + fn + tp)
    hss2 = (2 * ((tp * tn) - (fn*fp))) / ((tp + fn) * (fn + tn) + (tn + fp) * (tp + fp))
    return hss2    



if __name__ == "__main__":
    
    
    path_cache = os.path.join(os.getcwd(), 'Cache\\')

    data = 'pooled'
    with open(path_cache + "filt_plr_" + data + "_data_b_mx_cyc.txt", "rb") as fp:
        x, y, pooled_events = pickle.load(fp)

    x_tr, y_tr, x_val, y_val, x_te, y_te,_,_,_ = split_data(x, y, pooled_events, regression=True, data = data)
