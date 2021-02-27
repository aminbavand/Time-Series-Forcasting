import numpy as np
import pandas as pd
import keras.backend as K

def load_data(dataset):
    
    if dataset==1:
        data = np.array(pd.read_csv("./Data/mgdata.csv"))
    else:
        data = np.genfromtxt('./Data/LORENZ.DAT', dtype=None, delimiter=',')

    return data

def data_construction(m,d,data):
    if len(data)==0:
        train_data = []
        train_target = []
    else:
        diff = np.arange(0,(m+1)*d,d)
        
        b = np.arange(len(data)-m*d)
        c1 = np.kron(np.ones((m+1,1)),b)
        c2 = np.kron(np.ones((len(data)-m*d,1)),diff).T
        c3 = (c1 + c2).flatten('F')
        c4 = data.take(list(c3))
        data_new = c4.reshape(int(len(c4)/(m+1)),m+1)
        
        train_data = data_new[:,0:m]
        train_target = data_new[:,m]
    
    return train_data,train_target

def NMSE(meas,pred):
    var = K.mean((meas-K.mean(meas))**2)
    NMSE = (K.mean((pred-meas)**2))/(var)
    
    return NMSE

def RMSE(meas,pred):
    RMSE = (K.mean((pred-meas)**2))**(0.5)

    return RMSE

def NDEI(meas,pred):
    var = K.mean((meas-K.mean(meas))**2)
    NDEI = ((K.mean((pred-meas)**2))**(0.5))/(var)

    return NDEI