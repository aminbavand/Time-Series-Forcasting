import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import SimpleRNN
import matplotlib.pyplot as plt
import keras.backend as K
from sklearn import preprocessing
from functions import load_data,data_construction,NMSE,RMSE,NDEI

#parameters found by exploring parameters for delay embedding problem using TISEAN:
m = 8
d= 2
#selecting the dataset and algorithm: (alg=1 =>LSTM), (alg=2 => FeedForward),(dataset=1 =>Mackey-Glass), (dataset=2 => Santa Fe Laser)
alg = 2
dataset = 2

#data construction and preprocessing:
data = load_data(dataset)
# data = preprocessing.scale(data)
train_part = data[0:int(np.floor(data.shape[0]*2/3))]
test_part = data[int(np.floor(data.shape[0]*2/3)):]
test_data,test_target = data_construction(m,d,test_part)

# k fold cross validation: validation set setup
fold = 5
valid_len = np.zeros(fold)
valid_len[:fold-1] = round(train_part.shape[0]/fold)
valid_len[-1] = train_part.shape[0]-sum(valid_len)




NMSE_av = []
NMSE_std = []

RMSE_av = []
RMSE_std = []

NDEI_av = []
NDEI_std = []


for num_hidden in range(100):
#metrics:
    NMSE_valid = []
    RMSE_valid = []
    NDEI_valid = []
    for k in range(fold):  
        #constructing training part and validation part for each fold      
        valid_range = np.arange(sum(valid_len[:k]),sum(valid_len[:k+1]))
        valid_range = valid_range.astype(np.int64)
        
        train_part1 = train_part[:valid_range[0]]
        train_part2 = train_part[valid_range[-1]+1:]    
            
        train_data_1,train_target_1 = data_construction(m,d,train_part1)
        train_data_2,train_target_2 = data_construction(m,d,train_part2)
         
        if (len(train_target_1)!=0) & (len(train_target_2)!=0):
            train_data = np.concatenate((train_data_1,train_data_2),axis=0)
            train_target = np.concatenate((train_target_1,train_target_2),axis=0)
        if len(train_target_1)==0:
            train_data = train_data_2
            train_target = train_target_2
        if len(train_target_2)==0:
            train_data = train_data_1
            train_target = train_target_1

        valid_data,valid_target = data_construction(m,d,train_part[valid_range])
    
        if alg==2:
            train_data = train_data.reshape(train_data.shape[0],1,train_data.shape[1])
            train_target = train_target.reshape(train_data.shape[0],1)
            
            valid_data = valid_data.reshape(valid_data.shape[0],1,valid_data.shape[1])
            valid_target = valid_target.reshape(valid_target.shape[0],1)
            
        
        #constructing neural netowork
        if alg==1:#feed forward
            model = Sequential()
            model.add(Dense(num_hidden))
            model.add(Dense(num_hidden))
            model.add(Dense(1))
            model.compile(loss='mean_absolute_error', optimizer='adam',metrics=[NMSE,RMSE,NDEI])
        
        
        else:#RNN lstm
            model = Sequential()  
            model.add(LSTM(num_hidden))    
#            model.add(SimpleRNN(num_hidden))
#            model.add(Dense(num_hidden))
            model.add(Dense(1))
            model.compile(loss='mean_absolute_error', optimizer='adam',metrics=[NMSE,RMSE,NDEI])
        
        #training neural network:    
        model.fit(train_data, train_target, epochs=100, batch_size=32)
        
        #evaluate network on validation data    
        _, NMSE_valid1,RMSE_valid1,NDEI_valid1= model.evaluate(valid_data, valid_target)        
        NMSE_valid.append(NMSE_valid1)
        RMSE_valid.append(RMSE_valid1)
        NDEI_valid.append(NDEI_valid1)
    
    NMSE_av.append(np.mean(NMSE_valid))
    NMSE_std.append(np.std(NMSE_valid))
    
    RMSE_av.append(np.mean(RMSE_valid))
    RMSE_std.append(np.std(RMSE_valid))
    
    NDEI_av.append(np.mean(NDEI_valid))
    NDEI_std.append(np.std(NDEI_valid))
    
    
print('NMSE average on validation data', NMSE_av)
print('NMSE standard deviation on validation data', NMSE_std)
print('RMSE average on validation data', RMSE_av)
print('RMSE standard deviation on validation data', RMSE_std)
print('NDEI average on validation data', NDEI_av)
print('NDEI standard deviation on validation data', NDEI_std)





#final out of sample result with THE extracted parameters:
#data preparation:
train_data,train_target = data_construction(m,d,train_part)
if alg==2:
    test_data = test_data.reshape(test_data.shape[0],1,test_data.shape[1])
    test_target = test_target.reshape(test_target.shape[0],1)
    
    train_data = train_data.reshape(train_data.shape[0],1,train_data.shape[1])
    train_target = train_target.reshape(train_target.shape[0],1)
    
#Network's training:    
if alg==1:#feed forward
   num_hidden = 10
   model = Sequential()
   model.add(Dense(num_hidden,activation='tanh'))
   model.add(Dense(num_hidden,activation='tanh'))
   model.add(Dense(1))
   model.compile(loss='mean_absolute_error', optimizer='adam',metrics=[NMSE])
else:#RNN lstm

    num_hidden = 200
    model = Sequential()  
    model.add(LSTM(num_hidden))
    model.add(Dense(1))
    model.compile(loss='mean_absolute_error', optimizer='adam',metrics=[NMSE,RMSE,NDEI])
    


model.fit(train_data, train_target, epochs=1000, batch_size=32, verbose=2)




predict_train = model.predict(train_data)
predict_test = model.predict(test_data)

_, NMSE_train,RMSE_train,NDEI_train = model.evaluate(train_data, train_target)
 
_, NMSE_test,RMSE_test,NDEI_test = model.evaluate(test_data, test_target)



print('NMSE on test data=', NMSE_test)
print('RMSE on test data=', RMSE_test)
print('NDEI on test data=', NDEI_test)




plt.plot(predict_test)
plt.plot(test_target)
plt.xlabel('# of sample')
plt.ylabel('value')
if algorithm==1:
    plt.title('Mackey Glass prediction on test data')
else:
    plt.title('Santa Fe Laser prediction on test data')

plt.show()