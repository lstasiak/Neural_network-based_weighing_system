#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importowanie
import pandas as pd
import numpy as np
import glob
#
from sklearn.preprocessing import PowerTransformer
from statsmodels.tsa.seasonal import seasonal_decompose
from arch.bootstrap import MovingBlockBootstrap as MBB

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

def format_data(list_paths, initialColumnNames = ['Time', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6'], 
                startObservation = 0, endObservation = 500, SI_conversion = True, masking = False,
                saveData = True, dataEveryRow = 1):
    
    # list with all measurment files (each folder refers to one day of measurment)
    nr_folders = len(list_paths)
    all_files = {}  # nuber of files with measurment
    iterators = {}   
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20] # list of measure weights
    id = 0
    n = 1
    all_data = pd.DataFrame()
    data_every_X_row = pd.DataFrame()
    
    # make a list of all file paths using glob
    for path in list_paths:
        all_files["data_{}".format(str(n))] = glob.glob(path)
        n += 1
    n = 1
    for i in range(len(all_files)):
        iterators["data_{}".format(str(n))] = [k+1 for k in range(len(all_files["data_{}".format(str(n))]))]
        n += 1
    
    # main loop
    for d in all_files:
        for k in all_files[d]:
            df = pd.read_excel(k, index_col=None, header=4+startObservation*100) # header = 104 for observation time = 1s
            df.columns = initialColumnNames
            df = df[:endObservation]
            #****************************************************************
            # SI conversion --> by sensor types
            #****************************************************************
            if SI_conversion:
                df[initialColumnNames[1]] = df[initialColumnNames[1]]*1999278.9 + 4817.1
                df[initialColumnNames[2]] = df[initialColumnNames[2]]*1999763.7 - 553.9
                df[initialColumnNames[3]] = (df[initialColumnNames[3]] + 0.093)*28.006
                df[initialColumnNames[4]] = df[initialColumnNames[4]] * 25 - 50
                df[initialColumnNames[5]] = (df[initialColumnNames[5]] + 0.119)*32.806
                df[initialColumnNames[6]] = (df[initialColumnNames[6]] + 0.105)*22.95
            
            #*****************************************************************
            # MASKING ZEROS
            #*****************************************************************
            if masking:
                zeros = pd.DataFrame(np.zeros([endObservation-df.shape[0], len(initialColumnNames)]), columns = initialColumnNames)
                df = df.append(zeros, ignore_index = True)
            #******************************************************************
            for iter in iterators[d]:
                for label in labels:
                    tempPath = k.split("\\")
                    if k == '{}\\p{}_{}kg.xls'.format(tempPath[0], str(iter),str(label)):
                        df['Y'] = pd.Series(np.ones(df.shape[0])*label)
                        
            df['id'] = pd.Series(np.ones(df.shape[0])*id)
            id += 1
            all_data = all_data.append(df, ignore_index = True)
    # names according to measurments     
    newColumnNames = ['Time [s]', 'X1 [Pa]', 'X2 [Pa]', 'X3 [mm]', 'X4 [deg C]', 'X5 [mm]', 'X6 [mm]', 'Y [kg]', 'id']
    all_data.columns = newColumnNames
    
    # if observation should be saved for every few rows              
    if dataEveryRow > 1:
        for i in range(all_data.shape[0]):
            if i%dataEveryRow == 0:
                data_every_X_row = data_every_X_row.append(all_data.iloc[i,:], ignore_index = True)
                
        if saveData:
            data_every_X_row.to_csv('data.csv')
            data_every_X_row.to_excel('data.xlsx')
        return data_every_X_row
    
    else:
        if saveData:
            all_data.to_csv('data.csv')
            all_data.to_excel('data.xlsx')
        return all_data

def prepare_data(file_path, normalization = True):
    
    DataFrame = pd.read_csv(file_path)
    #choose columns with features and labels
    X = DataFrame.iloc[:, 2:8].values
    Y = DataFrame.iloc[:, 8].values
    Y2 = Y.copy()
    Id = DataFrame.iloc[:, 9].values # Id counts all measurment sequences -> helps to reshape X and Y
    list_X = []
    list_y = []
    for id in range(156):
        measurement_X = X[np.argwhere(Id == id)]
        list_X.append(np.expand_dims(measurement_X, axis = 0))
        list_y.append(Y[np.argwhere(Id == id)[0]])

    X = np.vstack(list_X)
    X = np.squeeze(X)
    Y = np.vstack(list_y)
    
    if normalization:
        X = (X - np.mean(X, axis = 0))/np.std(X, axis = 0)

    return X, Y

# data augmentation
def augmentation(X, Y, noise = False, bootstrapping = True, noiseSTD = [0.1/2, 0.1/2, 0.01/2, 0.0002/2,0.01/2,0.02/2], nr_boot =1000, bootstrap_bl_size = 488, boot_freq = 100):
    
    if noise:
        Xn = X.copy()
        for i, j, k in np.ndindex(X.shape):
            Xn[i, j, k] += np.random.normal(0, 1)*noiseSTD[k] 

        X = np.vstack([X, Xn])
        Y = np.vstack([Y, Y])
        
    if bootstrapping:
        Xb = X.copy()
        pt = PowerTransformer(method='yeo-johnson', standardize=True)
        
        for i in range(Xb.shape[0]):
            pt.fit(Xb[i])
            lambda_param = pt.lambdas_
            transformed = pt.transform(Xb[i])
            result = seasonal_decompose(transformed, model='additive', freq=boot_freq)
            
            # Moving Block Bootstrap on Residuals
            bootstrapRes = MBB(bootstrap_bl_size, result.resid)
            for data in bootstrapRes.bootstrap(nr_boot):
                bs_x = data[0][0]
            
            reconSeriesYC = result.trend + result.seasonal + bs_x
            Xb[i] = pt.inverse_transform(reconSeriesYC)
        
        for i,j,k in np.ndindex(X.shape):
            if np.isnan(Xb[i,j,k]):
                Xb[i,j,k] = X[i,j,k]
        X = np.vstack([X, Xb])
        Y = np.vstack([Y, Y])

    return X, Y


# PLOTING
def plot_cv_history(model_history, cv = 1, figsize=(6,4)):
    
    if cv > 1:
        plt.title('Loss functions')
        for i in range(cv):
            plt.plot(model_history[i].history['loss'], label='Training Fold {}'.format(i))
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()
        
        plt.title('Validation Loss')
        for i in range(cv):
            plt.plot(model_history[i].history['val_loss'], label='Training Fold {}'.format(i))
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

        plt.title('Mean Absolute Error')
        for i in range(cv):
            plt.plot(model_history[i].history['mean_absolute_error'], label='Training Fold {}'.format(i))
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

        plt.title('Validation Mean Absolute Error')
        for i in range(cv):
            plt.plot(model_history[i].history['val_mean_absolute_error'], label='Training Fold {}'.format(i))
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()
        
    else:
        plt.figure(figsize=figsize)
        plt.title('Loss functions')
        plt.plot(model_history.history['loss'])
        plt.plot(model_history.history['val_loss'], '--')
        plt.xlabel('Epoch')
        plt.legend(['train', 'val'], loc='upper right')
        plt.show()
        
        plt.figure(figsize=figsize)
        plt.title('Mean Absolute Error')
        plt.plot(model_history.history['mean_absolute_error'])
        plt.plot(model_history.history['val_mean_absolute_error'], '--')
        plt.xlabel('Epoch')
        plt.legend(['train', 'val'], loc='upper right')
        plt.show()
        

