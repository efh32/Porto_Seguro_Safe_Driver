#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:42:52 2017

@author: ethan
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#import training data
#os.chdir('./data/')

raw_data = pd.read_csv('../data/train.csv')
raw_test_data = pd.read_csv('../data/test.csv')

#processed data to write out to csv
processed_train_data = raw_data.copy()
processed_test_data = raw_test_data.copy()

#check which columns have null data
processed_train_data.isnull().sum()
processed_test_data.isnull().sum()


#remove ps_car_03_cat and ps_car_05_cat because it is missing in both test and train data
columns_to_drop = ['ps_car_03_cat','ps_car_05_cat']

processed_train_data.drop(columns_to_drop, inplace=True, axis=1)
processed_train_data.isnull().sum()


processed_test_data.drop(columns_to_drop, inplace=True, axis=1)
processed_test_data.isnull().sum()


#imput with mean or mode
mean_imp = Imputer(strategy='mean', axis=0, missing_values=-1)
mode_imp = Imputer(strategy='most_frequent', axis=0, missing_values=-1)

#imput training data
processed_train_data['ps_reg_03'] = mean_imp.fit_transform(processed_train_data[['ps_reg_03']]).ravel()
processed_train_data['ps_car_12'] = mean_imp.fit_transform(processed_train_data[['ps_car_12']]).ravel()
processed_train_data['ps_car_14'] = mean_imp.fit_transform(processed_train_data[['ps_car_14']]).ravel()
processed_train_data['ps_car_11'] = mode_imp.fit_transform(processed_train_data[['ps_car_11']]).ravel()
processed_train_data['ps_ind_02_cat'] = mode_imp.fit_transform(processed_train_data[['ps_ind_02_cat']]).ravel()
processed_train_data['ps_ind_04_cat'] = mode_imp.fit_transform(processed_train_data[['ps_ind_04_cat']]).ravel()
processed_train_data['ps_ind_05_cat'] = mode_imp.fit_transform(processed_train_data[['ps_ind_05_cat']]).ravel()
processed_train_data['ps_car_01_cat'] = mode_imp.fit_transform(processed_train_data[['ps_car_01_cat']]).ravel()
processed_train_data['ps_car_02_cat'] = mode_imp.fit_transform(processed_train_data[['ps_car_02_cat']]).ravel()
processed_train_data['ps_car_07_cat'] = mode_imp.fit_transform(processed_train_data[['ps_car_07_cat']]).ravel()
processed_train_data['ps_car_09_cat'] = mode_imp.fit_transform(processed_train_data[['ps_car_09_cat']]).ravel()

#transform int data changed to float back into int
processed_train_data['ps_car_11'] = processed_train_data['ps_car_11'].astype(int)
processed_train_data['ps_ind_02_cat'] = processed_train_data['ps_ind_02_cat'].astype(int)
processed_train_data['ps_ind_04_cat'] = processed_train_data['ps_ind_04_cat'].astype(int)
processed_train_data['ps_ind_05_cat'] = processed_train_data['ps_ind_05_cat'].astype(int)
processed_train_data['ps_car_01_cat'] = processed_train_data['ps_car_01_cat'].astype(int)
processed_train_data['ps_car_02_cat'] = processed_train_data['ps_car_02_cat'].astype(int)
processed_train_data['ps_car_07_cat'] = processed_train_data['ps_car_07_cat'].astype(int)
processed_train_data['ps_car_09_cat'] = processed_train_data['ps_car_09_cat'].astype(int)



#imput testing data
processed_test_data['ps_reg_03'] = mean_imp.fit_transform(processed_test_data[['ps_reg_03']]).ravel()
processed_test_data['ps_car_12'] = mean_imp.fit_transform(processed_test_data[['ps_car_12']]).ravel()
processed_test_data['ps_car_14'] = mean_imp.fit_transform(processed_test_data[['ps_car_14']]).ravel()
processed_test_data['ps_car_11'] = mode_imp.fit_transform(processed_test_data[['ps_car_11']]).ravel()

processed_test_data['ps_ind_02_cat'] = mode_imp.fit_transform(processed_test_data[['ps_ind_02_cat']]).ravel()
processed_test_data['ps_ind_04_cat'] = mode_imp.fit_transform(processed_test_data[['ps_ind_04_cat']]).ravel()
processed_test_data['ps_ind_05_cat'] = mode_imp.fit_transform(processed_test_data[['ps_ind_05_cat']]).ravel()
processed_test_data['ps_car_01_cat'] = mode_imp.fit_transform(processed_test_data[['ps_car_01_cat']]).ravel()
processed_test_data['ps_car_02_cat'] = mode_imp.fit_transform(processed_test_data[['ps_car_02_cat']]).ravel()
processed_test_data['ps_car_07_cat'] = mode_imp.fit_transform(processed_test_data[['ps_car_07_cat']]).ravel()
processed_test_data['ps_car_09_cat'] = mode_imp.fit_transform(processed_test_data[['ps_car_09_cat']]).ravel()

#transform int data changed to float back into int
processed_test_data['ps_car_11'] = processed_test_data['ps_car_11'].astype(int)
processed_test_data['ps_ind_02_cat'] = processed_test_data['ps_ind_02_cat'].astype(int)
processed_test_data['ps_ind_04_cat'] = processed_test_data['ps_ind_04_cat'].astype(int)
processed_test_data['ps_ind_05_cat'] = processed_test_data['ps_ind_05_cat'].astype(int)
processed_test_data['ps_car_01_cat'] = processed_test_data['ps_car_01_cat'].astype(int)
processed_test_data['ps_car_02_cat'] = processed_test_data['ps_car_02_cat'].astype(int)
processed_test_data['ps_car_07_cat'] = processed_test_data['ps_car_07_cat'].astype(int)
processed_test_data['ps_car_09_cat'] = processed_test_data['ps_car_09_cat'].astype(int)


#Take a look at features distribution/freqncy for the two classes.  Drop features that have similar distributions

features = processed_train_data.ix[:,2:57].columns


#vizualize features
plt.figure(figsize=(10,28*4))
gs = gridspec.GridSpec(28,1)
for i, cn in enumerate(processed_train_data[features]):
    print(i)
    ax = plt.subplot(gs[i])
    plt.hist(processed_train_data[cn][processed_train_data.target==1], alpha = 0.6)
    plt.hist(processed_train_data[cn][processed_train_data.target==0], alpha = 0.3)
    ax.set_xlabel('')
    ax.set_title('histogram of feature : ' + str(cn))
plt.show

#ps_ind_01
claim = processed_train_data.loc[processed_train_data['target']==1]
no_claim = processed_train_data.loc[processed_train_data['target']==0]


claim['ps_ind_01'].value_counts()/len(claim)
no_claim['ps_ind_01'].value_counts()/len(no_claim) #remove

claim['ps_ind_02_cat'].value_counts()/len(claim)
no_claim['ps_ind_02_cat'].value_counts()/len(no_claim) #remove

claim['ps_ind_03'].value_counts()/len(claim)
no_claim['ps_ind_03'].value_counts()/len(no_claim) #keep

claim['ps_ind_04_cat'].value_counts()/len(claim)
no_claim['ps_ind_04_cat'].value_counts()/len(no_claim) #keep

claim['ps_ind_05_cat'].value_counts()/len(claim)
no_claim['ps_ind_05_cat'].value_counts()/len(no_claim) #keep

claim['ps_ind_06_bin'].value_counts()/len(claim)
no_claim['ps_ind_06_bin'].value_counts()/len(no_claim) #keep

claim['ps_ind_07_bin'].value_counts()/len(claim)
no_claim['ps_ind_07_bin'].value_counts()/len(no_claim) #keep

claim['ps_ind_08_bin'].value_counts()/len(claim)
no_claim['ps_ind_08_bin'].value_counts()/len(no_claim) #keep

claim['ps_ind_09_bin'].value_counts()/len(claim)
no_claim['ps_ind_09_bin'].value_counts()/len(no_claim) #keep

claim['ps_ind_10_bin'].value_counts()/len(claim)
no_claim['ps_ind_10_bin'].value_counts()/len(no_claim) #remove

claim['ps_ind_11_bin'].value_counts()/len(claim)
no_claim['ps_ind_11_bin'].value_counts()/len(no_claim) #remove

claim['ps_ind_12_bin'].value_counts()/len(claim)
no_claim['ps_ind_12_bin'].value_counts()/len(no_claim) #remove

claim['ps_ind_13_bin'].value_counts()/len(claim)
no_claim['ps_ind_13_bin'].value_counts()/len(no_claim) #remove

claim['ps_ind_14'].value_counts()/len(claim)
no_claim['ps_ind_14'].value_counts()/len(no_claim) #keep

claim['ps_ind_15'].value_counts()/len(claim)
no_claim['ps_ind_15'].value_counts()/len(no_claim) #keep

claim['ps_ind_16_bin'].value_counts()/len(claim)
no_claim['ps_ind_16_bin'].value_counts()/len(no_claim) #keep

claim['ps_ind_17_bin'].value_counts()/len(claim)
no_claim['ps_ind_17_bin'].value_counts()/len(no_claim) #keep

claim['ps_ind_18_bin'].value_counts()/len(claim)
no_claim['ps_ind_18_bin'].value_counts()/len(no_claim) #keep

#reg01 to reg03 are continuous
claim['ps_reg_01'].describe()
no_claim['ps_reg_01'].describe() #keep

claim['ps_reg_02'].describe()
no_claim['ps_reg_02'].describe() #keep

claim['ps_reg_03'].describe()
no_claim['ps_reg_03'].describe() #keep

claim['ps_car_01_cat'].value_counts()/len(claim)
no_claim['ps_car_01_cat'].value_counts()/len(no_claim) #keep

claim['ps_car_02_cat'].value_counts()/len(claim)
no_claim['ps_car_02_cat'].value_counts()/len(no_claim) #keep

claim['ps_car_04_cat'].value_counts()/len(claim)
no_claim['ps_car_04_cat'].value_counts()/len(no_claim) #keep

claim['ps_car_06_cat'].value_counts()/len(claim)
no_claim['ps_car_06_cat'].value_counts()/len(no_claim) #keep

claim['ps_car_07_cat'].value_counts()/len(claim)
no_claim['ps_car_07_cat'].value_counts()/len(no_claim) #keep

claim['ps_car_08_cat'].value_counts()/len(claim)
no_claim['ps_car_08_cat'].value_counts()/len(no_claim) #keep

claim['ps_car_09_cat'].value_counts()/len(claim)
no_claim['ps_car_09_cat'].value_counts()/len(no_claim) #keep

claim['ps_car_10_cat'].value_counts()/len(claim)
no_claim['ps_car_10_cat'].value_counts()/len(no_claim) #remove

claim['ps_car_11_cat'].value_counts()/len(claim)
no_claim['ps_car_11_cat'].value_counts()/len(no_claim) #keep

claim['ps_car_11'].value_counts()/len(claim)
no_claim['ps_car_11'].value_counts()/len(no_claim) #keep

claim['ps_car_12'].describe()
no_claim['ps_car_12'].describe() #keep

claim['ps_car_13'].describe()
no_claim['ps_car_13'].describe() #keep

claim['ps_car_14'].describe()
no_claim['ps_car_14'].describe() #remove

claim['ps_car_15'].describe()
no_claim['ps_car_15'].describe() #keep

claim['ps_calc_01'].describe()
no_claim['ps_calc_01'].describe() #remove

claim['ps_calc_02'].describe()
no_claim['ps_calc_02'].describe() #keep - different median

claim['ps_calc_03'].describe()
no_claim['ps_calc_03'].describe() #remove 

claim['ps_calc_04'].describe()
no_claim['ps_calc_04'].describe() #remove

claim['ps_calc_05'].describe()
no_claim['ps_calc_05'].describe() #remove

claim['ps_calc_06'].describe()
no_claim['ps_calc_06'].describe() #remove

claim['ps_calc_07'].describe()
no_claim['ps_calc_07'].describe() #remove

claim['ps_calc_08'].describe()
no_claim['ps_calc_08'].describe() #remove

claim['ps_calc_09'].describe()
no_claim['ps_calc_09'].describe() #remove

claim['ps_calc_10'].describe()
no_claim['ps_calc_10'].describe() #remove

claim['ps_calc_11'].describe()
no_claim['ps_calc_11'].describe() #remove

claim['ps_calc_12'].describe()
no_claim['ps_calc_12'].describe() #remove

claim['ps_calc_13'].describe()
no_claim['ps_calc_13'].describe() #remove

claim['ps_calc_14'].describe()
no_claim['ps_calc_14'].describe() #remove

claim['ps_calc_15_bin'].value_counts()/len(claim)
no_claim['ps_calc_15_bin'].value_counts()/len(no_claim) #remove

claim['ps_calc_16_bin'].value_counts()/len(claim)
no_claim['ps_calc_16_bin'].value_counts()/len(no_claim) #remove

claim['ps_calc_17_bin'].value_counts()/len(claim)
no_claim['ps_calc_17_bin'].value_counts()/len(no_claim) #remove

claim['ps_calc_18_bin'].value_counts()/len(claim)
no_claim['ps_calc_18_bin'].value_counts()/len(no_claim) #remove

claim['ps_calc_19_bin'].value_counts()/len(claim)
no_claim['ps_calc_19_bin'].value_counts()/len(no_claim) #remove

claim['ps_calc_20_bin'].value_counts()/len(claim)
no_claim['ps_calc_20_bin'].value_counts()/len(no_claim) #remove

columns_to_drop_2 = ['ps_ind_01','ps_ind_02_cat','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_car_10_cat',
                     'ps_calc_01', 'ps_calc_03', 'ps_calc_04', 'ps_calc_05','ps_calc_06', 'ps_calc_07', 'ps_calc_08',
                     'ps_calc_09', 'ps_calc_10','ps_calc_11', 'ps_calc_12','ps_calc_13', 'ps_calc_14',  'ps_calc_15_bin',
                     'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin', 'ps_car_14']

processed_train_data.drop(columns_to_drop_2, inplace=True, axis=1)

processed_test_data.drop(columns_to_drop_2, inplace=True, axis=1)

#find mean/std for continous features between test set and training set- exclude features with discrepancy
processed_train_data['ps_reg_01'].describe()
processed_test_data['ps_reg_01'].describe() #looks the same, keep

processed_train_data['ps_reg_02'].describe()
processed_test_data['ps_reg_02'].describe() #looks the same, keep

processed_train_data['ps_reg_03'].describe()
processed_test_data['ps_reg_03'].describe() #looks the same, keep

processed_train_data['ps_car_12'].describe()
processed_test_data['ps_car_12'].describe() #kind of close, keep

processed_train_data['ps_car_13'].describe()
processed_test_data['ps_car_13'].describe() #looks the same, keep

processed_train_data['ps_car_15'].describe()
processed_test_data['ps_car_15'].describe() #looks the same, keep

processed_train_data['ps_calc_02'].describe()
processed_test_data['ps_calc_02'].describe() #kind of close, keep

#find frequency for categorical features between test set and training set- exclude features with discrepancy
processed_train_data['ps_ind_03'].value_counts()/len(processed_train_data)
processed_test_data['ps_ind_03'].value_counts()/len(processed_test_data) #keep

processed_train_data['ps_ind_04_cat'].value_counts()/len(processed_train_data)
processed_test_data['ps_ind_04_cat'].value_counts()/len(processed_test_data) #keep

processed_train_data['ps_ind_05_cat'].value_counts()/len(processed_train_data)
processed_test_data['ps_ind_05_cat'].value_counts()/len(processed_test_data) #keep

processed_train_data['ps_ind_06_bin'].value_counts()/len(processed_train_data)
processed_test_data['ps_ind_06_bin'].value_counts()/len(processed_test_data) #keep

processed_train_data['ps_ind_07_bin'].value_counts()/len(processed_train_data)
processed_test_data['ps_ind_07_bin'].value_counts()/len(processed_test_data) #keep

processed_train_data['ps_ind_08_bin'].value_counts()/len(processed_train_data)
processed_test_data['ps_ind_08_bin'].value_counts()/len(processed_test_data) #keep

processed_train_data['ps_ind_09_bin'].value_counts()/len(processed_train_data)
processed_test_data['ps_ind_09_bin'].value_counts()/len(processed_test_data) #keep

processed_train_data['ps_ind_14'].value_counts()/len(processed_train_data)
processed_test_data['ps_ind_14'].value_counts()/len(processed_test_data) #keep

processed_train_data['ps_ind_15'].value_counts()/len(processed_train_data)
processed_test_data['ps_ind_15'].value_counts()/len(processed_test_data) #keep

processed_train_data['ps_ind_16_bin'].value_counts()/len(processed_train_data)
processed_test_data['ps_ind_16_bin'].value_counts()/len(processed_test_data) #keep

processed_train_data['ps_ind_17_bin'].value_counts()/len(processed_train_data)
processed_test_data['ps_ind_17_bin'].value_counts()/len(processed_test_data) #keep

processed_train_data['ps_ind_18_bin'].value_counts()/len(processed_train_data)
processed_test_data['ps_ind_18_bin'].value_counts()/len(processed_test_data) #keep

processed_train_data['ps_car_01_cat'].value_counts()/len(processed_train_data)
processed_test_data['ps_car_01_cat'].value_counts()/len(processed_test_data) #keep

processed_train_data['ps_car_02_cat'].value_counts()/len(processed_train_data)
processed_test_data['ps_car_02_cat'].value_counts()/len(processed_test_data) #keep

processed_train_data['ps_car_04_cat'].value_counts()/len(processed_train_data)
processed_test_data['ps_car_04_cat'].value_counts()/len(processed_test_data) #keep

processed_train_data['ps_car_06_cat'].value_counts()/len(processed_train_data)
processed_test_data['ps_car_06_cat'].value_counts()/len(processed_test_data) #keep

processed_train_data['ps_car_07_cat'].value_counts()/len(processed_train_data)
processed_test_data['ps_car_07_cat'].value_counts()/len(processed_test_data) #keep

processed_train_data['ps_car_08_cat'].value_counts()/len(processed_train_data)
processed_test_data['ps_car_08_cat'].value_counts()/len(processed_test_data) #keep

processed_train_data['ps_car_09_cat'].value_counts()/len(processed_train_data)
processed_test_data['ps_car_09_cat'].value_counts()/len(processed_test_data) #keep

processed_train_data['ps_car_11_cat'].value_counts()/len(processed_train_data)
processed_test_data['ps_car_11_cat'].value_counts()/len(processed_test_data) #keep

processed_train_data['ps_car_11'].value_counts()/len(processed_train_data)
processed_test_data['ps_car_11'].value_counts()/len(processed_test_data) #keep

#combine test and train set and find the mean/ std deviation for normalizing continous features
combined_data = processed_train_data[['ps_reg_01','ps_reg_02','ps_reg_03','ps_car_12', 'ps_car_13', 'ps_car_15', 'ps_calc_02']] 

combined_data.append(processed_test_data[['ps_reg_01','ps_reg_02','ps_reg_03','ps_car_12', 'ps_car_13', 'ps_car_15', 'ps_calc_02']]) 

continous_features = combined_data.columns.values

for feature in continous_features:
    mean,std = combined_data[feature].mean(), combined_data[feature].std()
    processed_train_data.loc[:, feature] = (processed_train_data[feature]-mean)/std
    processed_test_data.loc[:,feature] = (processed_test_data[feature]-mean)/std
    


#write out processed dat to csv
processed_train_data.to_csv('processed_train.csv', sep=',', index=False)
processed_test_data.to_csv('processed_test.csv', sep=',', index=False)
