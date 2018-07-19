#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:39:00 2017

@author: ethan
"""

import pandas as pd
import numpy as np
import os
from sklearn.utils import resample #module for resampling


#import training data
#os.chdir('/home/ethan/Documents/P556 Project 3/Data')

#raw data is the kaggle training data
data = pd.read_csv('./data/processed_train.csv')
#raw_data = pd.read_csv('augmented_data.csv', na_values = -1)

#look for null data
data.isnull().sum()


"""
Create training, testing, and validation set from data
"""

#split data based on its label
claim = data.loc[data['target']==1]
no_claim = data.loc[data['target']==0]

#randomize the order of data
claim = claim.reindex(np.random.permutation(claim.index))
no_claim = no_claim.reindex(np.random.permutation(no_claim.index))

#seperate claim into training, testing, validation
#print length all to make sure it is divided to 80, 10, 10
claim_training = claim.sample(frac=0.8)
claim_testing = claim.loc[~claim.index.isin(claim_training.index)]


#separate the no_claim into training, testing, validation
#print length all to make sure it is divided to 80, 10, 10
no_claim_training = no_claim.sample(frac=0.8)
no_claim_testing = no_claim.loc[~no_claim.index.isin(no_claim_training.index)]



#create full training, testing, and validation data by merging and randomizing claim and no_claim data
training_data = pd.concat([claim_training, no_claim_training], axis = 0)
training_data = training_data.reindex(np.random.permutation(training_data.index))
testing_data = pd.concat([claim_testing, no_claim_testing], axis = 0)
testing_data = testing_data.reindex(np.random.permutation(testing_data.index))


training_data.to_csv('./data/training_data.csv', sep=',', index=False, header=None)
testing_data.to_csv('./data/testing_data.csv', sep=',', index=False, header=None)


#created  augmented training data to train w and d network
augmented_claim = resample(claim_training, replace=True, n_samples=400000, random_state=234)
augmented_no_claim = resample(no_claim_training, replace=False, n_samples=400000, random_state=213)

training_data_augmented = pd.concat([augmented_no_claim,augmented_claim])

training_data_augmented = training_data_augmented.sample(frac=1)

training_data_augmented.to_csv('./data/training_data_augmented.csv', sep=',', index=False, header=None)
