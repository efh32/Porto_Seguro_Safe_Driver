#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 20:10:32 2017

The goal of this file is to augment the training data in order to fix balance the
labels in the data.


@author: ethan
"""

import pandas as pd
import numpy as np
import os
from sklearn.utils import resample #module for resampling

#import training data
#os.chdir('/home/ethan/Documents/P556 Project 3/Data')


#raw data is the kaggle training data
processed_test_data = pd.read_csv('processed_train.csv')

#split data based on its label
claim = processed_test_data.loc[processed_test_data['target']==1]
no_claim = processed_test_data.loc[processed_test_data['target']==0]

#check length of claim and no_claim
processed_test_data.target.value_counts()

#find ration of co_claim to claim
len(no_claim)/len(claim)

augmented_no_claim = resample(no_claim, replace=False, n_samples=300000, random_state=123)
augmented_claim = resample(claim, replace=True, n_samples=300000, random_state=123)

data= pd.concat([augmented_no_claim,augmented_claim])

data = data.sample(frac=1)

data.to_csv('augmented_data.csv', sep=',', index=False)

#check whether csv file works
check_augment_data = pd.read_csv('augmented_data.csv')
