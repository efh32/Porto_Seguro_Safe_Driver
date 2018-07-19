#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 22:19:57 2017

@author: ethan
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import Imputer
from sklearn.utils import resample #module for resampling


#import results of augmented data and test data
#os.chdir('/home/ethan/Documents/P556 Project 3/Data/results/')
train_results = pd.read_csv('train_results.csv')
test_results = pd.read_csv('test_results.csv')
submission_results = pd.read_csv('submission.csv')


#add columns onto train results for accuracy
test_results['target2']= np.where(test_results['probability'] >= 0.9, 1, 0)
test_results['correct']= np.where(test_results['target2'] - test_results['target1'] == 0, 0, 1)


test_results.loc[test_results['probability'] > 0.9]
test_results.loc[test_results['probability'] > 0.8]

#find data to remove from second model
lower_limit = test_results.loc[test_results['probability']<=0.5] 
lower_accuracy = (len(lower_limit.loc[lower_limit['correct']==0]))/(len(lower_limit))




#create new training data and testing data
#os.chdir('/home/ethan/Documents/P556 Project 3/Data')
training_data = pd.read_csv('./data/training_data.csv', header=None)


remove_train_id = train_results.loc[train_results['probability']<=0.5, 'id'] 
training_data_2 = training_data.loc[~training_data.iloc[:,0].isin(remove_train_id)]


testing_data = pd.read_csv('./data/testing_data.csv', header=None)

remove_test_id = test_results.loc[test_results['probability']<=0.5, 'id']
testing_data_2 = testing_data.loc[~testing_data.iloc[:,0].isin(remove_test_id)]

training_data_2.to_csv('training_data_2.csv', sep=',', index=False, header=None)
testing_data_2.to_csv('testing_data_2.csv', sep=',', index=False, header=None)


#augment training_data_2
training_2_claim = training_data_2.loc[training_data_2.iloc[:,1]==1]
training_2_no_claim = training_data_2.loc[training_data_2.iloc[:,1]==0]

augmented_claim = resample(training_2_claim, replace=True, n_samples=180600, random_state=234)


training_data_2_augmented = pd.concat([training_2_no_claim,augmented_claim])

training_data_2_augmented = training_data_2_augmented.sample(frac=1)

training_data_2_augmented.to_csv('./data/training_data_augmented_2.csv', sep=',', index=False, header=None)


#final submission part 1
submission_results_part_1_id = submission_results.loc[submission_results['probability']<=0.5, 'id']
submission_results_part_1 = submission_results.loc[submission_results['id'].isin(submission_results_part_1_id)]

#use if 3 models are trained
submission_results_part_1['final target'] = submission_results_part_1['probability']*0.66
submission_results_part_1.to_csv('./data/submission_results_part_1.csv', sep=',', index=False)

#create test_2.csv for second model prediction
test_data = pd.read_csv('./data/processed_test.csv')

test_data_2 = test_data.loc[~test_data['id'].isin(submission_results_part_1_id)]
test_data_2.to_csv('./data/test_2.csv', sep=',', index=False)
