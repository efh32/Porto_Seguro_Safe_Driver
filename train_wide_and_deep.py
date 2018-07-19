#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:52:11 2017

@author: ethan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import sys

import tensorflow as tf



CSV_COLUMNS = ['id','target1','ps_ind_03','ps_ind_04_cat',
          'ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin',
          'ps_ind_09_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin',
          'ps_ind_17_bin','ps_ind_18_bin','ps_reg_01','ps_reg_02','ps_reg_03',
          'ps_car_01_cat','ps_car_02_cat','ps_car_04_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat',
          'ps_car_09_cat','ps_car_11_cat','ps_car_11',
          'ps_car_12','ps_car_13','ps_car_15',
          'ps_calc_02']




# 3 ind cat
ps_ind_04_cat = tf.feature_column.categorical_column_with_identity(
    key="ps_ind_04_cat", num_buckets=3, default_value=2)
ps_ind_05_cat = tf.feature_column.categorical_column_with_identity(
    key="ps_ind_05_cat", num_buckets=8, default_value=7)

# 11 ind bin
ps_ind_06_bin = tf.feature_column.categorical_column_with_identity(
    key="ps_ind_06_bin", num_buckets=2, default_value=0)
ps_ind_07_bin = tf.feature_column.categorical_column_with_identity(
    key="ps_ind_07_bin", num_buckets=2, default_value=0)
ps_ind_08_bin = tf.feature_column.categorical_column_with_identity(
    key="ps_ind_08_bin", num_buckets=2, default_value=0)
ps_ind_09_bin = tf.feature_column.categorical_column_with_identity(
    key="ps_ind_09_bin", num_buckets=2, default_value=0)
ps_ind_16_bin = tf.feature_column.categorical_column_with_identity(
    key="ps_ind_16_bin", num_buckets=2, default_value=0)
ps_ind_17_bin = tf.feature_column.categorical_column_with_identity(
    key="ps_ind_17_bin", num_buckets=2, default_value=0)
ps_ind_18_bin = tf.feature_column.categorical_column_with_identity(
    key="ps_ind_18_bin", num_buckets=2, default_value=0)

# 9 car cat
ps_car_01_cat = tf.feature_column.categorical_column_with_identity(
    key="ps_car_01_cat", num_buckets=13, default_value=12)
ps_car_02_cat = tf.feature_column.categorical_column_with_identity(
    key="ps_car_02_cat", num_buckets=3, default_value=2)
ps_car_04_cat = tf.feature_column.categorical_column_with_identity(
    key="ps_car_04_cat", num_buckets=11, default_value=10)
ps_car_06_cat = tf.feature_column.categorical_column_with_identity(
    key="ps_car_06_cat", num_buckets=19, default_value=18)
ps_car_07_cat = tf.feature_column.categorical_column_with_identity(
    key="ps_car_07_cat", num_buckets=3, default_value=2)
ps_car_08_cat = tf.feature_column.categorical_column_with_identity(
    key="ps_car_08_cat", num_buckets=3, default_value=2)
ps_car_09_cat = tf.feature_column.categorical_column_with_identity(
    key="ps_car_09_cat", num_buckets=6, default_value=5)
ps_car_11_cat = tf.feature_column.categorical_column_with_identity(
    key="ps_car_11_cat", num_buckets=106, default_value=105)





# 4 ind numeric
ps_ind_03 = tf.feature_column.numeric_column("ps_ind_03")
ps_ind_14 = tf.feature_column.numeric_column("ps_ind_14")
ps_ind_15 = tf.feature_column.numeric_column("ps_ind_15")

# 3 reg numeric
ps_reg_01 = tf.feature_column.numeric_column("ps_reg_01")
ps_reg_02 = tf.feature_column.numeric_column("ps_reg_02")
ps_reg_03 = tf.feature_column.numeric_column("ps_reg_03")

# 5 car numeric
ps_car_11 = tf.feature_column.numeric_column("ps_car_11")
ps_car_12 = tf.feature_column.numeric_column("ps_car_12")
ps_car_13 = tf.feature_column.numeric_column("ps_car_13")
ps_car_15 = tf.feature_column.numeric_column("ps_car_15")

# 14 calc numeric
ps_calc_02 = tf.feature_column.numeric_column("ps_calc_02")




"""
Define Wide and Deep columns
"""

deep_columns = [
  # 3 ps_ind_??_cat

  tf.feature_column.indicator_column(ps_ind_04_cat),
  tf.feature_column.indicator_column(ps_ind_05_cat),
  # 11 ps_ind_??_bin
  tf.feature_column.indicator_column(ps_ind_06_bin),
  tf.feature_column.indicator_column(ps_ind_07_bin),
  tf.feature_column.indicator_column(ps_ind_08_bin),
  tf.feature_column.indicator_column(ps_ind_09_bin),

  tf.feature_column.indicator_column(ps_ind_16_bin),
  tf.feature_column.indicator_column(ps_ind_17_bin),
  tf.feature_column.indicator_column(ps_ind_18_bin),
  # 9 ps_car_??_cat
  tf.feature_column.indicator_column(ps_car_01_cat),
  tf.feature_column.indicator_column(ps_car_02_cat),
  tf.feature_column.indicator_column(ps_car_04_cat),
  tf.feature_column.indicator_column(ps_car_06_cat),
  tf.feature_column.indicator_column(ps_car_07_cat),
  tf.feature_column.indicator_column(ps_car_08_cat),
  tf.feature_column.indicator_column(ps_car_09_cat),

  tf.feature_column.indicator_column(ps_car_11_cat),
  # 6 ps_calc_??_bin

  # 4 ps_ind_??
  ps_ind_03,
  ps_ind_14,
  ps_ind_15,
  # 3 ps_reg_??
  ps_reg_01,
  ps_reg_02,
  ps_reg_03,
  # 5 ps_car_??
  ps_car_11,
  ps_car_12,
  ps_car_13,
  ps_car_15,
  # 14 ps_calc_??

  ps_calc_02,
]

base_columns =[

  ps_ind_04_cat,
  ps_ind_05_cat,
  # 11 ps_ind_??_bin
  ps_ind_06_bin,
  ps_ind_07_bin,
  ps_ind_08_bin,
  ps_ind_09_bin,
  ps_ind_16_bin,
  ps_ind_17_bin,
  ps_ind_18_bin,
  # 9 ps_car_??_cat
  ps_car_01_cat,
  ps_car_02_cat,
  ps_car_04_cat,
  ps_car_06_cat,
  ps_car_07_cat,
  ps_car_08_cat,
  ps_car_09_cat,
  ps_car_11_cat,
  # 6 ps_calc_??_bin

        
]

crossed_columns = [
    tf.feature_column.crossed_column(
        ["ps_car_06_cat", "ps_car_11_cat"], hash_bucket_size=1000)
]

no_columns =[]

wide_columns = base_columns + crossed_columns

"""
Create Models
"""


def build_estimator(model_dir, model_type):
  """Build an estimator."""
  if model_type == "wide":
    m = tf.estimator.LinearClassifier(
        model_dir=model_dir, feature_columns=no_columns)
  elif model_type == "deep":
    m = tf.estimator.DNNClassifier(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=[100, 50, 50],
        optimizer=tf.train.ProximalAdagradOptimizer(
            learning_rate=0.001,
            l1_regularization_strength=0.001)
        )
  else:
    m = tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=no_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[30, 50],
        dnn_optimizer=tf.train.ProximalAdagradOptimizer(
            learning_rate=0.001,
            l1_regularization_strength=0.001,
            l2_regularization_strength=0.001)
        )
  return m

def input_fn(data_file, num_epochs, shuffle):
  """Input builder function."""
  df_data = pd.read_csv(
      tf.gfile.Open(data_file),
      names=CSV_COLUMNS,
      skipinitialspace=True,
      engine="python",
      skiprows=0)
  # remove NaN elements
  df_data = df_data.dropna(how="any", axis=0)
  labels = df_data["target1"].apply(lambda x: x == 1).astype(int)
  return tf.estimator.inputs.pandas_input_fn(
      x=df_data,
      y=labels,
      batch_size=20000,
      num_epochs=num_epochs,
      shuffle=shuffle,
      num_threads=5)


def maybe_download(train_data, test_data, train2_data=""):
  """Maybe downloads training data and returns train and test file names."""
  if train_data:
    train_file_name = train_data
  else:
    train_file_name = "training_data.csv"

  if test_data:
    test_file_name = test_data
  else:
    #test_file_name = "./data/test-1.csv"
    test_file_name = "testing_data.csv"
    
  if train2_data:
    train2_file_name = train2_data
  else:
    train2_file_name = "training_data_augmented.csv"
    #test_file_name = "./data/train-1.csv"

  return train_file_name, test_file_name, train2_file_name

"""
download the training, testing, validation data created from preprocess_data

"""
import os
import pandas as pd


os.chdir('/home/ethan/Documents/P556 Project 3/Data')



#set model director



def train_and_eval(model_dir, model_type, train_steps, train_data, test_data):
  """Train and evaluate the model."""
  train_file_name, test_file_name, train2_file_name = maybe_download(train_data, test_data)
  # Specify file path below if want to find the output easily
  model_dir = '/home/ethan/Documents/P556 Project 3/model_2'

  m = build_estimator(model_dir, model_type)
  # set num_epochs to None to get infinite stream of data.
  m.train(
      input_fn=input_fn(train2_file_name, num_epochs=None, shuffle=True),
      steps=train_steps)
  # set steps to None to run evaluation until all data consumed.
  results = m.evaluate(
      input_fn=input_fn(train_file_name, num_epochs=1, shuffle=False),
      steps=None)
  print("model directory = %s" % model_dir)
  for key in sorted(results):
    print("%s: %s" % (key, results[key]))
  
  results2 = m.evaluate(
      input_fn=input_fn(test_file_name, num_epochs=1, shuffle=False),
      steps=None)
  print("model directory = %s" % model_dir)
  for key in sorted(results2):
    print("%s: %s" % (key, results2[key]))
  # Manual cleanup
  #shutil.rmtree(model_dir)

FLAGS = None

def main(_):
    
  #clean model directory if present
  #shutil.rmtree(model_dir)
    
  train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
                 FLAGS.train_data, FLAGS.test_data)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="",
      help="Base directory for output models."
  )
  parser.add_argument(
      "--model_type",
      type=str,
      default="deep",
      help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
  )
  parser.add_argument(
      "--train_steps",
      type=int,
      default=1000,
      help="Number of training steps."
  )
  parser.add_argument(
      "--train_data",
      type=str,
      default="",
      help="Path to the training data."
  )
  parser.add_argument(
      "--test_data",
      type=str,
      default="",
      help="Path to the test data."
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)



