# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 18:12:26 2017

@author: Venky
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

train_data.isnull().sum()
train = train_data['libor_rate'].mean()
test = test_data['libor_rate'].mean()
train1 = train_data['sold'].mean()
train2 = train_data['bought'].mean()
test_data.isnull().sum()
train_data['libor_rate'].fillna(train, inplace=True)
train_data['sold'].fillna(train1, inplace=True)
train_data['bought'].fillna(train2, inplace=True)

test_data['libor_rate'].fillna(train, inplace=True)



output = pd.Series(train_data['return'], dtype='float64')
train_data.drop(['portfolio_id', 'desk_id', 'start_date','creation_date','sell_date','return','hedge_value','indicator_code'],axis=1,inplace=True)
test_data.drop(['portfolio_id', 'desk_id', 'start_date','creation_date','sell_date','hedge_value','indicator_code'],axis=1,inplace=True)
train_data.status.value_counts()
train_data['status'].fillna(False, inplace=True)


label = LabelEncoder()
train_data.dtypes


test_data.isnull().sum()
test_data.status.value_counts()
test_data['status'].fillna(False, inplace=True)

train_data['office_id'] = label.fit_transform(train_data['office_id'])
train_data['pf_category'] = label.fit_transform(train_data['pf_category'])
train_data['country_code'] = label.fit_transform(train_data['country_code'])
train_data['currency'] = label.fit_transform(train_data['currency'])
train_data['type'] = label.fit_transform(train_data['type'])
train_data['status'] = label.fit_transform(train_data['status'])

test_data['office_id'] = label.fit_transform(test_data['office_id'])
test_data['pf_category'] = label.fit_transform(test_data['pf_category'])
test_data['country_code'] = label.fit_transform(test_data['country_code'])
test_data['currency'] = label.fit_transform(test_data['currency'])
test_data['type'] = label.fit_transform(test_data['type'])
test_data['status'] = label.fit_transform(test_data['status'])

for col in train_data.columns:
    train_data[col] = train_data[col].astype('float64')

for col in train_data.columns:
    test_data[col] = test_data[col].astype('float64')
    

    
xgb_tr = xgb.DMatrix(train_data, label=output)

xgb_ts = xgb.DMatrix(test_data)

param = {}
param['objective'] = 'reg:linear'
param['eta'] = 0.1
param['max_delta_step'] = 2000
param['max_depth'] = 7		#20
# # param['num_class'] = 2
# param['lambda']=0.0001
# param['subsample'] = 0.85
# # param['colsample_bytree'] = 1
# param['gamma'] = 0.1
param['min_child_weight'] = 100		#100
num_round = 96
    

gbm = xgb.train(param,xgb_tr,num_round)

test_pred = gbm.predict(xgb_ts)
    
sub1 = pd.read_csv('sample_submission.csv')
sub1['return'] = test_pred
sub1.dtypes
sub1.to_csv('sub7.csv', index=False)





