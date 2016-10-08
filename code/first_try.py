# -*- coding: utf-8 -*-

import numpy as np
import xgboost as xgb

train_f = '../feature/1_feature/train.xgboost.txt'
test_f = '../feature/1_feature/test.xgboost.txt'
# train_f = '../feature/1_feature/train.tfidf.xgboost.txt'
# test_f = '../feature/1_feature/test.tfidf.xgboost.txt'

dtrain = xgb.DMatrix(train_f)
dtest = xgb.DMatrix(test_f)

param = {'min_child_weight':5, 'max_depth':3, 'eta':0.1, 'max_delta_step':5, 'subsample':0.5, 'colsample_bytree':1, 'scale_pos_weight':1,
        'silent':1 ,'objective':'binary:logistic', 'eval_metric':['rmse', 'error']}
# param = {'min_child_weight':5, 'max_depth':4, 'eta':0.05, 'gamma':1, 'subsample':0.7, 'colsample_bytree':0.5, 'scale_pos_weight':1,
#     'silent':1 ,'objective':'binary:logistic'}
num_round = 1000

print ('running cross validation')
# do cross validation, this will print result out as
# [iteration]  metric_name:mean_value+std_value
# std_value is standard deviation of the metric
xgb.cv(param, dtrain, num_round, nfold=5,
       metrics=['rmse', 'error'], seed=0,
       callbacks=[xgb.callback.print_evaluation(show_stdv=True),
                        xgb.callback.early_stop(10)])
# xgb.cv(param, dtrain, num_round, nfold=5,
#        metrics={'rmse', 'error'}, seed=0,
#        callbacks=[xgb.callback.print_evaluation(show_stdv=True),
#                         xgb.callback.early_stop(10)])

# num_round = 1000
# param['silent'] = 1
# watchlist  = [(dtrain, 'train')]

# bst = xgb.train(param, dtrain, num_round, watchlist, obj=None, feval=None, early_stopping_rounds=10)
# # make prediction
# preds = bst.predict(dtest)
# print preds.shape
# with open('submit.csv', 'w') as fo:
# 	fo.write('qid,uid,label\n')
# 	with open('../data/0_raw/validate_nolabel.txt', 'r') as fp:
# 		for i, line in enumerate(fp):
# 			fo.write(line.strip().replace('\t', ',') + ',' + str(preds[i]) + '\n')
