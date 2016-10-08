# -*- coding: utf-8 -*-

import xgboost as xgb


def cv(feature_prefix, feature_name, params, num_round=1000, early_stopping_rounds=10, kfolder=10):
	vals = []
	for i in xrange(kfolder):
		train_f = feature_prefix + '/Folder%d/' % i + feature_name + '.train.xgboost.txt'
		test_f = feature_prefix + '/Folder%d/' % i + feature_name + '.test.xgboost.txt'
		dtrain = xgb.DMatrix(train_f)
		dtest = xgb.DMatrix(test_f)
		watchlist = [(dtrain, 'train'), (dtest, 'valid')]
		bst = xgb.train(params, dtrain, num_round, watchlist, obj=None, feval=None, early_stopping_rounds=early_stopping_rounds)
		eresult = bst.eval(dtest)
		vals.append(eresult)
	return vals


def train():
	pass


def submit(bst, dtest):
	# make prediction
	preds = bst.predict(dtest)
	print preds.shape
	with open('submit.csv', 'w') as fo:
		fo.write('qid,uid,label\n')
		with open('../data/0_raw/validate_nolabel.txt', 'r') as fp:
			for i, line in enumerate(fp):
				fo.write(line.strip().replace('\t', ',') + ',' + str(preds[i]) + '\n')


def gradsearch(feature_name='stat'):
	fo = open('gradsearch.%s.rs.txt' % feature_name, 'w')
	min_child_weights = [1, 2, 5]
	max_depths = [3, 4, 5]
	etas = [0.01, 0.05, 0.1]
	max_delta_steps = [0, 1, 5, 10]
	subsamples = [0.5, 0.7, 1]
	colsample_bytrees = [0.5, 0.7, 1]
	scale_pos_weights = [1, 5, 10]
	for m1 in min_child_weights:
		for m2 in max_depths:
			for eta in etas:
				for m3 in max_delta_steps:
					for subsample in subsamples:
						for colsample_bytree in colsample_bytrees:
							for w in scale_pos_weights:
								params = {}
								params['min_child_weight'] = m1
								params['max_depth'] = m2
								params['eta'] = eta
								params['max_delta_step'] = m3
								params['subsample'] = subsample
								params['colsample_bytree'] = colsample_bytree
								params['scale_pos_weight'] = w
								params['silent'] = True
								params['objective'] = 'reg:logistic'
								params['eval_metric'] = ['rmse']
								evals = cv('../feature/feature', feature_name, params, num_round=1000, early_stopping_rounds=5, kfolder=10)
								# print('%d %d %f %d %f %f %d' % (m1, m2, eta, m3, subsample, colsample_bytree, w))
								# print('\n'.join(evals) + '\n\n')
								fo.write('%d %d %f %d %f %f %d' % (m1, m2, eta, m3, subsample, colsample_bytree, w))
								fo.write('\n'.join(evals) + '\n\n')
								fo.flush()
	fo.close()


# params = {'min_child_weight': 5, 'max_depth': 3, 'eta': 0.1, 'max_delta_step': 5, 'subsample': 0.5, 'colsample_bytree': 1}
# params['scale_pos_weight'] = 1
# params['silent'] = True
# params['objective'] = 'reg:logistic'
# params['eval_metric'] = ['rmse']

# cv('../feature/feature', 'stat', params, num_round=1000, early_stopping_rounds=5, kfolder=10)
# feature_name = 'stat'
feature_name = 'merge.stat_tags'
gradsearch(feature_name=feature_name)
