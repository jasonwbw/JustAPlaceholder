# -*- coding: utf-8 -*-

import numpy as np
import xgboost as xgb


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    # return a pair metric_name, result
    # since preds are margin(before logistic transformation, cutoff at 0)
    return 'error', float(sum(labels != (preds > 0.0))) / len(labels)


def cv(feature_prefix, feature_name, params, num_round=1000, early_stopping_rounds=10, kfolder=10):
    vals = []
    for i in xrange(kfolder):
        train_f = feature_prefix + '/Folder%d/' % i + \
            feature_name + '.train.xgboost.4rank.txt'
        test_f = feature_prefix + '/Folder%d/' % i + \
            feature_name + '.test.xgboost.4rank.txt'
        bst, eresult = train(train_f, test_f, params,
                             num_round, early_stopping_rounds, evaluate=True)
        vals.append(eresult)
    return vals


def train(train_f, test_f, params, num_round, early_stopping_rounds, evaluate=False):
    train_group_f = train_f.replace('.txt', '.txt.group')
    dtrain = xgb.DMatrix(train_f)
    dtest = xgb.DMatrix(test_f)
    dtrain.set_group(np.loadtxt(train_group_f).astype(int))
    if evaluate:
        test_group_f = test_f.replace('.txt', '.txt.group')
        dtest.set_group(np.loadtxt(test_group_f).astype(int))
    else:
    	dval = xgb.DMatrix(train_f.replace('train', 'test'))
    	dval.set_group(np.loadtxt(train_group_f.replace('train', 'test')).astype(int))
    if evaluate:
        watchlist = [(dtrain, 'train'), (dtest, 'valid')]
    else:
        watchlist = [(dtrain, 'train'), (dval, 'valid')]
    bst = xgb.train(params, dtrain, num_round, watchlist, obj=None,
                    feval=None, early_stopping_rounds=early_stopping_rounds)
    return bst, dtest if not evaluate else bst.eval(dtest)


def normed_by_group(preds, groups):
	min_v = np.min(preds)
	max_v = np.max(preds)
	print min_v, max_v
	for lines in groups:
		if len(lines) == 1:
			preds[lines[0]] = 0
			continue
		tmp = preds[lines]
		candidates = (tmp - min_v) / (max_v - min_v)
		for i, l in enumerate(lines):
			preds[l] = candidates[i]
	return preds


def submit(bst, dtest):
    # make prediction
    preds = bst.predict(dtest)
    print preds.shape
    groups = {}
    with open('../data/0_raw/validate_nolabel.txt', 'r') as fp:
    	for i, line in enumerate(fp):
    		qid, uid = line.strip().split()
    		if qid in groups:
    			groups[qid].append(i)
    		else:
    			groups[qid] = [i]
    preds = normed_by_group(preds, groups.values())
    with open('submit.csv', 'w') as fo:
        fo.write('qid,uid,label\n')
        with open('../data/0_raw/validate_nolabel.txt', 'r') as fp:
            for i, line in enumerate(fp):
                fo.write(line.strip().replace('\t', ',') +
                         ',' + str(preds[i]) + '\n')


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
                                # params['objective'] = 'rank:pairwise'
                                # params['objective'] = 'rank:ndcg'
                                params['eval_metric'] = ['ndcg@5', 'ndcg@10']
                                evals = cv('../feature/feature', feature_name, params,
                                           num_round=1000, early_stopping_rounds=5, kfolder=10)
                                # print('%d %d %f %d %f %f %d' % (m1, m2, eta, m3, subsample, colsample_bytree, w))
                                # print('\n'.join(evals) + '\n\n')
                                fo.write('%d %d %f %d %f %f %d' % (
                                    m1, m2, eta, m3, subsample, colsample_bytree, w))
                                fo.write('\n'.join(evals) + '\n\n')
                                fo.flush()
    fo.close()


feature_prefix = '../feature/feature'
# feature_name = 'stat'
feature_name = 'merge.stat_tags'
# gradsearch(feature_name=feature_name)


params = {'min_child_weight': 1, 'max_depth': 3, 'eta': 0.1,
          'max_delta_step': 1, 'subsample': 0.7, 'colsample_bytree': 0.7}
params['scale_pos_weight'] = 1
params['silent'] = True
params['objective'] = 'reg:logistic'
# params['objective'] = 'rank:pairwise'
# params['objective'] = 'rank:ndcg'
params['eval_metric'] = ['ndcg@5-', 'ndcg@10-']
train_f = feature_prefix + '/Folder1/' + feature_name + '.train.xgboost.4rank.txt'
test_f = feature_prefix + '/' + feature_name + '.test.xgboost.txt'
bst, dtest = train(train_f, test_f, params, 1000, 50, evaluate=False)
submit(bst, dtest)
