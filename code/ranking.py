# -*- coding: utf-8 -*-

import numpy as np
import xgboost as xgb

import random


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


def submit(bst, dtest, need_norm=False):
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
    if need_norm:
        preds = normed_by_group(preds, groups.values())
    with open('submit.csv', 'w') as fo:
        fo.write('qid,uid,label\n')
        with open('../data/0_raw/validate_nolabel.txt', 'r') as fp:
            for i, line in enumerate(fp):
                fo.write(line.strip().replace('\t', ',') +
                         ',' + str(preds[i]) + '\n')


def gradsearch(feature_name='stat', kfolder=8, num_round=1000, early_stopping_rounds=20):
    fo = open('gradsearch.%s.rs.txt' % feature_name, 'w')
    min_child_weights = [1, 2, 5]
    max_depths = [2, 3, 4, 5]
    etas = [0.01, 0.05, 0.1]
    max_delta_steps = [0, 1, 5, 10]
    subsamples = [0.5, 0.7, 1]
    colsample_bytrees = [0.5, 0.7, 1]
    scale_pos_weights = [1, 5, 10]
    tmp_len = len(etas) * len(max_delta_steps) * len(subsamples) * len(colsample_bytrees) * len(scale_pos_weights)
    tmp_total_len = tmp_len * len(min_child_weights) * len(max_depths)
    best_result = (0, )
    for i, m1 in enumerate(min_child_weights):
        for j, m2 in enumerate(max_depths):
            fo.write('%d passed of %d\n\n' % (tmp_len * i * len(max_depths) + tmp_len * j, tmp_total_len))
            for eta in etas:
                for m3 in max_delta_steps:
                    for subsample in subsamples:
                        for colsample_bytree in colsample_bytrees:
                            for w in scale_pos_weights:
                                if random.randint(0, 9) != 0:
                                    continue
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
                                params['eval_metric'] = ['ndcg@5-', 'ndcg@10-']
                                evals = cv('../feature/feature', feature_name, params,
                                           num_round=num_round, early_stopping_rounds=early_stopping_rounds, kfolder=kfolder)
                                metrics = 0.
                                for eva in evals:
                                    eva_tmp = eva.split('eval-ndcg@', 2)
                                    ndcg_at5 = eva_tmp[1].strip().replace('5-:', '')
                                    ndcg_at10 = eva_tmp[2].strip().replace('10-:', '')
                                    metrics += (float(ndcg_at5) + float(ndcg_at10)) / 2
                                metrics /= len(evals)
                                if metrics > best_result[0]:
                                    best_result = (metrics, m1, m2, eta, m3, subsample, colsample_bytree, w)
                                fo.write('%d %d %f %d %f %f %d\n' % (
                                    m1, m2, eta, m3, subsample, colsample_bytree, w))
                                fo.write('\n'.join(evals) + '\n')
                                fo.write('average (ndcg@5 + ndcg@10)/2 %f\n\n' % metrics)
                                fo.flush()
    fo.write('the best params and result is\nndcg@5 + ndcg@10)/2 = %f\nparams is %d %d %f %d %f %f %d\n' % best_result)
    fo.close()


feature_prefix = '../feature/feature'
# feature_name = 'stat'
feature_name = 'merge.stat_tags'
# feature_name = 'merge.stat_tags_ngram'
gradsearch(feature_name=feature_name, kfolder=3)


# params = {'min_child_weight': 1, 'max_depth': 2, 'eta': 0.1,
#           'max_delta_step': 0, 'subsample': 0.7, 'colsample_bytree': 1}
# params['scale_pos_weight'] = 5
# params['silent'] = True
# params['objective'] = 'binary:logistic'
# # params['objective'] = 'rank:pairwise'
# # params['objective'] = 'rank:ndcg'
# params['eval_metric'] = ['ndcg@5-', 'ndcg@10-']
# train_f = feature_prefix + '/Folder1/' + feature_name + '.train.xgboost.4rank.txt'
# test_f = feature_prefix + '/' + feature_name + '.test.xgboost.txt'
# bst, dtest = train(train_f, test_f, params, 1000, 100, evaluate=False)
# submit(bst, dtest)
