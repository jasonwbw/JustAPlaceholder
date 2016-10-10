# -*- coding: utf-8 -*-

import config
from itertools import izip


def load_xgboost_feature_file(fname, max_feature=0, current_features=None, need_label=False):
	labels = []
	new_max_feature = 0
	features = current_features if current_features is not None else []
	with open(fname, 'r') as fp:
		for i, line in enumerate(fp):
			items = line.strip().split()
			if need_label:
				labels.append(items[0])
			for j, item in enumerate(items[1:]):
				idx, value = item.split(':')
				idx = int(idx)
				new_max_feature = max(new_max_feature, idx)
				if current_features is None and j == 0:
					features.append([])
				features[i].append((str(idx + max_feature), value))
	return features, labels, new_max_feature + max_feature


def back_xgboost_feature(current_features, labels, fname):
	with open(fname, 'w') as fo:
		for label, feature in izip(labels, current_features):
			fo.write(label)
			for idx, value in feature:
				fo.write(' %s:%s' % (idx, value))
			fo.write(' \n')


def merge_xgboost_feature(file_prefixs, file_merge_prefix):
	for i, file_prefix in enumerate(file_prefixs):
		print 'load %dth features and renumber the feauter index' % (i + 1)
		if i == 0:
			current_features, labels, max_feature = load_xgboost_feature_file(file_prefix + 'train.xgboost.txt', need_label=True)
			current_features_test, labels_test, max_feature_test = load_xgboost_feature_file(file_prefix + 'test.xgboost.txt', need_label=True)
		else:
			current_features, x, max_feature = load_xgboost_feature_file(file_prefix + 'train.xgboost.txt', max_feature=max_feature, current_features=current_features)
			current_features_test, x_test, max_feature_test = load_xgboost_feature_file(file_prefix + 'test.xgboost.txt', max_feature=max_feature_test, current_features=current_features_test)
	back_xgboost_feature(current_features, labels, file_merge_prefix + 'train.xgboost.txt')
	back_xgboost_feature(current_features_test, labels_test, file_merge_prefix + 'test.xgboost.txt')
	print ''


if __name__ == '__main__':
	# merge stat and tags
	featurenames = ['stat', 'tags', 'ngram']
	merged_name = 'merge.%s.' % '_'.join(featurenames)
	merge_xgboost_feature(['./feature/%s.' % f for f in featurenames], './feature/%s' % merged_name)
	for i in xrange(config.kfolder):
		merge_xgboost_feature(['./feature/Folder%d/%s.' % (i, f) for f in featurenames], './feature/Folder%d/%s' % (i, merged_name))
