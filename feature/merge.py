# -*- coding: utf-8 -*-


from itertools import izip

def load_feature_file(fname, max_feature=0, current_features=None, need_label=False):
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


def back(current_features, labels, fname):
	with open(fname, 'w') as fo:
		for label, feature in izip(labels, current_features):
			fo.write(label)
			for idx, value in feature:
				fo.write(' %s:%s' % (idx, value))
			fo.write(' \n')


def merge():
	current_features, labels, max_feature = load_feature_file('./1_feature/train.xgboost.txt', need_label=True)
	current_features, x, max_feature = load_feature_file('./1_feature/train.tfidf.xgboost.txt', max_feature=max_feature, current_features=current_features)
	back(current_features, labels, './1_feature/train.xgboost.merge1.txt')
	current_features, labels, max_feature = load_feature_file('./1_feature/test.xgboost.txt', need_label=True)
	current_features, x, max_feature = load_feature_file('./1_feature/test.tfidf.xgboost.txt', max_feature=max_feature, current_features=current_features)
	back(current_features, labels, './1_feature/test.xgboost.merge1.txt')


if __name__ == '__main__':
	merge()
