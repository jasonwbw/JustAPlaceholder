# -*- coding: utf-8 -*-
# 使用../data/stat.py统计的基本数字型特征


import pickle

import numpy as np

data_folder = '../data/'


class FeatureGenerator:

	def gen_feauture(self, q, u):
		pass

	def gen_feature_file(self, folder, fname, train=True, xgboost=True, pkl=True):
		if train:
			fname += '.train'
			data_file = data_folder + '1_reorder/invited_info_train.txt'
		else:
			fname += '.test'
			data_file = data_folder + '1_reorder/validate_nolabel.txt'
		fo_4xgboost = open('%s%s.xgboost.txt' % (folder, fname), 'w')
		features = []
		labels = []
		with open(data_file, 'r') as fp:
			for line in fp:
				if train:
					q, u, label = map(int, line.strip().split('\t'))
				else:
					q, u = map(int, line.strip().split('\t'))
				fea = self.gen_feauture(q, u)
				features.append(fea)
				if train:
					labels.append(label)
					fo_4xgboost.write(str(label) + ' ' + ' '.join([str(i) + ':' + str(f) for i, f in enumerate(fea)]) + '\n')
				else:
					fo_4xgboost.write('0 ' + ' '.join([str(i) + ':' + str(f) for i, f in enumerate(fea)]) + '\n')
		fo_4xgboost.close()
		fo = open('%s%s.feature.pkl' % (folder, fname), 'w')
		pickle.dump(np.array(features, dtype='float32'), fo)
		fo.close()
		if train:
			fo_l = open('%s%s.label.pkl' % (folder, fname), 'w')
			pickle.dump(np.array(label, dtype='int32'), fo_l)
			fo_l.close()
# end FeatureGenerator
