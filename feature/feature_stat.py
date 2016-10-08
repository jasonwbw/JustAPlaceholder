# -*- coding: utf-8 -*-

import pickle

import numpy as np

data_folder = '../data/'


def question_load_scores(fname, q2feature):
	with open(fname, 'r') as fp:
		for line in fp:
			q, score = map(int, line.strip().split('\t'))
			if q in q2feature:
				q2feature[q].append(score)
			else:
				q2feature[q] = [score]
	return q2feature


def question():
	print 'load question'
	q2feature = {}
	with open(data_folder + 'stat/question_info.txt', 'r') as fp:
		for line in fp:
			items = map(float, line.strip().split('\t'))
			q2feature[int(items[0])] = items[1:]
	return q2feature


def user():
	print 'load user'
	u2feature = {}
	with open(data_folder + 'stat/user_info.txt', 'r') as fp:
		for line in fp:
			items = map(float, line.strip().split('\t'))
			u2feature[int(items[0])] = items[1:]
	return u2feature


def tag_matching():
	print 'load tags'
	qtags = {}
	utags = {}
	with open(data_folder + '1_reorder/question_info.txt', 'r') as fp:
		for line in fp:
			q, tags, w, c, x1, x2, x3 = line.strip().split('\t')
			qtags[int(q)] = set(map(int, tags.split('/')))
	with open(data_folder + '1_reorder/user_info.txt', 'r') as fp:
		for line in fp:
			try:
				u, tags, w, c = line.strip().split('\t')
			except:
				u, tags, wc = line.strip().split('\t')
			utags[int(u)] = set(map(int, tags.split('/')))
	return qtags, utags


def gen_feature_file():
	q2feature = question()
	u2feature = user()
	qtags, utags = tag_matching()
	# train
	fo = open('./1_feature/train.feature.pkl', 'w')
	fo_l = open('./1_feature/train.label.pkl', 'w')
	fo_4xgboost = open('./1_feature/train.xgboost.txt', 'w')
	features = []
	labels = []
	tag_match_count, tag_match_with_label1 = 0, 0
	with open(data_folder + '1_reorder/invited_info_train.txt', 'r') as fp:
		for line in fp:
			q, u, label = map(int, line.strip().split('\t'))
			tag_match = 0
			for tag in qtags[q]:
				if tag in utags[u]:
					tag_match = 1
					# print q, u, label
					if label == 1:
						tag_match_with_label1 += 1
					tag_match_count += 1
					break
			fea = q2feature[q] + u2feature[u] + [tag_match]
			features.append(fea)
			labels.append(label)
			fo_4xgboost.write(str(label) + ' ' + ' '.join([str(i) + ':' + str(f) for i, f in enumerate(fea)]) + '\n')
	print float(tag_match_with_label1) / tag_match_count, tag_match_with_label1, tag_match_count
	pickle.dump(np.array(features, dtype='float32'), fo)
	pickle.dump(np.array(label, dtype='int32'), fo_l)
	fo.close()
	fo_l.close()
	fo_4xgboost.close()
	# test
	fo = open('./1_feature/test.feature.pkl', 'w')
	fo_4xgboost = open('./1_feature/test.xgboost.txt', 'w')
	features = []
	with open(data_folder + '1_reorder/validate_nolabel.txt', 'r') as fp:
		for line in fp:
			q, u = map(int, line.strip().split('\t'))
			tag_match = 0
			for tag in qtags[q]:
				if tag in utags[u]:
					tag_match = 1
					break
			fea = q2feature[q] + u2feature[u] + [tag_match]
			features.append(fea)
			fo_4xgboost.write('0 ' + ' '.join([str(i) + ':' + str(f) for i, f in enumerate(fea)]) + '\n')
	pickle.dump(np.array(features, dtype='float32'), fo)
	fo.close()
	fo_4xgboost.close()


if __name__ == '__main__':
	gen_feature_file()
