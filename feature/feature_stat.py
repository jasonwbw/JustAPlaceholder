# -*- coding: utf-8 -*-
# 使用../data/stat.py统计的基本数字型特征


from feature_abstract import FeatureGenerator

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
	'''Load预先统计的问题信息'''
	print 'load question'
	q2feature = {}
	with open(data_folder + 'stat/question_info.txt', 'r') as fp:
		for line in fp:
			items = map(float, line.strip().split('\t'))
			q2feature[int(items[0])] = items[1:]
	return q2feature


def user():
	'''Load预先统计的用户信息'''
	print 'load user'
	u2feature = {}
	with open(data_folder + 'stat/user_info.txt', 'r') as fp:
		for line in fp:
			items = map(float, line.strip().split('\t'))
			u2feature[int(items[0])] = items[1:]
	return u2feature


class StatFeatureGenerator(FeatureGenerator):

	def __init__(self, q2feature, u2feature):
		self.q2feature = q2feature
		self.u2feature = u2feature

	def gen_feauture(self, q, u):
		fea = self.q2feature[q] + self.u2feature[u]
		return fea
# end StatFeatureGenerator


if __name__ == '__main__':
	q2feature = question()
	u2feature = user()
	g = StatFeatureGenerator(q2feature, u2feature)
	print 'start generating feature for training data'
	g.gen_feature_file('./feature/', 'stat', train=True, xgboost=True, pkl=True)
	print 'start generating feature for test data'
	g.gen_feature_file('./feature/', 'stat', train=False, xgboost=True, pkl=True)
