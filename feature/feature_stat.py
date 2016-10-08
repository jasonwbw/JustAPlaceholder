# -*- coding: utf-8 -*-
# 使用../data/stat.py统计的基本数字型特征
#
# Feature List:
# [Query about]
# 00-02 - 问题的 [vote/ans/bestans]
# [User about]
# 03-06 - 回答问题的 [平均/最大/最小/方差] [vote/ans/bestans]
# 07-10 - 拒绝问题的 [平均/最大/最小/方差] [vote/ans/bestans]
# 11-12 - 回答率, 已经回答的个数

import os

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


def user(kfolder):
	'''Load预先统计的用户信息'''
	print 'load user'
	fname = data_folder + 'stat/user_info%s.txt' % ('' if kfolder == -1 else ('_Folder' + str(kfolder)))
	u2feature = {}
	with open(fname, 'r') as fp:
		for line in fp:
			items = map(float, line.strip().split('\t'))
			u2feature[int(items[0])] = items[1:]
	return u2feature


class StatFeatureGenerator(FeatureGenerator):

	def __init__(self, q2feature, u2feature):
		self.q2feature = q2feature
		self.u2feature = u2feature
		for i in xrange(100):
			if i in q2feature:
				self.qfeature_len = len(q2feature[i])
				break
		for i in xrange(100):
			if i in u2feature:
				self.ufeature_len = len(u2feature[i])
				break

	def gen_feauture(self, q, u):
		if q not in self.q2feature:
			qfea = [0] * self.qfeature_len
		else:
			qfea = self.q2feature[q]
		if u not in self.u2feature:
			ufea = [0] * self.ufeature_len
		else:
			ufea = self.u2feature[u]
		fea = qfea + ufea
		return fea
# end StatFeatureGenerator


def gen_feature(kfolder=-1):
	if kfolder != -1:
		print 'current fit %d folder' % kfolder
	q2feature = question()
	u2feature = user(kfolder)
	g = StatFeatureGenerator(q2feature, u2feature)
	prefix = 'stat' if kfolder == -1 else 'Folder%d/stat' % kfolder
	if kfolder != -1 and not os.path.exists('./feature/Folder%d' % kfolder):
		os.makedirs('./feature/Folder%d' % kfolder)
	print 'start generating feature for training data'
	g.gen_feature_file('./feature/', prefix, train=True, xgboost=True, pkl=False, kfolder=kfolder)
	print 'start generating feature for test data'
	g.gen_feature_file('./feature/', prefix, train=False, xgboost=True, pkl=False, kfolder=kfolder)
	print ''


if __name__ == '__main__':
	for i in xrange(-1, 10):
		gen_feature(i)
