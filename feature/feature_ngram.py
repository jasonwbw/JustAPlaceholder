# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np

import config
import ngram

from feat_utils import try_divide
from feature_abstract import FeatureGenerator


data_folder = '../data/'

join_str = '_'

def load_texts(fname, column):
	texts = []
	with open(fname, 'r') as fp:
		for line in fp:
			items = line.strip().split('\t')
			text = items[column]
			if text == '/':
				texts.append([])
			else:
				texts.append(text.split('/'))
	# print texts[0]
	return texts


class NgramCoutingFeatureGenerator(FeatureGenerator):

	def __init__(self, q2text, u2text, grams=3):
		self.grams = grams
		self.q2text = [q2text]
		self.u2text = [u2text]
		if grams > 1:
			self.q2text.append(self._gen_gram(q2text, 2))
			self.u2text.append(self._gen_gram(u2text, 2))
			if grams > 2:
				self.q2text.append(self._gen_gram(q2text, 3))
				self.u2text.append(self._gen_gram(u2text, 3))

	def _gen_gram(self, texts, gram=2):
		ngram_texts = []
		for text in texts:
			if gram == 2:
				ngram_texts.append(ngram.getBigram(text, join_str))
			elif gram == 3:
				ngram_texts.append(ngram.getTrigram(text, join_str))
		return ngram_texts

	def gen_feauture(self, q, u):
		features = []
		for gram in xrange(self.grams):
			q_text = self.q2text[gram][q]
			u_text = self.u2text[gram][u]
			features += self._gen_word_count(q_text)
			features += self._gen_word_count(u_text, need_lack=True)
			features += self._intersect_word_count(q_text, u_text)
			features += self._intersect_word_position(q_text, u_text)
		return features

	def _gen_word_count(self, text, need_lack=False):
		# count: word, unique_word, unique_word / word
		count = len(text)
		count_of_unique = len(set(text))
		features = [count, count_of_unique, try_divide(count_of_unique, count)]
		if need_lack:
			lack_of_des = count > 0
			features.append(lack_of_des)
		return features

	def _intersect_word_count(self, t1, t2):
		# count: t1's word in t2, ratio to t1 and t2
		t2_set = set(t2)
		count = sum([1 for w in t1 if w in t2_set])
		features = [count, try_divide(count, len(t1)), try_divide(count, len(t2))]
		return features

	def _intersect_word_position(self, t1, t2):
		# stats feature on position and normed position
		t1_set = set(t1)
		t2_set = set(t2)
		positions = [i for i, w in enumerate(t1) if w in t2_set]
		positions = [0] if len(positions) == 0 else positions
		features = [np.max(positions), np.min(positions), np.mean(positions), np.median(positions), np.std(positions)]
		positions = [i for i, w in enumerate(t2) if w in t1_set]
		positions = [0] if len(positions) == 0 else positions
		features += [np.max(positions), np.min(positions), np.mean(positions), np.median(positions), np.std(positions)]
		return features
# end NgramCoutingFeatureGenerator


def gen_feature(g, kfolder=-1):
	if kfolder != -1:
		print 'current fit %d folder' % kfolder
	prefix = 'ngram' if kfolder == -1 else 'Folder%d/ngram' % kfolder
	if kfolder != -1 and not os.path.exists('./feature/Folder%d' % kfolder):
		os.makedirs('./feature/Folder%d' % kfolder)
	print 'start generating feature for training data'
	g.gen_feature_file('./feature/', prefix, train=True, xgboost=True, pkl=False, kfolder=kfolder)
	print 'start generating feature for test data'
	g.gen_feature_file('./feature/', prefix, train=False, xgboost=True, pkl=False, kfolder=kfolder)
	print ''


if __name__ == '__main__':
	questions = load_texts(data_folder + '1_reorder/question_info.txt', 2)
	users = load_texts(data_folder + '1_reorder/user_info.txt', 2)
	g = NgramCoutingFeatureGenerator(questions, users)
	for i in xrange(-1, config.kfolder):
		gen_feature(g, kfolder=i)
