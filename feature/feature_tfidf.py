# -*- coding: utf-8 -*-

import pickle

import numpy as np
# import xgboost as xgb

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from scipy.sparse import hstack, vstack

from sklearn.datasets import dump_svmlight_file


data_folder = '../data/'


def load_texts(fname, column):
	texts = []
	with open(fname, 'r') as fp:
		for line in fp:
			items = line.strip().split('\t')
			texts.append(items[column].replace('/', ' '))
	print texts[0]
	return texts

def trans():
	vectorizer = TfidfVectorizer(min_df=3, ngram_range=(1,1), analyzer='word',token_pattern=r'\w{1,}', use_idf=1,smooth_idf=1,sublinear_tf=1)
	questions = load_texts(data_folder + '1_reorder/question_info.txt', 2)
	users = load_texts(data_folder + '1_reorder/user_info.txt', 2)
	texts = questions + users
	vectorizer.fit(texts)
	vq = vectorizer.transform(questions)
	vu = vectorizer.transform(users)
	dump_svmlight_file(vq, np.ones((vq.shape[0],)), './svmlight.txt')
	print vq.shape
	print vu.shape
	print '0', vq[0]
	print '1', vq[1]
	print '2', vq[2]
	print '3', vq[3]
	print '4', vq[4]
	print 'u0', vu[0]
	print 'hstack', hstack([vq[0], vu[0]])
	cosine_similarities = linear_kernel(vq[0], vq[1]).flatten()
	print cosine_similarities
	fo = open('./1_feature/tfidf_q.pkl', 'w')
	pickle.dump(vq, fo)
	fo.close()
	fo = open('./1_feature/tfidf_u.pkl', 'w')
	pickle.dump(vu, fo)
	fo.close()
	# xgb_model = xgb.XGBClassifier().fit(vq, np.ones((vq.shape[0], ), dtype='int32'))
	fo = open('./1_feature/tfidf_ngram.pkl', 'w')
	pickle.dump(vectorizer, fo)
	fo.close()


def trans_pair():
	fo = open('./1_feature/tfidf_q.pkl', 'r')
	vq = pickle.load(fo)
	fo.close()
	fo = open('./1_feature/tfidf_u.pkl', 'r')
	vu = pickle.load(fo)
	fo.close()
	features = []
	labels = []
	with open(data_folder + '1_reorder/invited_info_train.txt', 'r') as fp:
		for line in fp:
			q, u, label = map(int, line.strip().split('\t'))
			labels.append(label)
			features.append(hstack([vq[q], vu[u], linear_kernel(vq[0], vq[1])]))
	features = vstack(features)
	labels = np.array(labels, dtype='float32')
	dump_svmlight_file(features, labels, './1_feature/train.tfidf.xgboost.txt')
	test(features, labels)
	features = []
	with open(data_folder + '1_reorder/validate_nolabel.txt', 'r') as fp:
		for line in fp:
			q, u = map(int, line.strip().split('\t'))
			features.append(hstack([vq[q], vu[u], linear_kernel(vq[0], vq[1])]))
	features = vstack(features)
	dump_svmlight_file(features, np.ones((features.shape[0],)), './1_feature/test.tfidf.xgboost.txt')


def test(X, y):
	from sklearn.linear_model import LogisticRegression
	model = LogisticRegression(penalty='l2', dual=True, tol=0.0005,
                           C=10.0, fit_intercept=True, intercept_scaling=2.0,
                           class_weight='auto', random_state=1981)
	# from sklearn.svm import SVC
	# model = SVC(C=1.0, kernel='linear', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False,
	# 	tol=0.001, cache_size=200, class_weight=None, verbose=True, max_iter=-1, random_state=None)
	from sklearn import cross_validation
	print "5 Fold CV Test start"
	score = np.mean(cross_validation.cross_val_score(model, X, y, cv=5, verbose=1, scoring='mean_squared_error'))
	print "5 Fold CV Score: ", score




if __name__ == '__main__':
	# trans()
	trans_pair()
