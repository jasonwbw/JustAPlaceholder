# -*- coding: utf-8 -*-

import os
import config

import random

import numpy as np
from sklearn.cross_validation import StratifiedKFold


def gen_kfolder(fname, fo_prefix, k=10):
	samples = []
	labels = []
	with open(fname, 'r') as fp:
		for line in fp:
			samples.append(line)
			labels.append(int(line.strip().split('\t')[-1]))
	random_seed = 2016
	skf = StratifiedKFold(labels, n_folds=k,
		shuffle=True, random_state=random_seed)
	for fold, (trainInd, validInd) in enumerate(skf):
		print("================================")
		print("Index for run: %s, fold: %s" % (0, fold + 1))
		print("Train (num = %s)" % len(trainInd))
		print(trainInd[:10])
		print("Valid (num = %s)" % len(validInd))
		print(validInd[:10])
		if not os.path.exists(fo_prefix + 'Folder%d' % fold):
			os.makedirs(fo_prefix + 'Folder%d' % fold)
		with open(fo_prefix + 'Folder%d/train.txt' % fold, 'w') as fo:
			for idx in trainInd:
				fo.write(samples[idx])
		with open(fo_prefix + 'Folder%d/val.txt' % fold, 'w') as fo:
			for idx in validInd:
				fo.write(samples[idx])


def gen_kfolder_with_labelone_in_rank(fname, fo_prefix, k=10):
	q2count = {}
	q2count_label_one = {}
	q2label_one = {}
	q2label_zero = {}
	with open(fname, 'r') as fp:
		for line in fp:
			q, u, label = map(int, line.strip().split('\t'))
			if q not in q2count_label_one:
				q2count_label_one[q] = 0
				q2label_one[q] = []
			if label == 1:
				q2count_label_one[q] += 1
				q2label_one[q].append(u)
			else:
				if q not in q2label_zero:
					q2label_zero[q] = [u]
				else:
					q2label_zero[q].append(u)
			if q not in q2count:
				q2count[q] = 1
			else:
				q2count[q] += 1
	sample4label_one = []
	count2idxes = {}
	max_count = 0
	for q, count in q2count_label_one.items():
		sample4label_one.append((count, q2count[q], q))
		if count not in count2idxes:
			count2idxes[count] = [len(sample4label_one) - 1]
		else:
			count2idxes[count].append(len(sample4label_one) - 1)
		max_count = max(max_count, count)
	print 'max_count', max_count
	for fold in xrange(k):
		print 'Folder %d from %d' % (fold, len(sample4label_one))
		choosen_q = set()
		train, test = [], []
		backed_idxes = set()
		# # move some question to test (as the total 8094 question have 386 just in test set)
		# as just total 81 case (some just one) is unique, we just remove the items
		count_just4test = 0
		# test_move_idxes = np.random.random_integers(0, len(sample4label_one) - 1, count_just4test)
		# for t in test_move_idxes:
		# 	backed_idxes.add(t)
		# 	count, total, q = sample4label_one[t]
		# 	for i, u in enumerate(q2label_one[q]):
		# 		test.append('%d\t%d\t1\n' % (q, u))
		# 	if q in q2label_zero:
		# 		for i, u in enumerate(q2label_zero[q]):
		# 			test.append('%d\t%d\t0\n' % (q, u))
		# 	choosen_q.add(q)
		# move some question to test (as the total 8094 question have 2410 just in test set)
		train_move_idxes = np.random.random_integers(0, len(sample4label_one) - 1, 2298 + count_just4test)
		choosed = 0
		for t in train_move_idxes:
			choosed += 1
			if choosed > 2298:
				break
			count, total, q = sample4label_one[t]
			if q in choosen_q:
				continue
			backed_idxes.add(t)
			for i, u in enumerate(q2label_one[q]):
				train.append('%d\t%d\t1\n' % (q, u))
			if q in q2label_zero:
				for i, u in enumerate(q2label_zero[q]):
					train.append('%d\t%d\t0\n' % (q, u))
			choosen_q.add(q)
		# choose some with random choose same label_one rate to test
		test_idxes = np.random.random_integers(0, len(sample4label_one) - 1, (len(sample4label_one) - 2298 - count_just4test) * 3 / k)
		for t in test_idxes:
			count, total, q = sample4label_one[t]
			if count + 5 >= max_count:
				continue
			q = None
			c4out = 0
			while c4out < 30 and (q is None or q in choosen_q or q not in q2label_zero or len(q2label_zero[q]) < 2):
				c4out += 1
				if count == 0:
					new_count = 0
				else:
					new_count = random.randint(count + 1, max_count - 1)
					tc4out = 0
					while tc4out < min(10, max_count - count - 1) and new_count not in count2idxes:
						new_count = random.randint(count + 1, max_count - 1)
						tc4out += 1
					if tc4out >= min(10, max_count - count - 1):
						continue
				idxes = count2idxes[new_count]
				idx = idxes[random.randint(0, len(idxes) - 1)]
				q = sample4label_one[idx][2]
			if c4out >= 30:
				continue
			backed_idxes.add(idx)
			choosen_q.add(q)
			if new_count > 0:
				us = set(np.random.choice(new_count, count))
				for i, u in enumerate(q2label_one[q]):
					if i in us:
						test.append('%d\t%d\t1\n' % (q, u))
					else:
						train.append('%d\t%d\t1\n' % (q, u))
			count4chosen = 0 if len(q2label_zero[q]) < 2 else min(5, random.randint(1, len(q2label_zero[q]) - 1))
			us = set(np.random.choice(len(q2label_zero[q]), count4chosen))
			for i, u in enumerate(q2label_zero[q]):
				if i in us:
					test.append('%d\t%d\t0\n' % (q, u))
				else:
					train.append('%d\t%d\t0\n' % (q, u))
		# lefted
		for t in xrange(len(sample4label_one)):
			if t in backed_idxes:
				continue
			count, total, q = sample4label_one[t]
			for i, u in enumerate(q2label_one[q]):
				train.append('%d\t%d\t1\n' % (q, u))
			if q in q2label_zero:
				for i, u in enumerate(q2label_zero[q]):
					train.append('%d\t%d\t0\n' % (q, u))
		print len(train)
		print len(test)
		if not os.path.exists(fo_prefix + 'Folder%d' % fold):
			os.makedirs(fo_prefix + 'Folder%d' % fold)
		with open(fo_prefix + 'Folder%d/train.txt' % fold, 'w') as fo:
			for s in train:
				fo.write(s)
		with open(fo_prefix + 'Folder%d/val.txt' % fold, 'w') as fo:
			for s in test:
				fo.write(s)
		print ''

if __name__ == '__main__':
    # gen_kfolder('./1_reorder/invited_info_train.txt', './1_reorder/', k=config.kfolder)
    gen_kfolder_with_labelone_in_rank('./1_reorder/invited_info_train.txt', './1_reorder/', k=config.kfolder)
