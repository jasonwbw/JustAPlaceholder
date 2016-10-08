# -*- coding: utf-8 -*-

import os

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


if __name__ == '__main__':
    gen_kfolder('./1_reorder/invited_info_train.txt', './1_reorder/', k=10)
