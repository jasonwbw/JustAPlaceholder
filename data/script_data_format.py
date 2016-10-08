# -*- coding: utf-8 -*-

#  重新编号UID和QID
def reorder(fin, fout, fmap):
	print 'reorder %s to %s' % (fin, fout)
	maps = {}
	with open(fin, 'r') as fp:
		with open(fout, 'w') as fo:
			with open(fmap, 'w') as fo_map:
				for i, line in enumerate(fp):
					items = line.strip().split('\t', 1)
					fo.write("%d\t%s\n" % (i, items[1]))
					fo_map.write("%s\t%d\n" % (items[0], i))
					maps[items[0]] = str(i)
	return maps


def reorder_pairs(fin, fout, qmaps, umaps):
	print 'reorder %s to %s' % (fin, fout)
	train = False
	if 'train' in fin:
		train = True
	with open(fin, 'r') as fp:
		with open(fout, 'w') as fo:
			for line in fp:
				items = line.strip().split('\t')
				fo.write(qmaps[items[0]] + '\t' + umaps[items[1]] + '\t' + (items[2] if train else '') + '\n')


if __name__ == '__main__':
	qmaps = reorder('./0_raw/question_info.txt', './1_reorder/question_info.txt', './1_reorder/question_map.txt')
	umaps = reorder('./0_raw/user_info.txt', './1_reorder/user_info.txt', './1_reorder/user_map.txt')
	reorder_pairs('./0_raw/invited_info_train.txt', './1_reorder/invited_info_train.txt', qmaps, umaps)
	reorder_pairs('./0_raw/validate_nolabel.txt', './1_reorder/validate_nolabel.txt', qmaps, umaps)
