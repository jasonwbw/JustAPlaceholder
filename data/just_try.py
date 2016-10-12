# -*- coding: utf-8 -*-


def group_by_question():
	'''
	把label数据按照question做一下group
	'''
	groups = {}
	groups_contain_true = {}
	with open('./1_reorder/invited_info_train.txt', 'r') as fp:
		for line in fp:
			q, u, label = map(int, line.strip().split('\t'))
			if q in groups:
				groups[q].append((u, label))
			else:
				groups[q] = [(u, label)]
				groups_contain_true[q] = False
			if label:
				groups_contain_true[q] = True
	with open('group_user2question.txt', 'w') as fo:
		for q in groups:
			fo.write('%d: %s\n' % (q, ' '.join([str(item[1]) for item in groups[q]])))


def find_valid_not_in_train():
	from collections import Counter
	qs = set()
	with open('./1_reorder/invited_info_train.txt', 'r') as fp:
		for line in fp:
			q, u, label = map(int, line.strip().split('\t'))
			qs.add(q)
	new_qs = Counter()
	total = 0
	with open('./1_reorder/validate_nolabel.txt', 'r') as fp:
		for line in fp:
			q, u = map(int, line.strip().split('\t'))
			if q not in qs:
				total += 1
				new_qs[q] += 1
	with open('validate_newq.txt', 'w') as fo:
		fo.write('total %d\n' % total)
		for q, s in new_qs.items():
			fo.write('%d:\t%d\n' % (q, s))


if __name__ == '__main__':
	# group_by_question()
	find_valid_not_in_train()
