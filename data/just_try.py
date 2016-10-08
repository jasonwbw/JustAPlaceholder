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


if __name__ == '__main__':
	group_by_question()
