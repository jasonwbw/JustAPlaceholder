# -*- coding: utf-8 -*-
# 使用tag matching的信息


def load_tags():
	# 加载question和user对应的tag列表
	print 'load tags'
	qtags = {}
	utags = {}
	with open('./1_reorder/question_info.txt', 'r') as fp:
		for line in fp:
			q, tags, w, c, x1, x2, x3 = line.strip().split('\t')
			qtags[int(q)] = set(map(int, tags.split('/')))
	with open('./1_reorder/user_info.txt', 'r') as fp:
		for line in fp:
			try:
				u, tags, w, c = line.strip().split('\t')
			except:
				u, tags, wc = line.strip().split('\t')
			utags[int(u)] = set(map(int, tags.split('/')))
	return qtags, utags


def try_tag_matching():
	'''统计tag match与label的关系'''
	print '统计tag match与label的关系'
	qtags, utags = load_tags()
	tag_match_count, tag_match_with_label1 = 0, 0
	with open('./1_reorder/invited_info_train.txt', 'r') as fp:
		for line in fp:
			q, u, label = map(int, line.strip().split('\t'))
			for tag in qtags[q]:
				if tag in utags[u]:
					# print q, u, label
					if label == 1:
						tag_match_with_label1 += 1
					tag_match_count += 1
					break
	print 'Label 1 rate with matching tags %f, %d(Label 1), %d(Total Matching)' % (float(tag_match_with_label1) / tag_match_count, tag_match_with_label1, tag_match_count)


if __name__ == '__main__':
	try_tag_matching()
