# -*- coding: utf-8 -*-


# 统计一下问题的平均vote/ans/bestans，以及相互的比重，并把三个值得dict保存
def question_numerical(back=True):
	q2vote = {}
	q2ans = {}
	q2bestans = {}
	vote_count, ans_count, bestans_count = {}, {}, {}
	avg_vote, avg_ans, avg_bestans = 0, 0, 0
	avg_ans2vote, avg_bestans2vote, avg_bestans2ans = 0, 0, 0
	total = 0
	with open('./1_reorder/question_info.txt', 'r') as fp:
		for line in fp:
			id, tags, words, chars, vote, ans, bestans = line.strip().split('\t')
			id = int(id)
			q2vote[id] = int(vote)
			q2ans[id] = int(ans)
			q2bestans[id] = int(bestans)
			total += 1
			vote, ans, bestans = int(vote), int(ans), int(bestans)
			avg_vote += vote
			avg_ans += ans
			avg_bestans += bestans
			avg_ans2vote += (float(ans) / vote) if vote != 0 else 0
			avg_bestans2vote += (float(bestans) / vote) if vote != 0 else 0
			avg_bestans2ans += (float(bestans) / ans) if ans != 0 else 0
			if vote in vote_count:
				vote_count[vote] += 1
			else:
				vote_count[vote] = 1
			if ans in ans_count:
				ans_count[ans] += 1
			else:
				ans_count[ans] = 1
			if bestans in bestans_count:
				bestans_count[bestans] += 1
			else:
				bestans_count[bestans] = 1
	if not back:
		return q2vote, q2ans, q2bestans
	total = float(total)
	# 972.174675726 40.780852378 9.6213712168
	print avg_vote / total, avg_ans / total, avg_bestans / total
	# 0.232499018085 0.112964930886 0.52957733885
	print avg_ans2vote / total, avg_bestans2vote / total, avg_bestans2ans / total
	_stat1_sorted_and_back(vote_count, './stat/vote_count.txt')
	_stat1_sorted_and_back(ans_count, './stat/ans_count.txt')
	_stat1_sorted_and_back(bestans_count, './stat/bestans_count.txt')
	sorted_q2vote = sorted(q2vote.iteritems(), key=lambda d: d[0], reverse=False)
	with open('./stat/question_info.txt', 'w') as fo:
		for q, l in sorted_q2vote:
			fo.write('%d\t%d\t%d\t%d\n' % (q, l, q2ans[q], q2bestans[q]))
	return q2vote, q2ans, q2bestans


def _stat1_sorted_and_back(dict, filename):
	sorted_dict = sorted(dict.iteritems(), key=lambda d: d[0], reverse=False)
	with open(filename, 'w') as fo:
		for name, count in sorted_dict:
			fo.write('%d\t%d\n' % (name, count))


# 统计一下专家的，回答问题的平均vote，ans，bestans，不回答问题的平均vote，ans，bestans
def user_numerical(back=True, kfolder=-1):
	trainf = './1_reorder/invited_info_train.txt' if kfolder == -1 else './1_reorder/Folder%d/train.txt' % kfolder
	backf = './stat/user_info.txt' if kfolder == -1 else './stat/user_info_Folder%d.txt' % kfolder
	print 'start counting...'
	q2vote, q2ans, q2bestans = question_numerical(back=False)
	# 对每个user记录： 回答个数, vote, ans, bestans, 不回答个数, vote, ans, bestans
	user2infos = {}
	with open(trainf, 'r') as fp:
		for line in fp:
			q, u, label = map(int, line.strip().split('\t'))
			if u not in user2infos:
				if label:
					user2infos[u] = [1, q2vote[q], q2ans[q], q2bestans[q]] + [0] * 4
				else:
					user2infos[u] = [0] * 4 + [1, q2vote[q], q2ans[q], q2bestans[q]]
			else:
				if label:
					base = 0
				else:
					base = 4
				user2infos[u][base + 0] += 1
				user2infos[u][base + 1] += q2vote[q]
				user2infos[u][base + 2] += q2ans[q]
				user2infos[u][base + 3] += q2bestans[q]
	print 'start averaging...'
	zero_ans, zero_ignore = 0, 0
	for u in user2infos:
		# 对回答的问题求平均值
		if user2infos[u][0] != 0:
			total = float(user2infos[u][0])
			user2infos[u][1] /= total
			user2infos[u][2] /= total
			user2infos[u][3] /= total
		else:
			zero_ans += 1
		# 对不回答问题求平均值
		if user2infos[u][4] != 0:
			total = float(user2infos[u][4])
			user2infos[u][5] /= total
			user2infos[u][6] /= total
			user2infos[u][7] /= total
		else:
			zero_ignore += 1
		# 添加回答率feature
		if user2infos[u][0] != 0:
			user2infos[u].append(float(user2infos[u][0]) / (user2infos[u][0] + user2infos[u][4]))
		else:
			user2infos[u].append(0)
	if back:
		print 'start backup...'
		sorted_dict = sorted(user2infos.iteritems(), key=lambda d: d[0], reverse=False)
		last_u = -1
		with open(backf, 'w') as fo:
			for u, l in sorted_dict:
				# 有些user没有被推送过信息
				if u != last_u + 1:
					for i in xrange(last_u + 1, u):
						fo.write('%d\t0\t0\t0\t0\t0\t0\t0\t0\t0\n' % i)
				fo.write('%d\t%s\n' % (u, '\t'.join(map(str, l))))
				last_u = u
	print zero_ans, zero_ignore, len(user2infos)
	if kfolder == -1:
		print user2infos[0]
		print user2infos[1]
		print user2infos[2]
		print user2infos[3]
		print user2infos[4]
		print user2infos[5]
		print user2infos[12]
		print user2infos[223]
		print user2infos[2354]
		print user2infos[637]
	return user2infos


if __name__ == '__main__':
    question_numerical()
    for i in xrange(-1, 10):
    	user_numerical(back=True, kfolder=i)
