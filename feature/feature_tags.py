# -*- coding: utf-8 -*-
# 使用tag matching的信息


from feature_abstract import FeatureGenerator

data_folder = '../data/'


def load_tags():
	# 加载question和user对应的tag列表
	print 'load tags'
	qtags = {}
	utags = {}
	with open(data_folder + '1_reorder/question_info.txt', 'r') as fp:
		for line in fp:
			q, tags, w, c, x1, x2, x3 = line.strip().split('\t')
			qtags[int(q)] = set(map(int, tags.split('/')))
	with open(data_folder + '1_reorder/user_info.txt', 'r') as fp:
		for line in fp:
			try:
				u, tags, w, c = line.strip().split('\t')
			except:
				u, tags, wc = line.strip().split('\t')
			utags[int(u)] = set(map(int, tags.split('/')))
	return qtags, utags


def user2ans_tag(qtags, utags):
	print 'compute tag rate in uset\'s answered and rejected question'
	# user 到 每个tag的出现比例(分别在回答的和拒绝的中)
	u2tags_ans_rate = {}
	u2tags_ans_count = {}
	u2tags_reject_rate = {}
	u2tags_reject_count = {}
	with open(data_folder + '1_reorder/invited_info_train.txt', 'r') as fp:
		for line in fp:
			q, u, label = map(int, line.strip().split('\t'))
			occupy = u2tags_ans_rate if label == 1 else u2tags_reject_rate
			count_occupy = u2tags_ans_count if label == 1 else u2tags_reject_count
			if u not in occupy:
				occupy[u] = {}
				count_occupy[u] = 0
			for tag in qtags[q]:
				if tag in occupy[u]:
					occupy[u][tag] += 1
				else:
					occupy[u][tag] = 1
				count_occupy[u] += 1
	for u in u2tags_ans_rate:
		count = float(u2tags_ans_count[u])
		for tag in u2tags_ans_rate[u]:
			u2tags_ans_rate[u][tag] /= count
	for u in u2tags_reject_rate:
		count = float(u2tags_reject_count[u])
		for tag in u2tags_reject_rate[u]:
			u2tags_reject_rate[u][tag] /= count
	return u2tags_ans_rate, u2tags_reject_rate


class TagsFeatureGenerator(FeatureGenerator):

	def __init__(self, qtags, utags, u2tags_ans_rate, u2tags_reject_rate):
		self.qtags = qtags
		self.utags = utags
		self.u2tags_ans_rate = u2tags_ans_rate
		self.u2tags_reject_rate = u2tags_reject_rate

	def gen_feauture(self, q, u):
		# tag 是不是完全match
		tag_match = 0
		for tag in self.qtags[q]:
			if tag in self.utags[u]:
				tag_match = 1
				break
		# tag 用户回答/拒绝的热度
		tag_ans, tag_reject = 0., 0.
		for tag in self.qtags[q]:
			if u in self.u2tags_ans_rate and tag in self.u2tags_ans_rate[u]:
				tag_ans += self.u2tags_ans_rate[u][tag]
			if u in self.u2tags_reject_rate and tag in self.u2tags_reject_rate[u]:
				tag_reject += self.u2tags_reject_rate[u][tag]
		fea = [tag_match, tag_ans, tag_reject]
		return fea
# end StatFeatureGenerator


if __name__ == '__main__':
	# 生成Feature
	qtags, utags = load_tags()
	u2tags_ans_rate, u2tags_reject_rate = user2ans_tag(qtags, utags)
	g = TagsFeatureGenerator(qtags, utags, u2tags_ans_rate, u2tags_reject_rate)
	print 'start generating feature for training data'
	g.gen_feature_file('./feature/', 'tags', train=True, xgboost=True, pkl=True)
	print 'start generating feature for test data'
	g.gen_feature_file('./feature/', 'tags', train=False, xgboost=True, pkl=True)
