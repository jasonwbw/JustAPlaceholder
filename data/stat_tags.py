# -*- coding: utf-8 -*-
# 使用tag matching的信息


from collections import Counter


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


def _try_basic_infos(k2tags):
	tag_distribution = {}
	for k, tags in k2tags.items():
		for tag in tags:
			if tag not in tag_distribution:
				tag_distribution[tag] = []
			tag_distribution[tag].append(k)
	tag_len = [len(dis) for tag, dis in tag_distribution.items()]
	print 'There are total %d type of tags' % len(tag_distribution)
	print 'Each tag have been used %d(min), %d(max), %d(avg)' % (min(tag_len), max(tag_len), sum(tag_len) / float(len(tag_len)))


def try_basic_infos():
    '''统计tag的一些基本情况'''
    print '统计tag match与label的关系'
    qtags, utags = load_tags()
    print ''
    print 'For total %d question:' % len(qtags)
    _try_basic_infos(qtags)
    print ''
    print 'For total %d user:' % len(utags)
    _try_basic_infos(utags)
# end try_basic_infos
#
# 结果:
# For total 8095 question:
# There are total 20 type of tags
# Each tag have been used 108(min), 837(max), 404(avg)
#
# For total 28763 user:
# There are total 143 type of tags
# Each tag have been used 2(min), 3332(max), 402(avg)


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
    print 'Label 1 rate with matching tags %f, %d(Label 1), %d(Total Matching)' %\
        (float(tag_match_with_label1) / tag_match_count,
         tag_match_with_label1, tag_match_count)
# end try_tag_matching
#
# 结果:
# Label 1 rate with matching tags 0.230874, 338(Label 1), 1464(Total Matching)
#
# 分析:
# 该结果无意义，结合try_basic_infos，发现两边的tag编号应该不是一致的。q的tag很少


def try_tag_relevant():
    '''统计q的tag到u的tag相关关系（根据回答的推送）'''
    print '统计q的tag到u的tag相关关系（根据回答的推送）'
    qtags, utags = load_tags()
    # 回答过qtag的utag分布
    qt2ut_ansed = {}
    # qt被回答的次数
    qt2count = {}
    # qtag推送到utag次数
    qt2ut_count = {}
    with open('./1_reorder/invited_info_train.txt', 'r') as fp:
        for line in fp:
            q, u, label = map(int, line.strip().split('\t'))
            for qt in qtags[q]:
            	if qt not in qt2ut_ansed:
            		qt2ut_ansed[qt] = Counter()
            		qt2count[qt] = 0.
            		qt2ut_count[qt] = Counter()
            	if label == 1:
            		qt2count[qt] += 1
            	for ut in utags[u]:
            		if label == 1:
            			qt2ut_ansed[qt][ut] += 1
            		qt2ut_count[qt][ut] += 1
    qt2ut_ans_rate = {}
    for qt, c in qt2ut_ansed.items():
    	# print qt, int(qt2count[qt])
    	# print [(key, value, '%.2f' % (value / qt2count[qt])) for key, value in c.most_common(10)]
    	qt2ut_ans_rate[qt] = sorted([(float(value) / qt2ut_count[qt][key], key, value, qt2ut_count[qt][key]) for key, value in c.items() if value >= 5], reverse=True)
    for qt, l in qt2ut_ans_rate.items():
    	print qt
    	print [(key, '%.2f' % value, v2, v3) for value, key, v2, v3 in l][:10]
    	print ''
# end
#
# 结果:
# 1. 通过输出qtag对应的top 回答/拒绝的utag，发现
#    对于qtag，推送到的utag基本比较固定，所以回答的和拒绝的基本分布差不多。
#    除去个别tag可能稍有差别，这部分应该是更有用的。
# 2. [Feature]
#    根据第一条发现，输出了utag 回答/推送到次数 的比率，来观察一下top的tag
#    并针对总推送次数低的排名靠前的问题，剔除了回答少于5次的
#    这个也许可以作为一个feature
# 3. [Feature]
#    Feature维度不多，也许而已直接把tag的onehot作为feature


if __name__ == '__main__':
	# try_basic_infos()
    # try_tag_matching()
    # try_tag_relevant()
    # try_q2tag()
