# -*- coding: utf-8 -*-
# 使用tag matching的信息
#
# Feature List:
# T - 0/1   - q/u是否有完全match的tag
# F - float - sum（q的tag u回答过，在所有回答中的占比）
# F - float - sum(q的tag u拒绝过，在所有拒绝中的占比)

import os
import config
from feature_abstract import FeatureGenerator

from itertools import izip

from collections import Counter

import numpy as np

data_folder = '../data/'


def load_tags():
    # 加载question和user对应的tag列表
    print 'load tags'
    qtags = {}
    utags = {}
    qtag_set = set()
    utag_set = set()
    with open(data_folder + '1_reorder/question_info.txt', 'r') as fp:
        for line in fp:
            q, tags, w, c, x1, x2, x3 = line.strip().split('\t')
            tags = set(map(int, tags.split('/')))
            qtags[int(q)] = tags
            qtag_set = qtag_set.union(tags)
    with open(data_folder + '1_reorder/user_info.txt', 'r') as fp:
        for line in fp:
            try:
                u, tags, w, c = line.strip().split('\t')
            except:
                u, tags, wc = line.strip().split('\t')
            tags = set(map(int, tags.split('/')))
            utags[int(u)] = tags
            utag_set = utag_set.union(tags)
    print 'total %d qtag and %d utag' % (len(qtag_set), len(utag_set))
    return qtags, utags, len(qtag_set), len(utag_set)


def user2ans_tag(qtags, utags, kfolder=-1):
    print 'compute tag rate in uset\'s answered and rejected question'
    # user 到 每个tag的出现比例(分别在回答的和拒绝的中)
    u2tags_ans_rate = {}
    u2tags_ans_count = {}
    u2tags_reject_rate = {}
    u2tags_reject_count = {}
    if kfolder == -1:
        data_file = data_folder + '1_reorder/invited_info_train.txt'
    else:
        data_file = data_folder + '1_reorder/Folder%d/train.txt' % kfolder
    with open(data_file, 'r') as fp:
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
    return u2tags_ans_rate, u2tags_reject_rate, u2tags_ans_count, u2tags_reject_count


def load_qtag2utag(qtags, utags, kfolder=-1):
    '''对每个qtag，输出了utag 回答/推送到次数 的比率, 剔除了回答少于5次的'''
    if kfolder == -1:
        data_file = data_folder + '1_reorder/invited_info_train.txt'
    else:
        data_file = data_folder + '1_reorder/Folder%d/train.txt' % kfolder
    qt2ut_ansed = {}
    qt2ut_count = {}
    with open(data_file, 'r') as fp:
        for line in fp:
            q, u, label = map(int, line.strip().split('\t'))
            for qt in qtags[q]:
                if qt not in qt2ut_ansed:
                    qt2ut_ansed[qt] = Counter()
                    qt2ut_count[qt] = Counter()
                for ut in utags[u]:
                    if label == 1:
                        qt2ut_ansed[qt][ut] += 1
                    qt2ut_count[qt][ut] += 1
    qtag2utag = {}
    for qt in qt2ut_ansed:
        qtag2utag[qt] = {}
        for ut, ans_count in qt2ut_ansed[qt].items():
            if ans_count >= 5:
                qtag2utag[qt][ut] = float(ans_count) / qt2ut_count[qt][ut]
    return qtag2utag


class TagsFeatureGenerator(FeatureGenerator):

    def __init__(self, qtags, utags, qtag_size, utag_size,
                 u2tags_ans_rate, u2tags_reject_rate,
                 u2tags_ans_count, u2tags_reject_count, qtag2utag,
                 feature_choosen=[False, False, False, False, True]):
        self.qtags = qtags
        self.utags = utags
        self.u2tags_ans_rate = u2tags_ans_rate
        self.u2tags_reject_rate = u2tags_reject_rate
        self.u2tags_ans_count = u2tags_ans_count
        self.u2tags_reject_count = u2tags_reject_count
        self.feature_choosen = feature_choosen
        self.qtag_size = qtag_size
        self.utag_size = utag_size
        self.qtag2utag = qtag2utag

    def gen_feauture(self, q, u):
        # [抛弃][发现tag不是一个体系]
        # tag 是不是完全match
        tag_match = 0
        ptag = 0
        for tag in self.qtags[q]:
            ptag = tag
            if tag in self.utags[u]:
                tag_match = 1
                break
        f1 = [tag_match]
        # [抛弃][评价体系不是很清晰，有点想当然]
        # tag 用户回答/拒绝的热度
        tag_ans_rate, tag_reject_rate = 0., 0.
        for tag in self.qtags[q]:
            if u in self.u2tags_ans_rate and tag in self.u2tags_ans_rate[u]:
                tag_ans_rate += self.u2tags_ans_rate[u][tag]
            if u in self.u2tags_reject_rate and tag in self.u2tags_reject_rate[u]:
                tag_reject_rate += self.u2tags_reject_rate[u][tag]
        f2 = [tag_ans_rate, tag_reject_rate]
        # [抛弃][发现回答和拒绝往往是一起发生的]
        # tag 用户是否回答过、拒绝过，回答拒绝的次数
        tag_ansed, tag_rejected = 0, 0
        for tag in self.qtags[q]:
            if u in self.u2tags_ans_rate and tag in self.u2tags_ans_rate[u]:
                tag_ansed += int(self.u2tags_ans_rate[u]
                                 [tag] * self.u2tags_ans_count[u])
            if u in self.u2tags_reject_rate and tag in self.u2tags_reject_rate[u]:
                tag_rejected += int(self.u2tags_reject_rate[u][
                                    tag] * self.u2tags_reject_count[u])
        f3 = [tag_ansed, tag_rejected, 1 if tag_ansed >
              0 else 0, 1 if tag_rejected > 0 else 0]
        # 直接使用raw的q、u的tag编号来作为feature
        f4 = [0] * (self.qtag_size + self.utag_size)
        for tag in self.qtags[q]:
            f4[tag] = 1
        for tag in self.utags[u]:
            f4[tag + self.qtag_size] = 1
        # 对每个qtag，输出了utag 回答/推送到次数 的比率
        probs = []
        for utag in self.utags[u]:
            if utag in self.qtag2utag[ptag]:
                probs.append(self.qtag2utag[ptag][utag])
        if len(probs) == 0:
            f5 = [0, 0]
        else:
            f5 = [max(probs), np.mean(probs)]
        # combine
        candidates = [f1, f2, f3, f4, f5]
        fea = []
        for f, chosen in izip(candidates, self.feature_choosen):
            if chosen:
                fea += f
        return fea
# end TagsFeatureGenerator


def gen_feature(kfolder=-1):
    # 生成Feature
    if kfolder != -1:
        print 'current fit %d folder' % kfolder
    # preload info
    qtags, utags, qtag_size, utag_size = load_tags()
    u2tags_ans_rate, u2tags_reject_rate, u2tags_ans_count, u2tags_reject_count = user2ans_tag(
        qtags, utags, kfolder=kfolder)
    qtag2utag = load_qtag2utag(qtags, utags, kfolder=kfolder)
    if kfolder != -1 and not os.path.exists('./feature/Folder%d' % kfolder):
        os.makedirs('./feature/Folder%d' % kfolder)
    g = TagsFeatureGenerator(qtags, utags, qtag_size, utag_size,
                             u2tags_ans_rate, u2tags_reject_rate, u2tags_ans_count, u2tags_reject_count, qtag2utag)
    feature_folder = './feature/' if kfolder == - \
        1 else './feature/Folder%d/' % kfolder
    print 'start generating feature for training data'
    g.gen_feature_file(feature_folder, 'tags', train=True,
                       xgboost=True, pkl=False, kfolder=kfolder)
    print 'start generating feature for test data'
    g.gen_feature_file(feature_folder, 'tags', train=False,
                       xgboost=True, pkl=False, kfolder=kfolder)
    print ''


if __name__ == '__main__':
    for i in xrange(-1, config.kfolder):
        gen_feature(i)
