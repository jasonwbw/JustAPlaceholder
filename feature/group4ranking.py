# -*- coding: utf-8 -*-


from itertools import izip


def group_feature4ranking(fdata, ffeature, fout, fgroupout, onlinetest=False, load4test=False, filter4rankloss=True):
    grouped = []
    with open(fdata, 'r') as fp:
        with open(ffeature, 'r') as fp_feature:
            for data, feature in izip(fp, fp_feature):
            	if onlinetest:
            		qid, uid = data.strip().split()
            		label = 0
            	else:
            		qid, uid, label = data.strip().split()
                grouped.append((int(qid), int(label), feature.strip()))
    grouped = sorted(grouped)
    lefted_groups = 0
    with open(fout, 'w') as fo:
        with open(fgroupout, 'w') as fo_group:
            current_group = []
            last_qid = -1
            contain_label_one = False
            not_contain_label_zero = True
            for qid, label, feature in grouped:
                if qid != last_qid:
                    if last_qid != -1 and ((contain_label_one and not not_contain_label_zero) or load4test or not filter4rankloss):
                        fo.write('\n'.join(current_group) + '\n')
                        fo_group.write('%d\n' % len(current_group))
                        lefted_groups += 1
                    contain_label_one = False
                    last_qid = qid
                    current_group = []
                if label == 1:
                    contain_label_one = True
                else:
                	not_contain_label_zero = False
                current_group.append(feature)
            if (contain_label_one and not not_contain_label_zero) or load4test or not filter4rankloss:
                fo.write('\n'.join(current_group) + '\n')
                fo_group.write('%d\n' % len(current_group))
    return lefted_groups


def group_data(featurename, kfolder=-1, filter4rankloss=True):
    print 'format %s - folder %d' % (featurename, kfolder)
    if kfolder == -1:
        ftrain = '../data/1_reorder/invited_info_train.txt'
        ftrain_feature = './feature/%s.train.xgboost.txt' % featurename
        ftest = '../data/1_reorder/validate_nolabel.txt'
        ftest_feature = './feature/%s.test.xgboost.txt' % featurename
    else:
        ftrain = '../data/1_reorder/Folder%d/train.txt' % kfolder
        ftrain_feature = './feature/Folder%d/%s.train.xgboost.txt' % (
            kfolder, featurename)
        ftest = '../data/1_reorder/Folder%d/val.txt' % kfolder
        ftest_feature = './feature/Folder%d/%s.test.xgboost.txt' % (
            kfolder, featurename)
    lefted_groups = group_feature4ranking(ftrain, ftrain_feature, ftrain_feature.replace(
        '.txt', '.4rank.txt'), ftrain_feature.replace('.txt', '.4rank.txt.group'), onlinetest=False, filter4rankloss=filter4rankloss)
    print '%d groups lefted for train' % lefted_groups
    lefted_groups = group_feature4ranking(ftest, ftest_feature, ftest_feature.replace(
        '.txt', '.4rank.txt'), ftest_feature.replace('.txt', '.4rank.txt.group'), onlinetest=(kfolder == -1), load4test=True, filter4rankloss=filter4rankloss)
    print '%d groups lefted for test' % lefted_groups
    print ''


if __name__ == '__main__':
    # merge stat and tags
    group_data('merge.stat_tags', filter4rankloss=False)
    for i in xrange(10):
        group_data('merge.stat_tags', kfolder=i, filter4rankloss=False)
