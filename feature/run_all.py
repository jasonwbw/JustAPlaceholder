# -*- coding: utf-8 -*-


import os


#######################
# Generate features
#######################
# stat about
cmd = "python ./feature_stat.py"
print cmd
os.system(cmd)

# tags about
cmd = "python ./feature_tags.py"
print cmd
os.system(cmd)

# ngram about
cmd = "python ./feature_ngram.py"
print cmd
os.system(cmd)


#####################
# Combine Feature and Format for Ranking
#####################
# combine feat
cmd = "python ./merge_feature.py"
print cmd
os.system(cmd)

# combine feat
cmd = "python ./group4ranking.py"
print cmd
os.system(cmd)
