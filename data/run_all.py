# -*- coding: utf-8 -*-

import os

#######################
# Generate features
#######################
# gen kfolder for train
cmd = "python ./gen_kfolder.py"
print cmd
os.system(cmd)

# stat about feature pre generation
cmd = "python ./stat.py"
print cmd
os.system(cmd)

