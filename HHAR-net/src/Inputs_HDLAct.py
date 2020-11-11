# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 12:57:42 2019

@author: 14342
"""

data_dir = "/Users/esestaff/Documents/GitHub/Hierarchical-DNN-Activity-Recognition/datasets/"
cvdir = "/Users/esestaff/Documents/GitHub/Hierarchical-DNN-Activity-Recognition/cv_5_folds/"



nb_epoch = 50
nb_batch = 10
val_split = 0.05
drop_out_per = 0.3
test_split = 0.2



modeling_style = 'LOO' # or 'GEN'


#combinationset = [('Acc'), ('Gyro'), ('Mag'), ('W_acc'), ('Loc'), ('Aud'), ('PS'), ('Acc', 'W_acc'), ('Acc', 'W_acc', 'Aud'), \
# ('Acc', 'W_acc', 'Gyro'), ('Acc', 'W_acc', 'Aud', 'Gyro')]
#sensors_to_use = ['Acc', 'Gyro','Mag']
#sensors_to_use = ['Acc', 'W_acc']
sensors_to_use = ['Acc', 'W_acc', 'Aud', 'Gyro', 'Loc', 'Mag', 'PS', 'Compass', 'AP']
#sensors_to_use = ['Acc', 'W_acc', 'Gyro']
#sensors_to_use = ['Acc', 'W_acc', 'Loc']
#sensors_to_use = ['Acc', 'W_acc', 'Aud', 'Gyro']



# create combination sets

#from itertools import chain, combinations
#
#def powerset(iterable):
#    s = list(iterable)
#    return list(chain.from_iterable(combinations(s, r) for r in range(2)))
#
#combinationset = powerset(sensors_to_use)
#combinationset.pop(0)